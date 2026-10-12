#!/usr/bin/env python3
"""Collect scripted-policy demonstrations from DexBioLab tasks.

The collector runs one or more registered DexBioLab task IDs with a scripted policy,
keeps a per-env trajectory buffer, and only writes episodes whose terminal
condition is `success`.

Output layout:

    <out_dir>/<task_id>/replay_buffer.zarr/
        data/action
        data/state
        data/timestamp
        replay/action
        replay/qpos
        replay/qvel
        ...
        meta/episode_ends
        meta/episode_env_origins
        meta/episode_prompts
        meta/goal_names          # variable-goal tasks only
        meta/episode_goals       # variable-goal tasks only

The `data/` group keeps policy state/action arrays for OpenPI rendering. The `replay/`
group stores the raw time-major simulation/action arrays needed to replay a
trajectory. Camera observations are disabled during collection; render saved
states later in a dedicated render/playback script.
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
from utils.replay_state import (
    extract_body_names,
    extract_replay_state,
    extract_task_replay_state,
)

from scripts.utils.collection_config import CollectConfig as CollectConfig
from scripts.utils.collection_goals import (
    BalancedGoalScheduler as BalancedGoalScheduler,
    EpisodeBuffer as EpisodeBuffer,
)
from scripts.utils.collection_writer import (
    _WUJI_HEATING_PLATE_TASK_ID as _WUJI_HEATING_PLATE_TASK_ID,
    _XHAND_BUNSEN_TASK_ID as _XHAND_BUNSEN_TASK_ID,
    _XHAND_TUBE_INSERT_TASK_ID as _XHAND_TUBE_INSERT_TASK_ID,
    _ZARR_COMPRESSOR as _ZARR_COMPRESSOR,
    StreamingZarrWriter as StreamingZarrWriter,
    chunks_for_shape as chunks_for_shape,
    create_empty_dataset as create_empty_dataset,
    data_arrays_from_source as data_arrays_from_source,
    episode_env_origin as episode_env_origin,
    episode_source_arrays as episode_source_arrays,
    find_first_non_static_frame as find_first_non_static_frame,
    normalize_time_major as normalize_time_major,
    replay_arrays_from_source as replay_arrays_from_source,
    select_state_key as select_state_key,
    stack_frames as stack_frames,
)
from src.utils.arrays import snapshot_to_numpy
from src.utils.environment import extract_success, extract_time_out

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is available in the uv lock.
    tqdm = None

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from play import (
    PolicyNoOp,
    PolicyRandom,
    reset_policy_after_env_done,
    use_standard_cameras_instead_of_gaussian,
)
from src.policies import make_scene_policy

import src.tasks  # noqa: F401 - register tasks
from src.tasks.task_goals import (
    default_goal_episodes_per_variant,
    extract_task_goal_values,
    install_task_goal_provider,
    task_goal_balance_mode,
    task_goal_names,
)

_MUJOCO_PREVIOUS_WARNING = None
_MUJOCO_WARNING_CALLBACK = None
_MUJOCO_WARNING_FILTER_INSTALLED = False
_CCD_ITERATIONS_WARNING = (
    "opt.ccd_iterations, currently set to 50, needs to be increased"
)


def split_env_value(value: Any, env_id: int, num_envs: int) -> Any:
    value = snapshot_to_numpy(value)
    if isinstance(value, np.ndarray) and value.shape[:1] == (num_envs,):
        return np.array(value[env_id], copy=True)
    return value


def is_camera_observation_group(name: str, group: Any) -> bool:
    lowered = name.lower()
    if "rgb" in lowered or "camera" in lowered or "cam" in lowered:
        return True

    for term in group.terms.values():
        func_name = getattr(term.func, "__name__", "")
        if func_name == "camera_rgb":
            return True
        params = term.params or {}
        sensor_name = str(params.get("sensor_name", "")).lower()
        if "rgb" in sensor_name or "camera" in sensor_name or "cam" in sensor_name:
            return True
    return False


def disable_camera_observations(env_cfg: Any) -> tuple[int, int]:
    """Remove camera observations and scene sensors so collection stays state-only."""
    observations = env_cfg.observations
    removed_obs = 0
    for name, group in list(observations.items()):
        if is_camera_observation_group(name, group):
            del observations[name]
            removed_obs += 1

    sensors = env_cfg.scene.sensors
    env_cfg.scene.sensors = ()

    return removed_obs, len(sensors)


def extract_task_goal(raw_env: ManagerBasedRlEnv, task_id: str) -> np.ndarray | None:
    values = extract_task_goal_values(raw_env, task_id)
    if values is None:
        return None
    return snapshot_to_numpy(values, name=f"{task_id}_episode_goal")


def install_warning_filter() -> None:
    """Hide the noisy MuJoCo CCD iteration warning during batch collection."""
    global \
        _MUJOCO_PREVIOUS_WARNING, \
        _MUJOCO_WARNING_CALLBACK, \
        _MUJOCO_WARNING_FILTER_INSTALLED

    warnings.filterwarnings(
        "ignore",
        message=r".*opt\.ccd_iterations.*needs to be increased.*",
    )

    if _MUJOCO_WARNING_FILTER_INSTALLED:
        return

    _MUJOCO_PREVIOUS_WARNING = mujoco.get_mju_user_warning()

    def _filtered_mujoco_warning(message: str) -> None:
        text = str(message)
        if _CCD_ITERATIONS_WARNING in text:
            return
        if _MUJOCO_PREVIOUS_WARNING is not None:
            _MUJOCO_PREVIOUS_WARNING(message)
        else:
            print(f"Warning: {text}", file=sys.stderr)

    _MUJOCO_WARNING_CALLBACK = _filtered_mujoco_warning
    mujoco.set_mju_user_warning(_MUJOCO_WARNING_CALLBACK)
    _MUJOCO_WARNING_FILTER_INSTALLED = True


def task_zarr_path(task_id: str, cfg: CollectConfig) -> Path:
    if cfg.zarr_path is not None:
        return cfg.zarr_path.expanduser().resolve()
    return (cfg.out_dir / task_id / "replay_buffer.zarr").expanduser().resolve()


def _make_policy(
    task_id: str,
    agent: str,
    action_shape: tuple[int, ...],
    device: str,
    env: ManagerBasedRlEnv,
    replay_path: str | None,
):
    if agent == "random":
        return PolicyRandom(action_shape, device)
    if agent == "slider":
        if action_shape[-1] == 0:
            return PolicyNoOp(action_shape, device)
        raise RuntimeError(
            "agent='slider' is interactive and is not supported by collect.py."
        )
    if agent == "auto":
        return make_scene_policy(
            task_id,
            action_shape=action_shape,
            device=device,
            env=env,
            server=None,
            replay_path=replay_path,
        )

    raise RuntimeError(f"Unsupported generic agent type: {agent!r}")


def collect_task(task_id: str, cfg: CollectConfig) -> None:
    install_warning_filter()
    agent = cfg.agent
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    if not cfg.gaussian:
        use_standard_cameras_instead_of_gaussian(env_cfg)
    removed_camera_obs, removed_sensors = disable_camera_observations(env_cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env)
    step_dt = float(env.unwrapped.step_dt)
    cfg._body_names = extract_body_names(env.unwrapped)
    policy = _make_policy(
        task_id=task_id,
        agent=agent,
        action_shape=env.unwrapped.action_space.shape,
        device=device,
        env=env.unwrapped,
        replay_path=cfg.replay_path,
    )
    reset_policy_after_env_done(env, policy)

    if cfg.overwrite and cfg.zarr_path is None:
        task_out_dir = cfg.out_dir / task_id
        if task_out_dir.exists():
            shutil.rmtree(task_out_dir)
    zarr_path = task_zarr_path(task_id, cfg)
    writer = StreamingZarrWriter(task_id, zarr_path, cfg, step_dt=step_dt)
    goal_scheduler: BalancedGoalScheduler | None = None
    if task_goal_names(task_id) is not None and cfg.balance_goals:
        episodes_per_variant = cfg.episodes_per_goal
        if episodes_per_variant is None:
            episodes_per_variant = default_goal_episodes_per_variant(task_id)
        if episodes_per_variant is None:
            raise ValueError(
                f"No default balanced goal quota is registered for {task_id!r}."
            )
        goal_scheduler = BalancedGoalScheduler(
            task_id,
            episodes_per_variant=episodes_per_variant,
            seed=cfg.goal_balance_seed,
            existing_goals=writer.existing_episode_goals(),
        )
    elif cfg.episodes_per_goal is not None:
        raise ValueError(
            "--episodes-per-goal requires a variable-goal task with goal balancing enabled."
        )
    writer.configure_goal_balance(goal_scheduler)
    install_task_goal_provider(env.unwrapped, goal_scheduler)

    buffers = [EpisodeBuffer() for _ in range(env.num_envs)]
    obs, _ = env.reset()
    reset_fn = getattr(policy, "reset", None)
    if reset_fn is not None:
        reset_fn()

    target_successes = (
        goal_scheduler.target_episodes
        if goal_scheduler is not None
        else int(cfg.num_episodes)
    )
    successes = writer.num_episodes
    if successes > target_successes:
        raise ValueError(
            f"Existing collection has {successes} episodes, above target {target_successes}; "
            "use --overwrite or increase the target."
        )
    attempts = 0
    step_count = 0
    progress = (
        tqdm(
            total=target_successes,
            initial=successes,
            desc=f"{task_id} successes",
            unit="ep",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )

    log = progress.write if progress is not None else print
    log(
        f"Collecting task_id={task_id}, agent={agent}, num_envs={env.num_envs}, "
        f"target_successes={target_successes}, "
        f"state_only=True, output={zarr_path}, "
        f"removed_camera_obs={removed_camera_obs}, removed_sensors={removed_sensors}"
    )
    if goal_scheduler is not None:
        log(
            f"Balanced goals: mode={task_goal_balance_mode(task_id)}, "
            f"variants={len(goal_scheduler.variants)}, "
            f"episodes_per_variant={goal_scheduler.episodes_per_variant}, "
            f"already_saved={successes}"
        )

    try:
        while successes < target_successes:
            action = policy(obs)
            replay_state = extract_replay_state(env.unwrapped, task_id)
            replay_state.update(extract_task_replay_state(env.unwrapped, task_id))
            task_goal = extract_task_goal(env.unwrapped, task_id)
            next_obs, reward, done, extras = env.step(action)

            done_bool = (
                torch.as_tensor(done, device=env.unwrapped.device).bool().reshape(-1)
            )
            success_bool = extract_success(env, extras)
            time_out_bool = extract_time_out(env, extras)

            del reward
            action_np = snapshot_to_numpy(action)

            for env_id in range(env.num_envs):
                lane_state = {
                    key: split_env_value(value, env_id, env.num_envs)
                    for key, value in replay_state.items()
                }
                buffers[env_id].append(
                    {
                        "state": lane_state,
                        "action": split_env_value(action_np, env_id, env.num_envs),
                        "done": bool(done_bool[env_id].item()),
                        "success": bool(success_bool[env_id].item()),
                        "time_out": bool(time_out_bool[env_id].item()),
                        "goal": (
                            None
                            if task_goal is None
                            else split_env_value(task_goal, env_id, env.num_envs)
                        ),
                    }
                )

            step_count += 1
            finished = done_bool
            if finished.any():
                for env_id in torch.where(finished)[0].detach().cpu().tolist():
                    attempts += 1
                    is_success = bool(success_bool[env_id].item())
                    num_steps = len(buffers[env_id])
                    quota_accepts = True
                    episode_goal = None
                    if is_success and num_steps > 0 and goal_scheduler is not None:
                        episode_goal = np.asarray(
                            buffers[env_id].steps[0]["goal"]
                        ).reshape(-1)
                        quota_accepts = goal_scheduler.accept_success(episode_goal)
                    if (
                        is_success
                        and num_steps > 0
                        and successes < target_successes
                        and quota_accepts
                    ):
                        saved_steps, trimmed = writer.write_episode(
                            buffers[env_id].steps,
                            env_id=env_id,
                            success=is_success,
                        )
                        log(
                            f"saved episode={writer.num_episodes - 1} to {zarr_path} "
                            f"success={is_success} steps={saved_steps} trimmed={trimmed}"
                        )
                        successes += 1
                        if progress is not None:
                            progress.update(1)
                    else:
                        quota_note = ""
                        if is_success and not quota_accepts:
                            quota_note = (
                                f" goal={episode_goal.tolist()} quota_complete=True"
                            )
                        log(
                            f"dropped task={task_id} env={env_id} "
                            f"success={is_success} time_out={bool(time_out_bool[env_id].item())} "
                            f"steps={num_steps}{quota_note}"
                        )
                    buffers[env_id].clear()

            obs = next_obs

            if step_count % 100 == 0:
                if progress is not None:
                    progress.set_postfix(
                        attempts=attempts,
                        steps=step_count,
                        refresh=True,
                    )
                else:
                    print(
                        f"progress task={task_id}: successes={successes}/"
                        f"{target_successes}, attempts={attempts}, steps={step_count}"
                    )
    finally:
        if progress is not None:
            progress.close()
        env.close()


def parse_config() -> CollectConfig:
    cfg = tyro.cli(CollectConfig, description=__doc__)
    if cfg.zarr_path is not None and len(cfg.task_ids) != 1:
        print("--zarr-path can only be used with exactly one task id.", file=sys.stderr)
        raise SystemExit(2)
    if cfg.num_episodes < 1:
        print("--num-episodes must be positive.", file=sys.stderr)
        raise SystemExit(2)
    if cfg.episodes_per_goal is not None and cfg.episodes_per_goal < 1:
        print("--episodes-per-goal must be positive.", file=sys.stderr)
        raise SystemExit(2)

    registered = set(list_tasks())
    unknown = [task_id for task_id in cfg.task_ids if task_id not in registered]
    if unknown:
        print(f"Unknown task id(s): {unknown}", file=sys.stderr)
        print("Available tasks:", file=sys.stderr)
        for task_id in sorted(registered):
            print(f"  {task_id}", file=sys.stderr)
        raise SystemExit(2)

    return cfg


def main() -> None:
    cfg = parse_config()
    for task_id in cfg.task_ids:
        collect_task(task_id, cfg)


if __name__ == "__main__":
    main()
