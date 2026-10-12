#!/usr/bin/env python3
"""Evaluate DexBioLab environments with an OpenPI policy server."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from play import (
    DexBioLabViserPlayViewer,
    PlayTaskGoalDisplay,
    count_gaussian_sensors,
    patch_viser_sdf_geom_rendering,
    reset_policy_after_env_done,
    use_standard_cameras_instead_of_gaussian,
)

import src.tasks  # noqa: F401 - registers DexBioLab tasks
from scripts.utils.diagnostics import (
    JsonlDiagnostics as JsonlDiagnostics,
    _termination_vectors as _termination_vectors,
    done_vector as done_vector,
    nonzero_dimension_counts as nonzero_dimension_counts,
)
from scripts.utils.openpi_config import (
    DUAL_ARM_TASK_IDS as DUAL_ARM_TASK_IDS,
    SUPPORTED_TASK_IDS as SUPPORTED_TASK_IDS,
    TASK_SECOND_WRIST_CAMERAS as TASK_SECOND_WRIST_CAMERAS,
    TWO_CAMERA_TASK_IDS as TWO_CAMERA_TASK_IDS,
    EvalOpenPIConfig as EvalOpenPIConfig,
    policy_camera_sources as policy_camera_sources,
)
from scripts.utils.openpi_policy import (
    OpenPIChunkPolicy as OpenPIChunkPolicy,
    _InferenceRequest as _InferenceRequest,
    _InferenceResult as _InferenceResult,
    load_quantile_bounds as load_quantile_bounds,
    normalize_action_quaternions as normalize_action_quaternions,
    resize_uint8_nhwc as resize_uint8_nhwc,
    to_numpy as to_numpy,
)
from src.tasks.success_diagnostics import (
    grouped_success_diagnostic_counts,
    success_diagnostics_for_envs,
    update_success_diagnostic_counts,
)
from src.utils.chunked_evaluation import (
    ChunkedSensingThrottle,
)
from src.utils.environment import extract_success
from src.utils.first_episode_evaluation import FirstEpisodeStateFreezer


def configure_camera_resolutions(env_cfg: Any, *, width: int, height: int) -> int:
    changed = 0
    for sensor_cfg in getattr(env_cfg.scene, "sensors", ()):
        if hasattr(sensor_cfg, "width") and hasattr(sensor_cfg, "height"):
            sensor_cfg.width = width
            sensor_cfg.height = height
            changed += 1
    return changed


def configure_learned_policy_success(env_cfg: Any) -> None:
    for term_cfg in getattr(env_cfg, "terminations", {}).values():
        params = getattr(term_cfg, "params", None)
        if params is None or "use_ring_history" not in params:
            continue
        learned_params = dict(params)
        ring_motion_threshold_rad = float(np.deg2rad(5.0))
        learned_params["min_yellow_angle"] = ring_motion_threshold_rad
        learned_params["max_silver_angle"] = -ring_motion_threshold_rad
        learned_params["stability_steps"] = 1
        learned_params["use_ring_history"] = True
        term_cfg.params = learned_params


def evaluate(cfg: EvalOpenPIConfig) -> None:
    if cfg.task_id not in SUPPORTED_TASK_IDS:
        raise ValueError(f"Unsupported completed DexBioLab task: {cfg.task_id!r}.")
    if cfg.num_envs < 1 or cfg.num_episodes < 1:
        raise ValueError("num_envs and num_episodes must be positive.")
    if cfg.seed is not None and cfg.seed < 0:
        raise ValueError("seed must be non-negative when provided.")
    if cfg.max_steps is not None and cfg.max_steps < 1:
        raise ValueError("max_steps must be positive when provided.")
    if cfg.progress_interval_steps < 1:
        raise ValueError("progress_interval_steps must be positive.")
    if cfg.request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be positive.")
    if not 0.0 < cfg.replan_ratio <= 1.0:
        raise ValueError("replan_ratio must be in (0, 1].")
    if cfg.inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive.")
    if cfg.action_clip_margin < 0:
        raise ValueError("action_clip_margin must be non-negative.")
    if cfg.diagnostic_interval_replans < 1:
        raise ValueError("diagnostic_interval_replans must be positive.")
    if cfg.camera_width < 1 or cfg.camera_height < 1:
        raise ValueError("Camera dimensions must be positive.")
    if cfg.replays_per_outcome < 1:
        raise ValueError("replays_per_outcome must be positive.")
    if cfg.replay_output_dir is not None and cfg.viewer != "none":
        raise ValueError("Replay recording requires viewer='none'.")
    if cfg.viewer == "none" and cfg.num_envs < cfg.num_episodes:
        raise ValueError(
            "One-episode-per-environment evaluation requires num_envs >= num_episodes."
        )
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    effective_num_envs = min(cfg.num_envs, cfg.num_episodes)
    if effective_num_envs != cfg.num_envs:
        print(
            f"Capping evaluation environments from {cfg.num_envs} to "
            f"num_episodes={cfg.num_episodes}."
        )
    if cfg.viewer == "none":
        print(
            "Evaluation protocol: one initial episode per environment; completed "
            "environments are frozen and retired from policy inference."
        )
    camera_sources = policy_camera_sources(cfg)
    camera_summary = ", ".join(
        f"{alias}<-{source}" for alias, source in camera_sources.items()
    )
    print(f"OpenPI camera mapping: {camera_summary}")

    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = effective_num_envs
    configure_learned_policy_success(env_cfg)
    if cfg.task_id == "XHand-Bunsen-Manipulate":
        print(
            "Bunsen success: each ring only needs 5 degrees of motion in the correct direction."
        )
    print(f"Evaluation seed: {cfg.seed}")
    if cfg.gaussian:
        print(
            f"Gaussian rendering enabled; using {count_gaussian_sensors(env_cfg)} Gaussian camera(s)."
        )
    else:
        converted = use_standard_cameras_instead_of_gaussian(env_cfg)
        if converted:
            print(
                f"Gaussian rendering disabled; using standard cameras for {converted} camera(s)."
            )
    camera_count = configure_camera_resolutions(
        env_cfg,
        width=cfg.camera_width,
        height=cfg.camera_height,
    )
    print(
        f"Configured {camera_count} camera sensor(s) to {cfg.camera_width}x{cfg.camera_height}."
    )

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
    policy = OpenPIChunkPolicy(cfg, env)
    diagnostics = JsonlDiagnostics(cfg.diagnostics_path)

    if cfg.viewer != "none":
        sensing_throttle = None
        try:
            env.reset()
            policy.reset()
            if not cfg.gaussian:
                sensing_throttle = ChunkedSensingThrottle.install_if_supported(
                    env,
                    policy.needs_replan,
                )
                if sensing_throttle is not None:
                    print(
                        "Standard-camera sensing runs only when the policy replans or "
                        "an environment resets."
                    )
            if cfg.viewer == "viser":
                import viser

                patch_viser_sdf_geom_rendering()
                server = viser.ViserServer()
                goal_display = PlayTaskGoalDisplay(
                    cfg.task_id,
                    env,
                    server,
                    prompt_override=cfg.prompt,
                )
                reset_policy_after_env_done(
                    env,
                    policy,
                    on_env_reset=goal_display.refresh_envs,
                )
                DexBioLabViserPlayViewer(
                    env,
                    policy,
                    goal_display=goal_display,
                    frame_rate=cfg.viewer_frame_rate,
                    viser_server=server,
                ).run(num_steps=cfg.max_steps)
            elif cfg.viewer == "mujoco":
                from mjlab.viewer import NativeMujocoViewer

                reset_policy_after_env_done(env, policy)
                NativeMujocoViewer(env, policy, frame_rate=cfg.viewer_frame_rate).run(
                    num_steps=cfg.max_steps
                )
            else:
                raise RuntimeError(f"Unsupported viewer type: {cfg.viewer!r}")
        finally:
            if sensing_throttle is not None:
                sensing_throttle.close()
            diagnostics.close()
            policy.close()
            env.close()
        return

    obs, _ = env.reset()
    policy.reset()
    state_freezer = FirstEpisodeStateFreezer(env)
    sensing_throttle = None
    if not cfg.gaussian:
        sensing_throttle = ChunkedSensingThrottle.install_if_supported(
            env,
            policy.needs_replan,
        )
        if sensing_throttle is not None:
            print(
                "Standard-camera sensing runs only when the policy replans or "
                "an environment resets."
            )
    episodes_done = 0
    successes = 0
    steps = 0
    episode_lengths = np.zeros(env.unwrapped.num_envs, dtype=np.int64)
    finished_mask = torch.zeros(
        env.unwrapped.num_envs,
        device=env.unwrapped.device,
        dtype=torch.bool,
    )
    termination_counts: Counter[str] = Counter()
    success_diagnostic_counts: Counter[str] = Counter()
    replay_recorder = None
    evaluation_finished = False

    try:
        if cfg.replay_output_dir is not None:
            from utils.replay_state_buffer import EpisodeReplayRecorder

            replay_recorder = EpisodeReplayRecorder(cfg, env)
            replay_recorder.write()
        while episodes_done < cfg.num_episodes:
            if cfg.max_steps is not None and steps >= cfg.max_steps:
                print(f"Reached max_steps={cfg.max_steps}; stopping early.")
                break

            active_ids = torch.where(~finished_mask)[0]
            action = policy(obs)
            obs, reward, done, extras = env.step(action)
            del reward
            if replay_recorder is not None:
                replay_recorder.write(env_ids=to_numpy(active_ids, dtype=np.int64))
            steps += 1
            episode_lengths[to_numpy(active_ids, dtype=np.int64)] += 1
            if steps % cfg.progress_interval_steps == 0:
                print(
                    f"progress: episodes={episodes_done}/{cfg.num_episodes}, "
                    f"successes={successes}, steps={steps}",
                    flush=True,
                )

            done_vec = done_vector(done, device=env.unwrapped.device) & ~finished_mask
            if done_vec.any():
                success_vec = extract_success(env, extras)
                termination_vectors = _termination_vectors(env)
                finished_ids = torch.where(done_vec)[0]
                terminal_success_diagnostics = success_diagnostics_for_envs(
                    env.unwrapped,
                    finished_ids,
                )
                finished = int(finished_ids.numel())
                succeeded = int(success_vec[finished_ids].sum().item())
                episode_policy_stats = policy.pop_episode_diagnostics(finished_ids)
                for episode_offset, env_id_tensor in enumerate(finished_ids):
                    env_id = int(env_id_tensor.item())
                    reasons = sorted(
                        name
                        for name, values in termination_vectors.items()
                        if bool(values[env_id].item())
                    )
                    termination_counts.update(reasons or ("unknown",))
                    success_diagnostics = terminal_success_diagnostics.get(env_id, {})
                    update_success_diagnostic_counts(
                        success_diagnostic_counts,
                        success_diagnostics,
                    )
                    diagnostics.write(
                        {
                            "episode": episodes_done + episode_offset,
                            "task_id": cfg.task_id,
                            "seed": cfg.seed,
                            "env_id": env_id,
                            "success": bool(success_vec[env_id].item()),
                            "length": int(episode_lengths[env_id]),
                            "termination_reasons": reasons,
                            "prompt": policy.prompts[env_id],
                            "success_diagnostics": success_diagnostics,
                            **episode_policy_stats[env_id],
                        }
                    )
                    if replay_recorder is not None:
                        replay_recorder.finish(
                            env_id,
                            success=bool(success_vec[env_id].item()),
                            episode=episodes_done + episode_offset,
                        )
                episodes_done += finished
                successes += succeeded
                finished_mask[finished_ids] = True
                state_freezer.freeze(finished_ids)
                policy.retire_envs(finished_ids)
                print(
                    f"finished={episodes_done}/{cfg.num_episodes}, "
                    f"successes={successes}, steps={steps}, "
                    f"terminations={dict(termination_counts)}"
                )
            state_freezer.restore()
        evaluation_finished = True
    finally:
        state_freezer.close()
        if sensing_throttle is not None:
            sensing_throttle.close()
        if replay_recorder is not None:
            replay_recorder.close(select=evaluation_finished)
        diagnostics.close()
        policy_summary = policy.summary()
        policy.close()
        env.close()

    success_rate = successes / episodes_done if episodes_done else 0.0
    print(
        "Evaluation diagnostics: "
        f"terminations={dict(termination_counts)}, "
        f"policy_requests={policy_summary['policy_requests']}, "
        f"inference_batches={policy_summary['inference_batches']}, "
        f"mean_model_batch_size={policy_summary['mean_model_batch_size']:.1f}, "
        f"padded_inference_items={policy_summary['padded_inference_items']}, "
        f"mean_infer_ms={policy_summary['mean_infer_ms']:.1f}, "
        f"state_ood={policy_summary['state_ood_values']}/"
        f"{policy_summary['state_values']} "
        f"({policy_summary['state_ood_fraction']:.2%}), "
        f"action_outside_training_range="
        f"{policy_summary['action_outside_training_range_values']}/"
        f"{policy_summary['action_values']} "
        f"({policy_summary['action_outside_training_range_fraction']:.2%}), "
        f"action_buffer_hold_steps={policy_summary['action_buffer_hold_steps']}."
    )
    print(
        "Diagnostic dimensions: "
        f"state_ood={nonzero_dimension_counts(policy_summary['state_ood_by_dim'])}, "
        f"action_outside_training_range="
        f"{nonzero_dimension_counts(policy_summary['action_outside_training_range_by_dim'])}."
    )
    diagnosed_episodes = success_diagnostic_counts["episodes_with_diagnostics"]
    object_complete_episodes = success_diagnostic_counts["object_complete_episodes"]
    final_motion_gate_episodes = success_diagnostic_counts[
        "episodes_with_final_motion_gate"
    ]
    final_motion_complete_episodes = success_diagnostic_counts[
        "final_motion_complete_episodes"
    ]
    object_completion_rate = (
        object_complete_episodes / diagnosed_episodes if diagnosed_episodes else 0.0
    )
    print(
        "Success-condition diagnostics: "
        f"object_complete={object_complete_episodes}/{diagnosed_episodes} "
        f"({object_completion_rate:.2%}), "
        f"final_motion_complete={final_motion_complete_episodes}/"
        f"{final_motion_gate_episodes}, "
        f"object_complete_without_full_success="
        f"{success_diagnostic_counts['object_complete_without_full_success']}, "
        f"object_complete_without_final_motion="
        f"{success_diagnostic_counts['object_complete_without_final_motion']}, "
        f"object_and_final_motion_never_coincided="
        f"{success_diagnostic_counts['object_and_final_motion_never_coincided']}."
    )
    diagnostic_breakdown = grouped_success_diagnostic_counts(success_diagnostic_counts)
    print(f"Success outcome counts: {diagnostic_breakdown['outcomes']}.")
    print(
        "Success component counts: "
        f"ever={diagnostic_breakdown['components_ever']}, "
        f"terminal={diagnostic_breakdown['components_terminal']}, "
        f"regressed={diagnostic_breakdown['components_regressed']}."
    )
    print(
        f"Evaluation complete: episodes={episodes_done}, successes={successes}, "
        f"success_rate={success_rate:.1%}, steps={steps}."
    )


def main() -> None:
    evaluate(tyro.cli(EvalOpenPIConfig))


if __name__ == "__main__":
    main()
