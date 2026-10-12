#!/usr/bin/env python3
"""Replay a collection and render directly into an OpenPI LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import zarr
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.tasks.registry import load_env_cfg
from play import count_gaussian_sensors, use_standard_cameras_instead_of_gaussian
from utils.lerobot_stream import finish_rgb_stream, write_rgb_batch
from utils.render_utils import (
    _freejoint_qpos_addresses,
    apply_bunsen_fire_state_for_render,
    apply_episode_camera_quaternions,
    apply_episode_origins_to_mocap_pos,
    apply_episode_origins_to_qpos,
    apply_heating_effect_state_for_render,
    apply_mixer_ui_values_for_render,
    bunsen_fire_replay_source,
    camera_rgb_from_sensor_context,
    configure_camera_resolutions,
    configured_camera_sensor_names,
    episode_origin_arrays,
    heating_effect_replay_source,
    mixer_display_asset_cfg,
    mixer_ui_replay_source,
    refresh_camera_state,
    sample_episode_camera_quaternions,
    time_values_for_render,
)

import src.tasks  # noqa: F401 - registers task configurations.
from src.tasks.policy_action import validate_action_layout_attrs

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
WRITER_CONFIG = {
    "openpi": (
        Path(sys.executable),
        REPO_ROOT
        / "thirdparty"
        / "openpi"
        / "scripts"
        / "write_dexbiolab_lerobot.py",
        REPO_ROOT / "thirdparty" / "openpi" / "datasets" / "dexbiolab_lerobot",
    ),
}


def write_replay_qpos_sidecar(
    output: Path,
    *,
    qpos_source,
    total_frames: int,
    episode_ends: np.ndarray,
    episode_env_origins: np.ndarray,
    freejoint_qpos_addresses: list[int],
    batch_size: int = 4096,
) -> None:
    """Store origin-normalized simulation qpos outside the training features."""
    replay_dir = output / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    final_path = replay_dir / "qpos.npy"
    temporary_path = replay_dir / ".qpos.npy.tmp"
    if temporary_path.exists():
        temporary_path.unlink()

    qpos_dim = int(qpos_source.shape[1])
    destination = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, qpos_dim),
    )
    try:
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            destination[start:end] = apply_episode_origins_to_qpos(
                np.asarray(qpos_source[start:end], dtype=np.float32),
                frame_start=start,
                episode_ends=episode_ends,
                episode_env_origins=episode_env_origins,
                freejoint_qpos_addresses=freejoint_qpos_addresses,
            )
        destination.flush()
    finally:
        del destination
    temporary_path.replace(final_path)

    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    provenance = info.setdefault("dexbiolab", {})
    provenance["replay_qpos"] = {
        "path": "replay/qpos.npy",
        "dtype": "float32",
        "shape": [total_frames, qpos_dim],
        "env_origin_normalization": "episode_origin_subtracted",
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n")


def write_camera_randomization_metadata(
    output: Path,
    *,
    enabled: bool,
    camera_names: list[str],
    rotation_range_deg: tuple[float, float],
    seed: int | None,
) -> None:
    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    provenance = info.setdefault("dexbiolab", {})
    provenance["camera_domain_randomization"] = {
        "enabled": enabled,
        "sampling": "per_episode" if enabled else None,
        "camera_sources": camera_names,
        "rotation_range_deg": list(rotation_range_deg) if enabled else None,
        "seed": seed,
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n")


def writer_command(
    args: argparse.Namespace,
    *,
    input_zarr: Path,
    output: Path,
    task_id: str,
    camera_names: list[str],
    height: int,
    width: int,
    total_frames: int,
) -> list[str]:
    default_python, writer_script, _ = WRITER_CONFIG[args.format]
    writer_python = (args.writer_python or default_python).expanduser()
    if not writer_python.is_absolute():
        writer_python = Path.cwd() / writer_python
    writer_python = writer_python.absolute()
    if not writer_python.is_file():
        raise FileNotFoundError(
            f"{args.format} writer Python not found: {writer_python}. Check --writer-python."
        )
    command = [
        str(writer_python),
        str(writer_script),
        str(input_zarr),
        "--output",
        str(output),
        "--task-id",
        task_id,
        "--camera-names-json",
        json.dumps(camera_names),
        "--height",
        str(height),
        "--width",
        str(width),
        "--max-frames",
        str(total_frames),
    ]
    if args.fps is not None:
        command.extend(("--fps", str(args.fps)))
    if args.overwrite:
        command.append("--overwrite")
    if args.format == "openpi":
        command.extend(("--repo-id", args.repo_id))
    else:
        command.extend(("--ffmpeg", args.ffmpeg, "--crf", str(args.crf)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=tuple(WRITER_CONFIG), default="openpi")
    parser.add_argument("--writer-python", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--repo-id", default="local_repo")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--camera-domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize every rendered camera rotation once per episode (default: enabled).",
    )
    parser.add_argument("--camera-rotation-min-deg", type=float, default=1.0)
    parser.add_argument("--camera-rotation-max-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--gaussian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use configured Gaussian/hybrid cameras (default: enabled).",
    )
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be positive.")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive.")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative.")
    if (
        args.camera_rotation_min_deg < 0.0
        or args.camera_rotation_max_deg < args.camera_rotation_min_deg
    ):
        parser.error("Camera rotation degrees must satisfy 0 <= min <= max.")

    input_zarr = args.input_zarr.expanduser().resolve()
    source_root = zarr.open_group(str(input_zarr), mode="r")
    if source_root.attrs.get("created_by") != "scripts/collect.py":
        raise ValueError(
            "Input must be produced by scripts/collect.py; "
            f"created_by={source_root.attrs.get('created_by')!r}."
        )
    task_id = str(source_root.attrs.get("task_id") or input_zarr.parent.name)
    validate_action_layout_attrs(source_root.attrs, task_id)
    if "data" not in source_root or "replay" not in source_root:
        raise KeyError("Collection must contain data and replay groups.")
    if "qpos" not in source_root["replay"]:
        raise KeyError("Collection has no replay/qpos array.")

    _, _, default_output_root = WRITER_CONFIG[args.format]
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (default_output_root / task_id).resolve()
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists. Pass --overwrite to replace it.")

    replay_group = source_root["replay"]
    qpos_source = replay_group["qpos"]
    time_source = replay_group["time"] if "time" in replay_group else None
    mocap_pos_source = (
        replay_group["mocap_pos"] if "mocap_pos" in replay_group else None
    )
    mocap_quat_source = (
        replay_group["mocap_quat"] if "mocap_quat" in replay_group else None
    )
    if (mocap_pos_source is None) != (mocap_quat_source is None):
        raise KeyError(
            "Collection must contain both replay/mocap_pos and replay/mocap_quat."
        )
    total_frames = int(qpos_source.shape[0])
    episode_ends, episode_env_origins = episode_origin_arrays(
        source_root,
        task_id=task_id,
        num_frames=total_frames,
    )
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)
    mixer_ui_source, mixer_ui_name = mixer_ui_replay_source(source_root, task_id)
    bunsen_fire_source, bunsen_fire_name = bunsen_fire_replay_source(
        source_root, task_id
    )
    heating_effect_source, heating_effect_name = heating_effect_replay_source(
        source_root,
        task_id,
    )
    for source, name, width in (
        (mixer_ui_source, mixer_ui_name, 3),
        (bunsen_fire_source, bunsen_fire_name, 2),
        (heating_effect_source, heating_effect_name, 3),
    ):
        if source is not None and (
            source.ndim != 2
            or source.shape[1] < width
            or int(source.shape[0]) < total_frames
        ):
            raise ValueError(
                f"{name} does not cover the {total_frames} selected frames."
            )

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    if args.gaussian:
        print(
            f"Using {count_gaussian_sensors(env_cfg)} configured Gaussian camera(s).",
            flush=True,
        )
    else:
        converted = use_standard_cameras_instead_of_gaussian(env_cfg)
        if converted:
            print(
                f"Using standard replacements for {converted} Gaussian camera(s).",
                flush=True,
            )
    camera_width, camera_height = configure_camera_resolutions(
        env_cfg,
        width=args.width,
        height=args.height,
    )
    CameraSensor.requires_sensor_context = True
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    writer = None

    try:
        sim = env.sim
        if int(qpos_source.shape[1]) != int(sim.mj_model.nq):
            raise ValueError(
                f"replay/qpos width {qpos_source.shape[1]} != model.nq {sim.mj_model.nq}."
            )
        if mocap_pos_source is not None and mocap_quat_source is not None:
            expected_mocap_pos_tail = (int(sim.mj_model.nmocap), 3)
            expected_mocap_quat_tail = (int(sim.mj_model.nmocap), 4)
            if (
                int(mocap_pos_source.shape[0]) < total_frames
                or tuple(mocap_pos_source.shape[1:]) != expected_mocap_pos_tail
            ):
                raise ValueError(
                    f"replay/mocap_pos shape {mocap_pos_source.shape} must contain at least "
                    f"{total_frames} frame(s) with per-frame shape {expected_mocap_pos_tail}."
                )
            if (
                int(mocap_quat_source.shape[0]) < total_frames
                or tuple(mocap_quat_source.shape[1:]) != expected_mocap_quat_tail
            ):
                raise ValueError(
                    f"replay/mocap_quat shape {mocap_quat_source.shape} must contain at least "
                    f"{total_frames} frame(s) with per-frame shape {expected_mocap_quat_tail}."
                )
        freejoint_qpos_addresses = _freejoint_qpos_addresses(sim.mj_model)
        camera_names = configured_camera_sensor_names(env)
        if not camera_names or sim._sensor_context is None:
            raise RuntimeError(
                "Rendering requires an active sensor context with camera sensors."
            )
        camera_rotation_range = (
            args.camera_rotation_min_deg,
            args.camera_rotation_max_deg,
        )
        episode_camera_quats: dict[str, torch.Tensor] = {}
        camera_randomization_seed = None
        if args.camera_domain_randomization:
            selected_episode_count = int(
                np.searchsorted(episode_ends, total_frames - 1, side="right") + 1
            )
            episode_camera_quats, camera_randomization_seed = (
                sample_episode_camera_quaternions(
                    env,
                    camera_names,
                    num_episodes=selected_episode_count,
                    rotation_range_deg=camera_rotation_range,
                    seed=args.seed,
                )
            )
            print(
                "Camera domain randomization: "
                f"per-episode rotation={camera_rotation_range} degrees, "
                f"seed={camera_randomization_seed}.",
                flush=True,
            )
        mixer_asset_cfg = (
            mixer_display_asset_cfg(env) if mixer_ui_source is not None else None
        )
        heating_effects_renderer = None
        heating_beaker_cfg = None
        if heating_effect_source is not None:
            from src.tasks.config.wuji_heating_plate_rotate.env_cfgs import (
                make_heating_effects_renderer,
            )

            heating_effects_renderer = make_heating_effects_renderer()
            heating_beaker_cfg = SceneEntityCfg("beaker", site_names=["beaker_center"])
            heating_beaker_cfg.resolve(env.scene)

        print("Preparing camera sensor context...", flush=True)
        if mixer_ui_source is not None and mixer_asset_cfg is not None:
            apply_mixer_ui_values_for_render(
                env, mixer_asset_cfg, np.asarray(mixer_ui_source[:1], dtype=np.float32)
            )
        if bunsen_fire_source is not None:
            apply_bunsen_fire_state_for_render(
                env, np.asarray(bunsen_fire_source[:1], dtype=np.float32)
            )
        heating_render_callback = None
        if heating_effect_source is not None:
            apply_heating_effect_state_for_render(
                env,
                np.asarray(heating_effect_source[:1], dtype=np.float32),
            )
            assert (
                heating_effects_renderer is not None and heating_beaker_cfg is not None
            )
            heating_render_callback = lambda: heating_effects_renderer.launch(
                env,
                heating_beaker_cfg,
            )
        if episode_camera_quats:
            apply_episode_camera_quaternions(
                env,
                episode_camera_quats,
                frame_start=0,
                active=1,
                episode_ends=episode_ends,
            )
        refresh_camera_state(env, post_kinematics=heating_render_callback)
        first_batch = camera_rgb_from_sensor_context(env, camera_names, active=1)
        num_cameras, height, width, channels = first_batch.shape[1:]
        if channels != 3 or (width, height) != (camera_width, camera_height):
            raise ValueError(
                f"Camera output is {(width, height, channels)}, expected "
                f"{(camera_width, camera_height, 3)}."
            )
        frame_shape = (height, width, channels)

        command = writer_command(
            args,
            input_zarr=input_zarr,
            output=output,
            task_id=task_id,
            camera_names=camera_names,
            height=height,
            width=width,
            total_frames=total_frames,
        )
        print(f"input:  {input_zarr}", flush=True)
        print(f"output: {output}", flush=True)
        print(
            f"format={args.format}, task={task_id}, frames={total_frames}, "
            f"num_envs={env.num_envs}, cameras={camera_names}, image={width}x{height}",
            flush=True,
        )
        writer = subprocess.Popen(
            command, stdin=subprocess.PIPE, start_new_session=True
        )
        if writer.stdin is None:
            raise RuntimeError("LeRobot writer subprocess has no stdin.")

        frame_starts = range(0, total_frames, int(env.num_envs))
        progress = (
            tqdm(
                frame_starts,
                total=(total_frames + int(env.num_envs) - 1) // int(env.num_envs),
                desc=f"Rendering {args.format} LeRobot",
                unit="batch",
            )
            if tqdm is not None
            else frame_starts
        )
        for start in progress:
            end = min(start + int(env.num_envs), total_frames)
            active = end - start
            qpos = apply_episode_origins_to_qpos(
                np.asarray(qpos_source[start:end], dtype=np.float32),
                frame_start=start,
                episode_ends=episode_ends,
                episode_env_origins=episode_env_origins,
                freejoint_qpos_addresses=freejoint_qpos_addresses,
            )
            sim.data.qpos[:active] = torch.as_tensor(
                qpos, dtype=sim.data.qpos.dtype, device=device
            )
            if mocap_pos_source is not None and mocap_quat_source is not None:
                mocap_pos = apply_episode_origins_to_mocap_pos(
                    np.asarray(mocap_pos_source[start:end], dtype=np.float32),
                    frame_start=start,
                    episode_ends=episode_ends,
                    episode_env_origins=episode_env_origins,
                )
                mocap_quat = np.asarray(mocap_quat_source[start:end], dtype=np.float32)
                sim.data.mocap_pos[:active] = torch.as_tensor(
                    mocap_pos,
                    dtype=sim.data.mocap_pos.dtype,
                    device=device,
                )
                sim.data.mocap_quat[:active] = torch.as_tensor(
                    mocap_quat,
                    dtype=sim.data.mocap_quat.dtype,
                    device=device,
                )
            if time_source is not None:
                time_values = time_values_for_render(
                    np.asarray(time_source[start:end]), active
                )
                sim.data.time[:active] = torch.as_tensor(
                    time_values, dtype=sim.data.time.dtype, device=device
                )
            if mixer_ui_source is not None and mixer_asset_cfg is not None:
                apply_mixer_ui_values_for_render(
                    env,
                    mixer_asset_cfg,
                    np.asarray(mixer_ui_source[start:end], dtype=np.float32),
                )
            if bunsen_fire_source is not None:
                apply_bunsen_fire_state_for_render(
                    env, np.asarray(bunsen_fire_source[start:end], dtype=np.float32)
                )
            heating_render_callback = None
            if heating_effect_source is not None:
                apply_heating_effect_state_for_render(
                    env,
                    np.asarray(heating_effect_source[start:end], dtype=np.float32),
                )
                assert (
                    heating_effects_renderer is not None
                    and heating_beaker_cfg is not None
                )
                heating_render_callback = lambda: heating_effects_renderer.launch(
                    env,
                    heating_beaker_cfg,
                )
            if episode_camera_quats:
                apply_episode_camera_quaternions(
                    env,
                    episode_camera_quats,
                    frame_start=start,
                    active=active,
                    episode_ends=episode_ends,
                )
            refresh_camera_state(env, post_kinematics=heating_render_callback)
            rendered = camera_rgb_from_sensor_context(
                env, camera_names, active, frame_shape=frame_shape
            )
            write_rgb_batch(writer.stdin, start, rendered)
            if tqdm is None:
                print(f"rendered {end}/{total_frames}", flush=True)

        finish_rgb_stream(writer.stdin)
        writer.stdin.close()
        return_code = writer.wait()
        if return_code:
            raise RuntimeError(
                f"{args.format} LeRobot writer exited with code {return_code}."
            )
        writer = None
        write_replay_qpos_sidecar(
            output,
            qpos_source=qpos_source,
            total_frames=total_frames,
            episode_ends=episode_ends,
            episode_env_origins=episode_env_origins,
            freejoint_qpos_addresses=freejoint_qpos_addresses,
        )
        write_camera_randomization_metadata(
            output,
            enabled=args.camera_domain_randomization,
            camera_names=camera_names,
            rotation_range_deg=camera_rotation_range,
            seed=camera_randomization_seed,
        )
    except BrokenPipeError as exc:
        return_code = writer.wait() if writer is not None else None
        raise RuntimeError(
            f"LeRobot writer closed its stream early (exit {return_code})."
        ) from exc
    finally:
        if writer is not None:
            writer_pid = writer.pid
            if writer.poll() is None:
                os.killpg(writer_pid, signal.SIGTERM)
                writer.wait()
            staging = output.parent / f".{output.name}.tmp-{writer_pid}"
            if staging.exists():
                shutil.rmtree(staging)
        env.close()

    print(f"Direct LeRobot render complete: {output}", flush=True)


if __name__ == "__main__":
    main()
