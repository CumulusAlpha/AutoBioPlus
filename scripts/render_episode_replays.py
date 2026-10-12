#!/usr/bin/env python3
"""Render balanced episode-state replay archives into multi-camera MP4 videos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO, cast

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.tasks.registry import load_env_cfg
from play import use_standard_cameras_instead_of_gaussian
from utils.render_utils import (
    _freejoint_qpos_addresses,
    apply_bunsen_fire_state_for_render,
    apply_heating_effect_state_for_render,
    apply_mixer_ui_values_for_render,
    camera_rgb_from_sensor_context,
    configure_camera_resolutions,
    configured_camera_sensor_names,
    mixer_display_asset_cfg,
    refresh_camera_state,
)

import src.tasks  # noqa: F401 - registers task configurations.


def _metadata(archive: np.lib.npyio.NpzFile, path: Path) -> dict[str, object]:
    metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("format") != "dexbiolab_episode_replay_v1":
        raise ValueError(
            f"Unsupported replay format in {path}: {metadata.get('format')!r}"
        )
    return metadata


def ffmpeg_process(
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    ffmpeg: str,
    crf: int,
) -> subprocess.Popen[bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-threads",
            "1",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def normalize_positions(
    values: np.ndarray,
    origins: np.ndarray,
    addresses: list[int],
) -> np.ndarray:
    normalized = np.array(values, dtype=np.float32, copy=True)
    for address in addresses:
        normalized[:, address : address + 3] -= origins
    return normalized


def render_directory(
    env: ManagerBasedRlEnv,
    replay_dir: Path,
    output_dir: Path,
    *,
    camera_names: list[str],
    freejoint_addresses: list[int],
    ffmpeg: str,
    crf: int,
    overwrite: bool,
) -> None:
    paths = sorted(replay_dir.glob("*.npz"))
    if not paths:
        return

    loaded: list[tuple[Path, np.lib.npyio.NpzFile, dict[str, object]]] = []
    try:
        for path in paths:
            archive = np.load(path, allow_pickle=False)
            loaded.append((path, archive, _metadata(archive, path)))
        loaded.sort(key=lambda item: int(item[1]["qpos"].shape[0]), reverse=True)

        fps_values = {
            round(1.0 / float(metadata["step_dt"])) for _, _, metadata in loaded
        }
        if len(fps_values) != 1:
            raise ValueError(f"Mixed replay frame rates in {replay_dir}: {fps_values}")
        fps = fps_values.pop()

        outputs = [output_dir / path.with_suffix(".mp4").name for path, _, _ in loaded]
        output_manifest = output_dir / "manifest.tsv"
        if (
            not overwrite
            and output_manifest.is_file()
            and all(path.is_file() and path.stat().st_size > 0 for path in outputs)
        ):
            print(f"Skipping complete directory: {output_dir}", flush=True)
            return

        frame_counts = np.asarray(
            [archive["qpos"].shape[0] for _, archive, _ in loaded], dtype=np.int64
        )
        max_frames = int(frame_counts.max())
        if len(loaded) > env.num_envs:
            raise ValueError(
                f"{replay_dir} contains {len(loaded)} replays but the environment has "
                f"only {env.num_envs} slots."
            )

        mixer_asset_cfg = (
            mixer_display_asset_cfg(env) if "mixer_ui_state" in loaded[0][1] else None
        )
        heating_render_callback = None
        if "heating_effect_state" in loaded[0][1]:
            from src.tasks.config.wuji_heating_plate_rotate.env_cfgs import (
                make_heating_effects_renderer,
            )

            heating_renderer = make_heating_effects_renderer()
            beaker_cfg = SceneEntityCfg("beaker", site_names=["beaker_center"])
            beaker_cfg.resolve(env.scene)
            heating_render_callback = lambda: heating_renderer.launch(env, beaker_cfg)

        refresh_camera_state(env)
        first = camera_rgb_from_sensor_context(env, camera_names, active=1)
        _, camera_count, height, width, channels = first.shape
        if channels != 3:
            raise ValueError(f"Expected RGB camera output, got {channels} channels.")

        processes: list[subprocess.Popen[bytes]] = []
        try:
            for output in outputs:
                if output.exists() and overwrite:
                    output.unlink()
                processes.append(
                    ffmpeg_process(
                        output,
                        width=width * camera_count,
                        height=height,
                        fps=fps,
                        ffmpeg=ffmpeg,
                        crf=crf,
                    )
                )

            sim = env.sim
            device = env.device
            for frame in range(max_frames):
                active = int(np.count_nonzero(frame_counts > frame))
                active_archives = [archive for _, archive, _ in loaded[:active]]
                origins = np.stack(
                    [
                        np.asarray(archive["env_origin"][frame], dtype=np.float32)
                        for archive in active_archives
                    ]
                )
                qpos = np.stack(
                    [
                        np.asarray(archive["qpos"][frame], dtype=np.float32)
                        for archive in active_archives
                    ]
                )
                qpos = normalize_positions(qpos, origins, freejoint_addresses)
                sim.data.qpos[:active] = torch.as_tensor(
                    qpos, dtype=sim.data.qpos.dtype, device=device
                )

                if "mocap_pos" in active_archives[0]:
                    mocap_pos = np.stack(
                        [
                            np.asarray(archive["mocap_pos"][frame], dtype=np.float32)
                            for archive in active_archives
                        ]
                    )
                    mocap_pos -= origins[:, None, :]
                    mocap_quat = np.stack(
                        [
                            np.asarray(archive["mocap_quat"][frame], dtype=np.float32)
                            for archive in active_archives
                        ]
                    )
                    sim.data.mocap_pos[:active] = torch.as_tensor(
                        mocap_pos, dtype=sim.data.mocap_pos.dtype, device=device
                    )
                    sim.data.mocap_quat[:active] = torch.as_tensor(
                        mocap_quat, dtype=sim.data.mocap_quat.dtype, device=device
                    )
                if "time" in active_archives[0]:
                    times = np.asarray(
                        [float(archive["time"][frame]) for archive in active_archives],
                        dtype=np.float32,
                    )
                    sim.data.time[:active] = torch.as_tensor(
                        times, dtype=sim.data.time.dtype, device=device
                    )

                if mixer_asset_cfg is not None:
                    apply_mixer_ui_values_for_render(
                        env,
                        mixer_asset_cfg,
                        np.stack(
                            [
                                archive["mixer_ui_state"][frame]
                                for archive in active_archives
                            ]
                        ),
                    )
                if "bunsen_fire_state" in active_archives[0]:
                    apply_bunsen_fire_state_for_render(
                        env,
                        np.stack(
                            [
                                archive["bunsen_fire_state"][frame]
                                for archive in active_archives
                            ]
                        ),
                    )
                if "heating_effect_state" in active_archives[0]:
                    apply_heating_effect_state_for_render(
                        env,
                        np.stack(
                            [
                                archive["heating_effect_state"][frame]
                                for archive in active_archives
                            ]
                        ),
                    )
                refresh_camera_state(env, post_kinematics=heating_render_callback)
                rendered = camera_rgb_from_sensor_context(
                    env,
                    camera_names,
                    active=active,
                    frame_shape=(height, width, channels),
                )
                combined = np.ascontiguousarray(
                    np.concatenate(
                        [rendered[:, index] for index in range(camera_count)], axis=2
                    )
                )
                for index, process in enumerate(processes[:active]):
                    cast(BinaryIO, process.stdin).write(combined[index].tobytes())

                if frame % 100 == 0 or frame + 1 == max_frames:
                    print(
                        f"{replay_dir.name}: rendered {frame + 1}/{max_frames} frames",
                        flush=True,
                    )

            for process in processes:
                cast(BinaryIO, process.stdin).close()
            for process in processes:
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, process.args)
        except BaseException:
            for process in processes:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.terminate()
                process.wait()
            raise

        manifest = replay_dir / "manifest.tsv"
        if manifest.is_file():
            shutil.copy2(manifest, output_dir / manifest.name)
        print(f"Rendered videos: {output_dir}", flush=True)
    finally:
        for _, archive, _ in loaded:
            archive.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    replay_dirs = sorted({path.parent for path in input_root.rglob("*.npz")})
    if not replay_dirs:
        parser.error(f"No replay archives found below {input_root}.")
    if shutil.which(args.ffmpeg) is None:
        parser.error(f"ffmpeg executable not found: {args.ffmpeg}")

    by_task: dict[str, list[Path]] = {}
    for replay_dir in replay_dirs:
        first_path = next(replay_dir.glob("*.npz"))
        with np.load(first_path, allow_pickle=False) as archive:
            task_id = str(_metadata(archive, first_path)["task_id"])
        by_task.setdefault(task_id, []).append(replay_dir)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    for task_id, task_replay_dirs in by_task.items():
        max_replays = max(len(list(path.glob("*.npz"))) for path in task_replay_dirs)
        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = max_replays
        use_standard_cameras_instead_of_gaussian(env_cfg)
        configure_camera_resolutions(env_cfg, width=args.width, height=args.height)
        CameraSensor.requires_sensor_context = True
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        try:
            camera_names = configured_camera_sensor_names(env)
            if not camera_names or env.sim._sensor_context is None:
                raise RuntimeError(
                    f"Task {task_id} has no active camera sensor context."
                )
            freejoint_addresses = _freejoint_qpos_addresses(env.sim.mj_model)
            for replay_dir in task_replay_dirs:
                relative = replay_dir.relative_to(input_root)
                render_directory(
                    env,
                    replay_dir,
                    output_root / relative,
                    camera_names=camera_names,
                    freejoint_addresses=freejoint_addresses,
                    ffmpeg=args.ffmpeg,
                    crf=args.crf,
                    overwrite=args.overwrite,
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
