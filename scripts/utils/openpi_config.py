"""OpenPI evaluation settings and per-task camera mappings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.task_goals import TASK_PROMPT_TEMPLATES

TASK_SECOND_WRIST_CAMERAS = {
    "Aloha-Screw-Tighten": "gs_aloha_2_wrist_cam_rgb",
    "Shadow-Tube-Rotate": "gs_tube_holder_wrist_cam_rgb",
}

DUAL_ARM_TASK_IDS = frozenset(TASK_SECOND_WRIST_CAMERAS)

TWO_CAMERA_TASK_IDS = frozenset({"XHand-Tube-Insert"})

SUPPORTED_TASK_IDS = frozenset(TASK_PROMPT_TEMPLATES)


@dataclass
class EvalOpenPIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    task_id: str = "XHand-Tube-Insert"
    prompt: str | None = None
    num_envs: int = 100
    num_episodes: int = 100
    seed: int | None = None
    max_steps: int | None = None
    device: str | None = None
    viewer: Literal["none", "viser", "mujoco"] = "none"
    viewer_frame_rate: float = 60.0
    gaussian: bool = True
    camera_width: int = 640
    camera_height: int = 480
    replan_ratio: float = 0.8
    inference_batch_size: int = 16
    progress_interval_steps: int = 100
    request_timeout_s: float = 300.0
    norm_stats_path: Path | None = None
    diagnostics_path: Path | None = None
    replay_output_dir: Path | None = None
    replays_per_outcome: int = 8
    clip_actions_to_training_quantiles: bool = False
    action_clip_margin: float = 0.05
    diagnostic_interval_replans: int = 100
    front_camera: str = "gs_front_cam_rgb"
    left_camera: str = "gs_left_cam_rgb"
    wrist1_camera: str = "gs_wrist_cam_rgb"
    wrist2_camera: str | None = None


def policy_camera_sources(cfg: EvalOpenPIConfig) -> dict[str, str]:
    if cfg.task_id in DUAL_ARM_TASK_IDS:
        return {
            "front": cfg.front_camera,
            "wrist1": cfg.wrist1_camera,
            "wrist2": cfg.wrist2_camera or TASK_SECOND_WRIST_CAMERAS[cfg.task_id],
        }
    if cfg.task_id in TWO_CAMERA_TASK_IDS:
        return {
            "front": cfg.front_camera,
            "wrist1": cfg.wrist1_camera,
        }
    return {
        "front": cfg.front_camera,
        "left": cfg.left_camera,
        "wrist1": cfg.wrist1_camera,
    }
