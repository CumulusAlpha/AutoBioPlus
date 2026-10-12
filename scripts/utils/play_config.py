"""Settings for standard and Gaussian play viewers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["auto", "random", "slider", "eef"] = "auto"
    viewer: Literal["viser", "mujoco"] = "viser"
    num_envs: int | None = None
    device: str | None = None
    replay_path: str | None = None
    gaussian: bool = False


@dataclass(frozen=True)
class GaussianPlayConfig:
    agent: Literal["auto", "random", "slider", "eef"] = "auto"
    viewer: Literal["viser", "mujoco"] = "viser"
    num_envs: int | None = None
    device: str | None = None
    replay_path: str | None = None
    max_points: int = 1_000_000
    splat_scale: float = 1.0
    camera_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    screenshot_width: int = 3840
    screenshot_height: int = 2160
    start_paused: bool = True
