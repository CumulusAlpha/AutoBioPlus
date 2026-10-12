"""Command-line configuration for demonstration collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import tyro


@dataclass
class CollectConfig:
    task_ids: Annotated[list[str], tyro.conf.Positional]
    agent: Literal["auto", "random", "slider"] = "auto"
    num_envs: int = 1
    num_episodes: int = 100
    balance_goals: bool = True
    episodes_per_goal: int | None = None
    goal_balance_seed: int = 0
    out_dir: Path = Path("logs/collect")
    zarr_path: Path | None = None
    device: str | None = None
    replay_path: str | None = None
    gaussian: bool = False
    video_fps: int = 30
    state_key: str | None = None
    skip_static_frames: bool = False
    float_dtype: Literal["float32", "float64", "preserve"] = "preserve"
    chunk_length: int = 16
    overwrite: bool = False
