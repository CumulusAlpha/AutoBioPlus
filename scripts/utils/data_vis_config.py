"""Command-line configuration for replay visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro


@dataclass(frozen=True)
class DataVisConfig:
    input: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=("-i", "--dataset-path"),
            help="Input collection Zarr or final LeRobot directory.",
        ),
    ] = None
    pkl_path: Path | None = None
    state_key: str = "state"
    replay: bool = False
    action_key: str = "action"
    rgb_key: str = "rgb"
    port: int | None = None
    playback_dt: float = 0.05
