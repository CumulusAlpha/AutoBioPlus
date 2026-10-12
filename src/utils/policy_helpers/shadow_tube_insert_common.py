"""Shared constants, states and functions for shadow_tube_insert."""

from __future__ import annotations

from enum import Enum


class ShadowTubeInsertStage(Enum):
    PREGRASP_ALIGN = 0
    REPLAY = 1
    LIFT = 2
    MOVE_ABOVE = 3
    PRE_INSERT = 4
    INSERT = 5
    RELEASE_LIFT = 6
    DONE = 7
    HOLD = 8
