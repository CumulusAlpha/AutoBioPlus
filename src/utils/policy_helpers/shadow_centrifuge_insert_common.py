"""Shared constants, states and functions for shadow_centrifuge_insert."""

from __future__ import annotations

from enum import Enum


class ShadowCentrifugeInsertStage(Enum):
    PREGRASP_ALIGN = 0
    REPLAY = 1
    LIFT = 2
    MOVE_ABOVE = 3
    PRE_INSERT = 4
    INSERT = 5
    RELEASE = 6
    BACK = 7
    DONE = 8
    HOLD = 9
