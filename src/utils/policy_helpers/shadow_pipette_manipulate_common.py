"""Shared constants, states and functions for shadow_pipette_manipulate."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F

from src.utils.quat_util import (
    quat_to_mat,
)


class ShadowPipetteManipulateState(Enum):
    MOVE_TO_PREGRASP = 0
    PREGRASP = 1
    GRASP = 2
    GRASP_HOLD = 3
    LIFT = 4
    MOVE_TO_CELL_DISH = 5
    ADJUST_PIPETTE_POSE = 6
    LOWER_TO_CELL_DISH = 7
    MANIPULATE_BUTTON = 8
    POST_BUTTON_LIFT = 9


PREGRASP_QUAT_WXYZ = (-0.499999, -0.500001, 0.499999, -0.500001)


START_REL_POS = (-0.049684, 0.146915, 0.095661)


START_REL_QUAT = (-0.499999, -0.500001, 0.499999, -0.500001)


SHADOW_INIT_REL_POS = (-0.022797, 0.123873, 0.121709)


SHADOW_INIT_REL_QUAT = (-0.50, -0.50, 0.50, -0.50)


MOCAP_TO_PALM_POS = (0.0, 0.0, 0.034)


MOCAP_TO_PALM_QUAT = (1.0, 0.0, 0.0, 0.0)


GRASP_QUAT_WXYZ = (0.499999, 0.500001, -0.499999, 0.500001)


GRASP_TENDON = {
    "hand:rh_FFT1": 2.686000,
    "hand:rh_MFT1": 2.658000,
    "hand:rh_RFT1": 2.714000,
    "hand:rh_LFT1": 2.909000,
}


GRASP_QPOS = {
    "hand:rh_FFJ4": -0.006000,
    "hand:rh_FFJ3": 1.035000,
    "hand:rh_FFJ2": 1.343000,
    "hand:rh_FFJ1": 1.343000,
    "hand:rh_MFJ4": 0.005000,
    "hand:rh_MFJ3": 1.168000,
    "hand:rh_MFJ2": 1.329000,
    "hand:rh_MFJ1": 1.329000,
    "hand:rh_RFJ4": -0.026000,
    "hand:rh_RFJ3": 1.052000,
    "hand:rh_RFJ2": 1.357000,
    "hand:rh_RFJ1": 1.357000,
    "hand:rh_LFJ5": 0.100000,
    "hand:rh_LFJ4": -0.148000,
    "hand:rh_LFJ3": 0.852000,
    "hand:rh_LFJ2": 1.454500,
    "hand:rh_LFJ1": 1.454500,
    "hand:rh_THJ5": -0.118000,
    "hand:rh_THJ4": 0.183000,
    "hand:rh_THJ3": -0.082000,
    "hand:rh_THJ2": -0.015000,
    "hand:rh_THJ1": 0.373000,
}


BUTTON_PRESSED_QPOS = {
    **GRASP_QPOS,
    "hand:rh_THJ4": 0.370000,
    "hand:rh_THJ2": 0.420000,
    "hand:rh_THJ1": 0.567000,
}


def pipette_vertical_quat_wxyz(pipette_quat_wxyz: torch.Tensor) -> torch.Tensor:
    current_mat = quat_to_mat(F.normalize(pipette_quat_wxyz, dim=-1))
    x_axis_xy = current_mat[:, :2, 0]
    x_axis_xy_norm = torch.linalg.norm(x_axis_xy, dim=-1)

    y_axis_xy = current_mat[:, :2, 1]
    yaw_from_x = torch.atan2(x_axis_xy[:, 1], x_axis_xy[:, 0])
    yaw_from_y = torch.atan2(-y_axis_xy[:, 0], y_axis_xy[:, 1])
    yaw = torch.where(x_axis_xy_norm > 1e-6, yaw_from_x, yaw_from_y)

    half_yaw = 0.5 * yaw
    return F.normalize(
        torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ),
            dim=-1,
        ),
        dim=-1,
    )
