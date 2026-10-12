"""Shared constants, states and functions for wuji_heatingplate_rotate."""

from __future__ import annotations

from enum import Enum

BEAKER_GRASP_REL_POSE = (
    -0.108643,
    -0.030684,
    0.052586,
    0.492446,
    0.507533,
    0.496652,
    0.503234,
)


# Fixed from the previous policy's computed pregrasp; do not derive it from grasp.
BEAKER_PREGRASP_REL_POSE = (
    0.108643,
    0.030684,
    0.132586,
    -0.503234,
    -0.496652,
    0.507533,
    0.492446,
)


BEAKER_GRASP_JOINTS = {
    "right_finger1_joint1": 1.6033,
    "right_finger1_joint2": -0.038,
    "right_finger1_joint3": -0.007,
    "right_finger1_joint4": 1.116,
    "right_finger2_joint1": 0.369,
    "right_finger2_joint2": 0.089,
    "right_finger2_joint3": 0.943,
    "right_finger2_joint4": 0.903,
    "right_finger3_joint1": 0.327,
    "right_finger3_joint2": 0.112,
    "right_finger3_joint3": 0.987,
    "right_finger3_joint4": 0.696,
    "right_finger4_joint1": 0.318,
    "right_finger4_joint2": 0.029,
    "right_finger4_joint3": 0.739,
    "right_finger4_joint4": 0.726,
    "right_finger5_joint1": 0.033,
    "right_finger5_joint2": 0.097,
    "right_finger5_joint3": 0.821,
    "right_finger5_joint4": 0.778,
}


BUTTON_GRASP_REL_POSE = (
    0.112726,
    -0.002236,
    0.094654,
    -0.028017,
    0.742123,
    -0.018137,
    -0.669433,
)


BUTTON_GRASP_JOINTS = {
    "right_finger1_joint1": 1.126,
    "right_finger1_joint2": -0.030,
    "right_finger1_joint3": 0.402,
    "right_finger1_joint4": 0.335,
    "right_finger2_joint1": 0.700,
    "right_finger2_joint2": 0.112,
    "right_finger2_joint3": 0.822,
    "right_finger2_joint4": 0.781,
    "right_finger3_joint1": 0.797,
    "right_finger3_joint2": -0.001,
    "right_finger3_joint3": 0.702,
    "right_finger3_joint4": 0.801,
    "right_finger4_joint1": 0.700,
    "right_finger4_joint2": 0.044,
    "right_finger4_joint3": 0.844,
    "right_finger4_joint4": 0.353,
    "right_finger5_joint1": 0.854,
    "right_finger5_joint2": 0.044,
    "right_finger5_joint3": 0.430,
    "right_finger5_joint4": 0.738,
}


BUTTON_PREGRASP_REL_POSE = (
    0.114819,
    0.030076,
    0.127903,
    0.031223,
    0.75409,
    0.106093,
    -0.647393,
)


BUTTON_PREGRASP_JOINTS = {
    "right_finger1_joint1": 0.634000,
    "right_finger1_joint2": 0.134000,
    "right_finger1_joint3": 0.321,
    "right_finger1_joint4": 0.418,
    "right_finger2_joint1": 0.543000,
    "right_finger2_joint2": 0.022000,
    "right_finger2_joint3": 0.718,
    "right_finger2_joint4": 0.904,
    "right_finger3_joint1": 0.622,
    "right_finger3_joint2": -0.016,
    "right_finger3_joint3": 0.804,
    "right_finger3_joint4": 0.510,
    "right_finger4_joint1": 0.300,
    "right_finger4_joint2": 0.127,
    "right_finger4_joint3": 1.234,
    "right_finger4_joint4": 0.395,
    "right_finger5_joint1": 0.713,
    "right_finger5_joint2": 0.119,
    "right_finger5_joint3": 0.595,
    "right_finger5_joint4": 0.716,
}


# The beaker root is at its bottom.  This target puts it on the center of the
# heating surface, whose top is approximately 0.098 m above the asset root.
BEAKER_TARGET_POSE_HEATING_PLATE = (-0.052, 0.0, 0.099, 1.0, 0.0, 0.0, 0.0)


BEAKER_RELEASE_Y_OFFSET = 0.05


BEAKER_RELEASE_RETREAT_STEPS = 10


HEATING_PLATE_KNOB_AXIS = (0.52573541, -2.29806267e-08, 0.850648152)


BUTTON_HOME_GRASP_AXIS_ROTATION = 0.0


class WujiHeatingPlateRotateState(Enum):
    PREGRASP_BEAKER = 0
    GRASP_BEAKER = 1
    LIFT = 2
    MOVE_TO_HEATING_PLATE = 3
    RELEASE_BEAKER = 4
    PREGRASP_BUTTON = 5
    GRASP_BUTTON = 6
    ROTATE_BUTTON_TO_TARGET = 7
    RELEASE_BUTTON_HOVER = 8
    PREGRASP_BUTTON_HOME = 9
    REGRASP_BUTTON = 10
    ROTATE_BUTTON_HOME = 11
    POST_ROTATE_LIFT = 12
