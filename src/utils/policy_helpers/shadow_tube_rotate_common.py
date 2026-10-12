"""Shared constants, states and functions for shadow_tube_rotate."""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import torch

DEFAULT_GRASP_PAYLOAD: dict[str, np.ndarray] = {
    "pregrasp_act_target": np.array(
        [
            [
                0.10619645,
                0.0343252,
                -0.10724832,
                0.188483,
                -0.30322477,
                -0.5645081,
                0.7442174,
                -0.44156453,
                1.1217537,
                0.0984474,
                -0.04304415,
                0.28962475,
                -0.24759068,
                0.6678108,
                -0.06895962,
                0.7427793,
                -0.24911867,
                0.9162631,
                0.09979869,
                -0.20513126,
                1.369119,
                0.86016536,
                0.8368365,
                0.8253145,
                2.5770733,
            ],
            [
                0.10438892,
                0.03729206,
                -0.10485923,
                0.18893133,
                -0.30352107,
                -0.5653323,
                0.74335676,
                -0.38703445,
                1.122232,
                0.09765308,
                -0.00935754,
                0.30164772,
                -0.24810493,
                0.70645165,
                -0.07264046,
                0.77293295,
                -0.24905744,
                0.9368714,
                0.09962408,
                -0.21733505,
                1.3544897,
                0.8776207,
                0.8499685,
                0.8343087,
                2.5732112,
            ],
            [
                0.10073067,
                0.04243504,
                -0.10111436,
                0.18825552,
                -0.30772457,
                -0.5622818,
                0.7441133,
                -0.26645878,
                1.1219293,
                0.09753326,
                0.06581495,
                0.32909766,
                -0.24913391,
                0.7869851,
                -0.07697761,
                0.82445526,
                -0.2492082,
                0.96454865,
                0.09949797,
                -0.24165022,
                1.3296275,
                0.9142537,
                0.8724513,
                0.84594023,
                2.5660057,
            ],
            [
                0.1014052,
                0.04442824,
                -0.10157144,
                0.19121666,
                -0.30761614,
                -0.5580102,
                0.7466144,
                -0.16481026,
                1.1216407,
                0.09679624,
                0.12960458,
                0.3529233,
                -0.24915911,
                0.8483233,
                -0.07955288,
                0.8753556,
                -0.249241,
                1.0291202,
                0.11051693,
                -0.24901207,
                1.3211972,
                0.94066,
                0.8936669,
                0.87365395,
                2.5604625,
            ],
            [
                0.102819,
                0.04556898,
                -0.1036949,
                0.19526587,
                -0.30735412,
                -0.551487,
                0.750511,
                -0.06918583,
                1.1218755,
                0.09553288,
                0.19229111,
                0.37938356,
                -0.24921389,
                0.9029619,
                -0.08178588,
                0.9228026,
                -0.2492261,
                1.1078598,
                0.10648694,
                -0.2490458,
                1.3191749,
                0.9632734,
                0.9123223,
                0.90748566,
                2.5604358,
            ],
            [
                0.10325174,
                0.04649743,
                -0.10607679,
                0.19548255,
                -0.31458148,
                -0.5405244,
                0.7554194,
                0.035731,
                1.1221157,
                0.09452802,
                0.2619175,
                0.4108965,
                -0.24925111,
                0.98291916,
                -0.08254591,
                0.9694714,
                -0.24901614,
                1.1750695,
                0.10194285,
                -0.24908672,
                1.3169787,
                0.9969462,
                0.9301203,
                0.9346485,
                2.5606024,
            ],
        ],
        dtype=np.float32,
    ),
    "grasp_act_target": np.array(
        [
            [
                0.10344163,
                0.04674608,
                -0.10703193,
                0.19524729,
                -0.3169889,
                -0.53607064,
                0.75764424,
                0.07370746,
                1.1220727,
                0.09524512,
                0.28626916,
                0.42139763,
                -0.24910554,
                1.0178083,
                -0.08197559,
                0.98643154,
                -0.24914911,
                1.2001957,
                0.09990521,
                -0.24914062,
                1.3164568,
                1.0118339,
                0.9366094,
                0.9445283,
                2.5605845,
            ]
        ],
        dtype=np.float32,
    ),
    "squeeze_act_target": np.array(
        [
            [
                0.10344163,
                0.04674608,
                -0.10703193,
                0.19524729,
                -0.3169889,
                -0.53607064,
                0.75764424,
                0.27567342,
                1.0990596,
                0.08141719,
                0.40822023,
                0.47013876,
                -0.23993288,
                1.0899029,
                -0.09124269,
                1.0917926,
                -0.24878675,
                1.2492578,
                0.09990521,
                -0.24914062,
                1.3164568,
                1.0429345,
                0.9763986,
                0.96617883,
                2.5605845,
            ]
        ],
        dtype=np.float32,
    ),
}


TUBE_ROTATION_AXIS_LOCAL = (0.0, 0.0, 1.0)


LEFT_TUBE_TARGET_X_OFFSET = 0.1


CAP_LIFT_ANGLE_DEG = -185.0


CAP_LIFT_ANGLE_RAD = math.radians(CAP_LIFT_ANGLE_DEG)


CAP_LIFT_TURNS = CAP_LIFT_ANGLE_RAD / math.tau


CAP_FINAL_ROTATE_TRIGGER_ANGLE_RAD = math.radians(-175.0)


CAP_FINAL_ROTATE_TARGET_ANGLE_RAD = math.radians(-195.0)


def random_sphere_torch(center, diameter, B):
    if not torch.is_tensor(center):
        center = torch.tensor(center, dtype=torch.float32)
    if center.ndim == 1:
        center = center.reshape(1, 3).expand(B, -1)
    elif center.ndim != 2 or center.shape != (B, 3):
        raise ValueError(
            f"center must have shape (3,) or ({B}, 3), got {tuple(center.shape)}"
        )

    device = center.device
    dtype = center.dtype
    if not torch.is_tensor(diameter):
        diameter = torch.full((B,), float(diameter), dtype=dtype, device=device)
    else:
        diameter = diameter.to(device=device, dtype=dtype)
        if diameter.ndim == 0:
            diameter = diameter.expand(B)

    radius = diameter.reshape(B, 1) / 2.0
    direction = torch.randn(B, 3, device=device, dtype=dtype)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-9)
    u = torch.rand(B, 1, device=device, dtype=dtype)
    return center + direction * radius * (u ** (1.0 / 3.0))


class ShadowTubeRotateState(Enum):
    LEFT_PLACE = 0
    PREGRASP = 1
    REPLAY = 2
    ROTATE = 3
    LIFT = 4
    CAP_LIFT = 5
    HOLD = 6
