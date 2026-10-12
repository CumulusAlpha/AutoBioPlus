"""Shared constants, states and functions for aloha_screw_tighten."""

from __future__ import annotations

from enum import IntEnum

import torch

from src.utils.kinematics import batched_interpolate_linear

CAP_COLLISION_GEOM_NAMES = {"cap_cyl", "total_cap_cylinder"}


GRIPPER_COLLISION_GEOM_NAMES = {
    "aloha:leftfinger",
    "aloha:rightfinger",
    "leftfinger",
    "rightfinger",
}


FINGERTIP_SPHERE_GEOM_NAMES = {
    "left_g0",
    "left_g1",
    "left_g2",
    "right_g0",
    "right_g1",
    "right_g2",
}


CAP_ROTATION_AXIS_LOCAL = (0.0, 0.0, 1.0)


def random_sphere_torch(center, diameter, B):
    if not torch.is_tensor(center):
        center = torch.tensor(center, dtype=torch.float32)
    if not torch.is_tensor(diameter):
        diameter = torch.full(
            (B,), float(diameter), dtype=center.dtype, device=center.device
        )
    elif diameter.ndim == 0:
        diameter = diameter.expand(B)

    center = center.reshape(1, 3).expand(B, -1)
    device = center.device

    radius = diameter.view(B, 1) / 2.0

    direction = torch.randn(B, 3, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True)

    u = torch.rand(B, 1, device=device)
    r = radius * (u ** (1.0 / 3.0))

    return center + direction * r


class AlohaScrewTightenStage(IntEnum):
    STAGE_DUAL_REACH = 0
    STAGE_APPROACH_MID = 1
    STAGE_DUAL_GRASP_CLOSE = 2
    STAGE_DUAL_GRASP_SETTLE = 3
    STAGE_DUAL_LIFT_MOVE = 4
    STAGE_ALIGN_INSERT = 5
    STAGE_PRE_INSERT_TILT = 6
    STAGE_INSERT = 7
    STAGE_SEAT_CAP = 8
    STAGE_SPIN_ALIGN = 9
    STAGE_TIGHTEN = 10
    STAGE_DONE = 11


def pack_action(*args):
    # Filter out None values (optional but useful)
    tensors = [x for x in args if x is not None]

    # Concatenate along last dimension
    return torch.cat(tensors, dim=-1)


def interpolate_dual(
    pos1,
    tgt1,
    quat1,
    quat1_t,
    grip1,
    pos2,
    tgt2,
    quat2,
    quat2_t,
    grip2,
    steps,
):
    pos_traj1, quat_traj1 = batched_interpolate_linear(
        pos1, tgt1, quat1, quat1_t, steps
    )
    pos_traj2, quat_traj2 = batched_interpolate_linear(
        pos2, tgt2, quat2, quat2_t, steps
    )

    B, T = pos_traj1.shape[:2]

    # (B, T, 7)
    eef1 = torch.cat([pos_traj1, quat_traj1], dim=-1)
    eef2 = torch.cat([pos_traj2, quat_traj2], dim=-1)

    # (B, T, 1)
    grip1 = torch.full((B, T, 1), grip1, device=pos1.device)
    grip2 = torch.full((B, T, 1), grip2, device=pos1.device)

    # final: (B, T, D)
    action = pack_action(eef1, grip1, eef2, grip2)

    return action


def hold_with_gripper(pos1, quat1, pos2, quat2, cap_grip, steps, tube_grip=None):
    B = pos1.shape[0]

    eef1 = torch.cat([pos1, quat1], dim=-1)  # (B, 7)
    eef2 = torch.cat([pos2, quat2], dim=-1)  # (B, 7)

    if tube_grip is None:
        tube_grip = cap_grip
    cap_g = torch.full((B, 1), cap_grip, device=pos1.device)  # (B, 1)
    tube_g = torch.full((B, 1), tube_grip, device=pos1.device)  # (B, 1)

    action = pack_action(eef1, cap_g, eef2, tube_g)  # (B, D)

    # Expand to (B, T, D)
    traj = action.unsqueeze(1).expand(B, steps, action.shape[-1])

    return traj
