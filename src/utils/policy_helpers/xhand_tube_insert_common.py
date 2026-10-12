"""XHand insertion state, grasp constants and rack sampling."""

from __future__ import annotations

from enum import Enum

import torch

from src.utils.quat_util import (
    quat_to_mat,
)


class XHandTubeInsertStage(Enum):
    APPROACH = 0
    ALIGN = 1
    GRASP = 2
    LIFT = 3
    MOVE = 4
    PRE_INSERT = 5
    INSERT = 6
    INSERT_HOLD = 7
    RELEASE = 8
    DONE = 9
    HOLD = 10


def sample_target_pos_from_site(site_pose):
    """
    site_pose: (B,7)  [pos, quat]
    return:    (B,7)
    """

    device = site_pose.device
    B = site_pose.shape[0]

    # -----------------------
    # 1. parse pose
    # -----------------------
    origin_pos = site_pose[:, :3]  # (B,3)
    quat = site_pose[:, 3:]  # (B,4)

    # -----------------------
    # 2. quat → rotation matrix
    # -----------------------
    R = quat_to_mat(quat)  # (B,3,3)

    row_dir = R[:, :, 0]  # (B,3)
    col_dir = R[:, :, 1]  # (B,3)

    # -----------------------
    # 3. Randomly sample an index
    # -----------------------
    rows = 2
    cols = 5

    r = torch.randint(0, rows, (B, 1), device=device).float()
    c = torch.randint(0, cols, (B, 1), device=device).float()
    # -----------------------
    # 4. Grid parameters (from XML)
    # -----------------------
    row_gap = 0.036
    col_gap = 0.036
    height = 0.06

    # -----------------------
    # 5. Compute the target position
    # -----------------------
    target_pos = origin_pos + r * row_gap * row_dir + c * col_gap * col_dir

    # height + additional elevation
    target_pos[:, 2] += height + 0.04

    return target_pos


def sample_rack_tube_pos_from_site(site_pose):
    """Sample a 50 ml tube body target using the same rack grid as reset."""

    device = site_pose.device
    B = site_pose.shape[0]

    origin_pos = site_pose[:, :3]
    quat = site_pose[:, 3:]
    R = quat_to_mat(quat)

    row_dir = R[:, :, 1]
    col_dir = R[:, :, 0]

    rows = 5
    cols = 2

    r = torch.randint(0, rows, (B, 1), device=device).float()
    c = torch.randint(0, cols, (B, 1), device=device).float()

    row_gap = 0.036
    col_gap = 0.036

    return origin_pos + r * row_gap * row_dir + c * col_gap * col_dir


_XHAND_INSERT_GRASP_QPOS = {
    "thumb_bend_joint": 1.211,
    "thumb_rota_joint1": 1.032,
    "thumb_rota_joint2": 0.682,
    "index_bend_joint": -0.175,
    "index_joint1": 0.645,
    "index_joint2": 1.075,
    "mid_joint1": 0.547,
    "mid_joint2": 1.047,
    "ring_joint1": 1.017,
    "ring_joint2": 1.663,
    "pinky_joint1": 1.447,
    "pinky_joint2": 1.920,
}


# Fixed T_tube_palm expressed in the tube-local frame.  The workcell frame
# conversion belongs on T_world_tube; compose_pose_wxyz below then gives
# T_world_palm' = C @ T_world_tube @ T_tube_palm without applying C twice.
_XHAND_INSERT_PALM_REL_POSE = (
    -0.0158382644,
    0.131945617,
    0.172442749,
    0.490654776,
    0.509173734,
    0.509173734,
    -0.490654776,
)
