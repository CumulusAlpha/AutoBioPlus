"""ShadowPipetteGeometry: geometry support for the task policy."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.policy_helpers.shadow_pipette_manipulate_common import (
    pipette_vertical_quat_wxyz,
)
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_mul,
    quat_rotate_vec,
)


class ShadowPipetteGeometry:
    """Provide geometry methods used by the task policy."""

    def _tip_aligned_palm_pose_w(
        self,
        obs,
        idxs: torch.Tensor,
        target_tip_pose_w: torch.Tensor,
        target_palm_quat_wxyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current_palm_pose_w = self._get_current_control_pose_w(obs)[idxs]
        current_tip_pose_w = self._get_pipette_tip_pose_w(obs)[idxs]
        if target_palm_quat_wxyz is not None:
            palm_to_tip_pose = compose_pose_wxyz(
                pose_inv_wxyz(current_palm_pose_w),
                current_tip_pose_w,
            )
            target_palm_pose_w = current_palm_pose_w.clone()
            target_palm_quat_wxyz = F.normalize(target_palm_quat_wxyz, dim=-1)
            target_palm_pose_w[:, 3:7] = target_palm_quat_wxyz
            target_palm_pose_w[:, :3] = target_tip_pose_w[:, :3] - quat_rotate_vec(
                target_palm_quat_wxyz,
                palm_to_tip_pose[:, :3],
            )
            return target_palm_pose_w

        tip_to_target_delta_w = compose_pose_wxyz(
            target_tip_pose_w,
            pose_inv_wxyz(current_tip_pose_w),
        )
        return compose_pose_wxyz(tip_to_target_delta_w, current_palm_pose_w)

    def _verticalized_palm_quat(
        self,
        pipette_pose_w: torch.Tensor,
        current_palm_pose_w: torch.Tensor,
    ) -> torch.Tensor:
        target_pipette_pose_w = pipette_pose_w.clone()
        target_pipette_pose_w[:, 3:7] = pipette_vertical_quat_wxyz(
            pipette_pose_w[:, 3:7]
        )
        relative_pose_w = compose_pose_wxyz(
            target_pipette_pose_w,
            pose_inv_wxyz(pipette_pose_w),
        )
        return F.normalize(
            quat_mul(relative_pose_w[:, 3:7], current_palm_pose_w[:, 3:7]),
            dim=-1,
        )

    def _pregrasp_pose_from_pipette(self, pipette_pose_w: torch.Tensor) -> torch.Tensor:
        rel_pose = self.start_rel_pose.unsqueeze(0).repeat(pipette_pose_w.shape[0], 1)
        mocap_pose_w = compose_pose_wxyz(pipette_pose_w, rel_pose)
        mocap_to_palm = self.mocap_to_palm_pose.unsqueeze(0).repeat(
            pipette_pose_w.shape[0], 1
        )
        return compose_pose_wxyz(mocap_pose_w, mocap_to_palm)

    def _grasp_pose_from_pipette(self, pipette_pose_w: torch.Tensor) -> torch.Tensor:
        rel_pose = self.shadow_init_rel_pose.unsqueeze(0).repeat(
            pipette_pose_w.shape[0], 1
        )
        rel_pose[:, :3] += self._current_grasp_pos_offset().unsqueeze(0)
        mocap_pose_w = compose_pose_wxyz(pipette_pose_w, rel_pose)
        mocap_to_palm = self.mocap_to_palm_pose.unsqueeze(0).repeat(
            pipette_pose_w.shape[0], 1
        )
        return compose_pose_wxyz(mocap_pose_w, mocap_to_palm)
