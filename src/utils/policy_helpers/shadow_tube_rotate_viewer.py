"""ShadowTubeRotateViewer: extracted viewer behavior for shadow_tube_rotate."""

from __future__ import annotations

import math

import torch

from src.utils.policy_helpers.shadow_tube_rotate_common import (
    ShadowTubeRotateState,
)
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
)


class ShadowTubeRotateViewer:
    """Provide viewer methods used by the task policy."""

    def _update_state_target_palm_frame(self) -> None:
        if self.state_target_palm_frame is None:
            return
        if self.traj is None or self.traj_len is None or self.traj_len[0] <= 0:
            return
        target_idx = int(self.traj_len[0].detach().cpu().item()) - 1
        pose = self.traj[0, target_idx, :7].detach().cpu()
        self.state_target_palm_frame.position = pose[:3].numpy()
        self.state_target_palm_frame.wxyz = pose[3:7].numpy()

    def _update_current_palm_frame(self, obs) -> None:
        if self.current_palm_frame is None:
            return
        pose = self._get_current_palm_pose_w(obs)[0, :7].detach().cpu()
        self.current_palm_frame.position = pose[:3].numpy()
        self.current_palm_frame.wxyz = pose[3:7].numpy()

    def _update_current_left_palm_frame(self, obs) -> None:
        if self.current_left_palm_frame is None:
            return
        pose = self._get_current_tube_holder_palm_pose_w(obs)[0, :7].detach().cpu()
        self.current_left_palm_frame.position = pose[:3].numpy()
        self.current_left_palm_frame.wxyz = pose[3:7].numpy()

    def _update_state_target_left_palm_frame(self, obs) -> None:
        if self.state_target_left_palm_frame is None:
            return
        pose = self._current_state_target_left_palm_pose_w(obs)
        if pose is None:
            return
        pose = pose[0, :7].detach().cpu()
        self.state_target_left_palm_frame.position = pose[:3].numpy()
        self.state_target_left_palm_frame.wxyz = pose[3:7].numpy()

    def _update_current_attached_tube_frame(self, obs) -> None:
        if self.current_attached_tube_frame is None:
            return
        pose = (
            self._extract_pose_w(obs, "attached_tube_body_pose_w")[0, :7].detach().cpu()
        )
        self.current_attached_tube_frame.position = pose[:3].numpy()
        self.current_attached_tube_frame.wxyz = pose[3:7].numpy()

    def _cap_relative_spin_angle_rad(
        self,
        obs,
        update_unwrap: bool = True,
        env_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self._initial_cap_local_pose is None:
            return None
        all_idxs = torch.arange(self.num_envs, device=self.device)
        tube_pose_w = self._current_tube_pose_w(obs, all_idxs)
        cap_pose_w = self._object_pose_from_cap_obs(obs)
        current_local_pose = compose_pose_wxyz(pose_inv_wxyz(tube_pose_w), cap_pose_w)
        spin_angle_wrapped = self._object_spin_angle_rad(
            self._initial_cap_local_pose,
            current_local_pose,
        )
        if update_unwrap:
            if env_mask is None:
                env_mask = torch.ones(
                    (self.num_envs,),
                    device=self.device,
                    dtype=torch.bool,
                )
            return self._unwrap_cap_open_angle(spin_angle_wrapped, env_mask)
        if (
            self._cap_open_angle_unwrapped is not None
            and self._cap_open_angle_valid is not None
        ):
            return torch.where(
                self._cap_open_angle_valid,
                self._cap_open_angle_unwrapped,
                spin_angle_wrapped,
            )
        return spin_angle_wrapped

    def _update_stage_text(self, obs) -> None:
        if self.stage_text is None:
            return
        if self.stage is None or self.idx is None or self.traj_len is None:
            self.stage_text.value = "UNINITIALIZED"
            return
        stage_value = int(self.stage[0].detach().cpu().item())
        try:
            stage_name = ShadowTubeRotateState(stage_value).name
        except ValueError:
            stage_name = f"UNKNOWN({stage_value})"
        idx = int(self.idx[0].detach().cpu().item())
        traj_len = int(self.traj_len[0].detach().cpu().item())
        lines = [
            f"env0 state: {stage_name}",
            f"idx: {idx}/{max(traj_len - 1, 0)}",
        ]
        cap_angle = self._cap_relative_spin_angle_rad(obs, update_unwrap=False)
        if cap_angle is not None:
            angle_rad = float(cap_angle[0].detach().cpu().item())
            lines.append(f"cap-tube angle: {math.degrees(angle_rad):.1f} deg")
            lines.append(f"cap-tube turns: {angle_rad / math.tau:.3f}")
        self.stage_text.value = "\n".join(lines)

    def _current_state_target_left_palm_pose_w(self, obs) -> torch.Tensor | None:
        site_pose_slice = self._tube_holder_slices.get("site_pose")
        if site_pose_slice is None:
            return None
        if (
            self.traj is not None
            and self.traj_len is not None
            and self.stage is not None
            and self.traj_len[0] > 0
            and self.stage[0] == ShadowTubeRotateState.LEFT_PLACE.value
        ):
            target_idx = int(self.traj_len[0].detach().cpu().item()) - 1
            return self.traj[:1, target_idx, site_pose_slice]
        return self._get_current_tube_holder_palm_pose_w(obs)[:1]
