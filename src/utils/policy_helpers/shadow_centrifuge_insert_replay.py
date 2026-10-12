"""ShadowCentrifugeReplay: replay support for the task policy."""

from __future__ import annotations

import torch

from src.utils.kinematics import batched_interpolate_linear


class ShadowCentrifugeReplay:
    """Provide replay methods used by the task policy."""

    def _interpolate_action_tail(
        self,
        start_tail: torch.Tensor,
        target_tail: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        t = torch.linspace(
            0.0,
            1.0,
            max(1, int(num_steps)),
            device=self.device,
            dtype=torch.float32,
        )[None, :, None]
        return (1.0 - t) * start_tail[:, None, :] + t * target_tail[:, None, :]

    def _smooth_action(self, raw_act: torch.Tensor) -> torch.Tensor:
        has_joint_smoothing = self.qpos_interp_steps > 1 and bool(
            torch.any(self._joint_mask)
        )
        has_tendon_smoothing = self.tendon_interp_steps > 1 and bool(
            torch.any(self._tendon_mask)
        )
        if not has_joint_smoothing and not has_tendon_smoothing:
            self._prev_action = raw_act.clone()
            return raw_act

        if self._prev_action is None:
            self._prev_action = raw_act.clone()
            return raw_act

        smoothed = raw_act.clone()
        if has_joint_smoothing:
            joint_alpha = 1.0 / float(self.qpos_interp_steps)
            smoothed[:, self._joint_mask] = self._prev_action[
                :, self._joint_mask
            ] + joint_alpha * (
                raw_act[:, self._joint_mask] - self._prev_action[:, self._joint_mask]
            )

        if has_tendon_smoothing:
            tendon_alpha = 1.0 / float(self.tendon_interp_steps)
            smoothed[:, self._tendon_mask] = self._prev_action[
                :, self._tendon_mask
            ] + tendon_alpha * (
                raw_act[:, self._tendon_mask] - self._prev_action[:, self._tendon_mask]
            )
        self._prev_action = smoothed.clone()
        return smoothed

    def _build_trajectory(self) -> torch.Tensor:
        pre_arr = self._to_tensor(self.PREGRASP_ACT_TARGET)
        grasp_arr = self._to_tensor(self.GRASP_ACT_TARGET)
        squeeze_arr = self._to_tensor(self.SQUEEZE_ACT_TARGET)

        for name, arr in (
            ("PREGRASP_ACT_TARGET", pre_arr),
            ("GRASP_ACT_TARGET", grasp_arr),
            ("SQUEEZE_ACT_TARGET", squeeze_arr),
        ):
            if arr.ndim != 2:
                raise ValueError(f"{name} must be a 2D array, got {tuple(arr.shape)}")
            if arr.shape[1] != self.expected_row_dim:
                raise ValueError(
                    f"{name} shape must be (N, {self.expected_row_dim}) "
                    f"(= action_dim), got {tuple(arr.shape)}"
                )

        phase_trajs: list[torch.Tensor] = [
            self._interpolate_sequence(pre_arr, self.interp_steps),
            self._interpolate_sequence(grasp_arr, self.interp_steps),
            self._interpolate_sequence(squeeze_arr, self.interp_steps),
        ]

        chunks: list[torch.Tensor] = [phase_trajs[0]]
        chunks.append(self._interpolate_pair(phase_trajs[0][-1], phase_trajs[1][0])[1:])
        chunks.append(phase_trajs[1][1:])
        chunks.append(self._interpolate_pair(phase_trajs[1][-1], phase_trajs[2][0])[1:])
        chunks.append(phase_trajs[2][1:])

        traj = torch.cat(chunks, dim=0)
        traj[:, :7] = self._apply_grasp_yaw_offset(traj[:, :7])
        return traj.to(device=self.device, dtype=torch.float32)

    def _apply_grasp_yaw_offset(self, local_pose: torch.Tensor) -> torch.Tensor:
        yaw_pose = self.grasp_yaw_offset_pose.unsqueeze(0).repeat(
            local_pose.shape[0], 1
        )
        return self._compose_pose_wxyz(yaw_pose, local_pose)

    def _to_tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _interpolate_sequence(
        self,
        seq_27: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        if seq_27.shape[0] == 1:
            return seq_27

        pieces: list[torch.Tensor] = []
        for i in range(seq_27.shape[0] - 1):
            pair = self._interpolate_pair(
                seq_27[i],
                seq_27[i + 1],
                num_steps=num_steps,
            )
            if i > 0:
                pair = pair[1:]
            pieces.append(pair)
        return torch.cat(pieces, dim=0)

    def _interpolate_pair(
        self,
        start_row: torch.Tensor,
        end_row: torch.Tensor,
        num_steps: int = 5,
    ) -> torch.Tensor:
        start_pos = start_row[:3].unsqueeze(0)
        start_quat = start_row[3:7].unsqueeze(0)
        start_act = start_row[7:].unsqueeze(0)

        end_pos = end_row[:3].unsqueeze(0)
        end_quat = end_row[3:7].unsqueeze(0)
        end_act = end_row[7:].unsqueeze(0)

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos=start_pos,
            end_pos=end_pos,
            start_quat=start_quat,
            end_quat=end_quat,
            num_steps=num_steps,
        )

        t = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            device=self.device,
            dtype=torch.float32,
        )[None, :, None]
        act_traj = (1.0 - t) * start_act[:, None, :] + t * end_act[:, None, :]

        pair = torch.cat([pos_traj, quat_traj, act_traj], dim=-1)
        return pair[0]
