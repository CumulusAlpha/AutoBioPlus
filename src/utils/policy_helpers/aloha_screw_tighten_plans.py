"""AlohaMotionPlans: extracted plans behavior for aloha_screw_tighten."""

from __future__ import annotations

import torch

from src.utils.policy_helpers.aloha_screw_tighten_common import (
    hold_with_gripper,
    interpolate_dual,
    random_sphere_torch,
)
from src.utils.quat_util import (
    axis_angle_to_quat,
    quat_mul,
    quat_rotate_vec,
)


class AlohaMotionPlans:
    """Provide plans methods used by the task policy."""

    def _plan_dual_reach(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)

        self.cap_pos_start[idxs], _ = self._get_cap_pos(obs, idxs)
        self.tube_pos_start[idxs], _ = self._get_tube_pos(obs, idxs)

        B = pos1.shape[0]
        target_quat1 = self._expand_quat(self.target_quat1, B)
        target_quat2 = self._expand_quat(self.target_quat2, B)
        target1 = self._apply_cap_grasp_offset(
            self.cap_pos_start[idxs] + torch.tensor([0, 0, 0.04], device=self.device),
            target_quat1,
        )
        target2 = self.tube_pos_start[idxs] + torch.tensor(
            [-0.1, 0.0, 0.15], device=self.device
        )

        grip1 = grip2 = 0.037
        return interpolate_dual(
            pos1,
            target1,
            quat1,
            target_quat1,
            grip1,
            pos2,
            target2,
            quat2,
            target_quat2,
            grip2,
            self.phase_steps,
        )

    def _plan_mid_approach(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)

        cap_pos, _ = self._get_cap_pos(obs, idxs)
        tube_pos, _ = self._get_tube_pos(obs, idxs)

        B = pos1.shape[0]
        q1 = self._expand_quat(self.target_quat1, B)
        q2 = self._expand_quat(self.target_quat2, B)
        self.cap_pos_mid[idxs] = self._apply_cap_grasp_offset(
            cap_pos + torch.tensor([0, 0, 0.003], device=self.device),
            q1,
        )
        self.tube_pos_mid[idxs] = tube_pos + torch.tensor(
            [0, 0, self.tube_grasp_z_offset],
            device=self.device,
        )
        grip1 = grip2 = 0.037
        return interpolate_dual(
            pos1,
            self.cap_pos_mid[idxs],
            quat1,
            q1,
            grip1,
            pos2,
            self.tube_pos_mid[idxs],
            quat2,
            q2,
            grip2,
            self.phase_steps,
        )

    def _plan_grasp_close(self, obs, idxs):
        self.close_start_step = getattr(self, "global_step", 0)
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)
        self.cap_pos_mid[idxs] = pos1
        return hold_with_gripper(
            pos1,
            quat1,
            pos2,
            quat2,
            cap_grip=self.tighten_cap_grip_close,
            tube_grip=self.tighten_tube_grip_hold,
            steps=10,
        )

    def _plan_grasp_settle(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)
        return hold_with_gripper(
            pos1,
            quat1,
            pos2,
            quat2,
            cap_grip=self.tighten_cap_grip_close,
            tube_grip=self.tighten_tube_grip_hold,
            steps=self.grasp_settle_steps,
        )

    def _apply_cap_grasp_offset(self, pos, quat):
        offset = self.cap_grasp_local_offset.unsqueeze(0).expand(pos.shape[0], -1)
        return pos + quat_rotate_vec(quat, offset)

    def _plan_lift_move(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)

        target1 = self.cap_pos_mid[idxs] + torch.tensor([0, 0, 0.3], device=self.device)

        target2 = pos2 + torch.tensor([0, 0, 0.08], device=self.device)

        B = pos1.shape[0]
        target_quat1 = self._expand_quat(self.target_quat1, B)
        target_quat2 = self._expand_quat(self.target_quat2, B)

        lift_traj = interpolate_dual(
            self.cap_pos_mid[idxs],
            target1,
            quat1,
            target_quat1,
            self.tighten_cap_grip_close,
            pos2,
            target2,
            quat2,
            target_quat2,
            self.tighten_tube_grip_hold,
            self.phase_steps,
        )

        random_center = torch.tensor(
            (0.0, 0.0, 1.1), device=self.device, dtype=pos1.dtype
        )
        cap_pos_random = random_sphere_torch(
            center=random_center, diameter=0.05, B=B
        ).to(self.device)
        cap_pos_random = cap_pos_random + self._env_origins(idxs, cap_pos_random)
        target_quat3 = self._expand_quat(self.target_quat3, B)
        tube_pos_random = cap_pos_random + torch.tensor(
            [0.0, 0.0, -0.0565], device=self.device
        )
        align_traj = interpolate_dual(
            target1,
            cap_pos_random,
            target_quat1,
            target_quat3,
            self.tighten_cap_grip_close,
            target2,
            tube_pos_random,
            target_quat2,
            target_quat2,
            self.tighten_tube_grip_hold,
            self.phase_steps,
        )
        total_traj = torch.concatenate([lift_traj, align_traj[:, 1:]], dim=1)
        return total_traj

    def _plan_align(self, obs, idxs):
        cap_pos, cap_quat = self._get_cap_pos(obs, idxs)
        tube_pos, tube_quat = self._get_tube_pos(obs, idxs)

        cur1, quat1, cur2, quat2 = self._get_dual_eef_pose(obs, idxs)

        err_xy = tube_pos - cap_pos
        err_xy[:, 2] = 0
        err_xy[:, 0] += self.insert_align_x_bias
        target1 = cur1 + err_xy

        B = cur1.shape[0]
        target_quat = self._expand_quat(self.target_quat3, B)

        if self.target_frame is not None:
            self.target_frame.position = target1[0].detach().cpu().numpy()
            self.target_frame.wxyz = target_quat[0].detach().cpu().numpy()
        traj = interpolate_dual(
            cur1,
            target1,
            quat1,
            target_quat,
            self.tighten_cap_grip_close,
            cur2,
            cur2,
            quat2,
            quat2,
            self.tighten_tube_grip_hold,
            14,
        )
        return traj

    def _plan_pre_insert_tilt(self, obs, idxs):
        cap_pos, cap_quat = self._get_cap_pos(obs, idxs)
        tube_pos, _ = self._get_tube_pos(obs, idxs)
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)

        B = pos1.shape[0]
        dtype = pos1.dtype
        cap_axis = self._cap_axis_w(cap_quat)
        world_z = (
            torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )

        rot_axis = torch.cross(cap_axis, world_z, dim=-1)
        rot_axis_norm = torch.linalg.norm(rot_axis, dim=-1, keepdim=True)
        fallback_axis = (
            torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )
        rot_axis = torch.where(
            rot_axis_norm > 1e-6,
            rot_axis / torch.clamp(rot_axis_norm, min=1e-6),
            fallback_axis,
        )
        tilt_angle = torch.acos(
            torch.clamp(torch.sum(cap_axis * world_z, dim=-1), -1.0, 1.0)
        )
        tilt_angle = torch.clamp(tilt_angle, max=float(self.pre_insert_max_tilt))
        correction = axis_angle_to_quat(rot_axis, tilt_angle, B)
        target_quat1 = self._normalize_vec(quat_mul(correction, quat1))

        cap_bottom = cap_pos - cap_axis * float(self.cap_site_to_bottom)
        target_bottom = tube_pos.clone()
        target_bottom[:, 0] += self.insert_align_x_bias
        err_xy = target_bottom - cap_bottom
        err_xy[:, 2] = 0.0
        target1 = pos1 + err_xy

        return interpolate_dual(
            pos1,
            target1,
            quat1,
            target_quat1,
            self.tighten_cap_grip_close,
            pos2,
            pos2,
            quat2,
            quat2,
            self.tighten_tube_grip_hold,
            self.pre_insert_tilt_steps,
        )

    def _plan_insert(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)

        # Bias the insertion column 1 mm outward. The previous +X offset moved
        # the cap inward, so apply the same correction in the opposite direction.
        fixed_pos = pos1 + torch.tensor([-0.001, 0.0, 0.0], device=self.device)
        dz = -0.003
        descent_loops = 5
        insert_phase_steps = 7
        q1 = quat1
        traj_chunks = []

        cur_pos1 = pos1
        cur_quat1 = quat1
        for i in range(descent_loops):
            next_pos = fixed_pos + torch.tensor(
                [0.0, 0.0, dz * (i + 1)], device=self.device
            )
            seg = interpolate_dual(
                cur_pos1,
                next_pos,
                cur_quat1,
                q1,
                self.tighten_cap_grip_close,
                pos2,
                pos2,
                quat2,
                quat2,
                self.tighten_tube_grip_hold,
                insert_phase_steps,
            )
            seg[:, :, :2] = fixed_pos[:, None, :2]
            self._append_segment(traj_chunks, seg)
            cur_pos1 = next_pos
            cur_quat1 = q1

        return torch.cat(traj_chunks, dim=1)

    def _plan_seat_cap(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)
        return hold_with_gripper(
            pos1,
            quat1,
            pos2,
            quat2,
            cap_grip=self.tighten_cap_grip_seat,
            tube_grip=self.tighten_tube_grip_hold,
            steps=self.cap_seat_loose_steps,
        )

    def _plan_spin_align(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)
        cap_pos, _ = self._get_cap_pos(obs, idxs)
        tube_pos, _ = self._get_tube_pos(obs, idxs)
        B = pos1.shape[0]
        target_quat1 = self._expand_quat(self.target_quat3, B)

        target1 = pos1.clone()
        target1[:, :2] = tube_pos[:, :2]
        self.tighten_spin_grasp_z_offset[idxs] = target1[:, 2] - cap_pos[:, 2]

        align = interpolate_dual(
            pos1,
            target1,
            quat1,
            target_quat1,
            self.tighten_cap_grip_seat,
            pos2,
            pos2,
            quat2,
            quat2,
            self.tighten_tube_grip_hold,
            self.spin_align_steps,
        )
        close = hold_with_gripper(
            target1,
            target_quat1,
            pos2,
            quat2,
            cap_grip=self.tighten_cap_grip_close,
            tube_grip=self.tighten_tube_grip_hold,
            steps=self.cap_seat_close_steps,
        )
        return torch.cat([align, close[:, 1:]], dim=1)

    def _plan_tighten(self, obs, idxs):
        if idxs.numel() == 0:
            return self.traj[idxs]

        newly_entered = (self.tighten_loop_idx[idxs] == 0) & (
            self.tighten_substage[idxs] == self._tighten_substage_spin
        )
        if newly_entered.any():
            entered_idxs = idxs[newly_entered]
            cap_pos_now, _ = self._get_cap_pos(obs, entered_idxs)
            self.tighten_mid1_pos[entered_idxs] = cap_pos_now + torch.tensor(
                [0.0, 0.0, 0.01], device=self.device
            )
            self.tighten_mid1_quat[entered_idxs] = self._expand_quat(
                self.target_quat3, entered_idxs.numel()
            )

        sub = self.tighten_substage[idxs]
        planner_by_substage = {
            self._tighten_substage_spin: self._plan_tighten_spin,
            self._tighten_substage_release: self._plan_tighten_release,
            self._tighten_substage_return_mid1: self._plan_tighten_return_mid1,
            self._tighten_substage_return_mid2: self._plan_tighten_return_mid2,
        }

        for substage, planner in planner_by_substage.items():
            if (sub == substage).all():
                return planner(obs, idxs)

        # Mixed substage inside one mask (rare): plan sample-by-sample then stack.
        planned = []
        for one_idx in idxs:
            one = one_idx.unsqueeze(0)
            one_sub = self.tighten_substage[one].item()
            planner = planner_by_substage.get(one_sub, self._plan_tighten_return_mid2)
            planned.append(planner(obs, one))
        max_len = max(plan.shape[1] for plan in planned)
        padded = []
        for plan in planned:
            if plan.shape[1] < max_len:
                pad = plan[:, -1:, :].expand(-1, max_len - plan.shape[1], -1)
                plan = torch.cat([plan, pad], dim=1)
            padded.append(plan)
        return torch.cat(padded, dim=0)

    def _plan_tighten_spin(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)

        B = pos1.shape[0]
        j = self.tighten_loop_idx[idxs]
        spin_loops = torch.where(
            j == 0,
            torch.full_like(j, self.tighten_first_spin_loops),
            torch.full_like(j, self.tighten_spin_loops),
        )

        ang = torch.tensor(self.tighten_angle, device=self.device, dtype=quat1.dtype)
        half = ang / 2
        delta_q = (
            torch.tensor(
                [torch.cos(half), torch.sin(half), 0.0, 0.0],
                device=self.device,
                dtype=quat1.dtype,
            )
            .unsqueeze(0)
            .expand(B, -1)
        )

        cur_pos = pos1
        cur_quat = quat1
        traj_chunks = []

        for i in range(int(spin_loops.max().item())):
            active = i < spin_loops
            if not active.any():
                break

            next_quat = quat_mul(cur_quat, delta_q)
            next_pos = cur_pos + torch.tensor(
                [0.0, 0.0, self.tighten_dz], device=self.device, dtype=cur_pos.dtype
            )
            next_pos = torch.where(active.unsqueeze(-1), next_pos, cur_pos)
            next_quat = torch.where(active.unsqueeze(-1), next_quat, cur_quat)

            seg = interpolate_dual(
                cur_pos,
                next_pos,
                cur_quat,
                next_quat,
                self.tighten_cap_grip_close,
                pos2,
                pos2,
                quat2,
                quat2,
                self.tighten_tube_grip_hold,
                self.tighten_interp_steps,
            )

            self._append_segment(traj_chunks, seg)

            cur_pos = torch.where(active.unsqueeze(-1), next_pos, cur_pos)
            cur_quat = torch.where(active.unsqueeze(-1), next_quat, cur_quat)

        traj = torch.cat(traj_chunks, dim=1)
        self.tighten_mid2_pos[idxs] = cur_pos
        return traj

    def _plan_tighten_release(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)
        return interpolate_dual(
            pos1,
            pos1,
            quat1,
            quat1,
            self.tighten_cap_grip_open,
            pos2,
            pos2,
            quat2,
            quat2,
            self.tighten_tube_grip_hold,
            9,
        )

    def _plan_tighten_return_mid1(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)
        target1 = self.tighten_mid1_pos[idxs]
        target_quat1 = self.tighten_mid1_quat[idxs]
        return interpolate_dual(
            pos1,
            target1,
            quat1,
            target_quat1,
            self.tighten_cap_grip_open,
            pos2,
            pos2,
            quat2,
            quat2,
            self.tighten_tube_grip_hold,
            10,
        )

    def _plan_tighten_return_mid2(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)
        cap_pos, _ = self._get_cap_pos(obs, idxs)
        tube_pos, _ = self._get_tube_pos(obs, idxs)
        B = pos1.shape[0]
        target_quat1 = self._expand_quat(self.target_quat3, B)

        target1 = pos1.clone()
        target1[:, 2] = cap_pos[:, 2] + self.tighten_spin_grasp_z_offset[idxs]
        target1[:, :2] = cap_pos[:, :2]

        align = interpolate_dual(
            pos1,
            target1,
            quat1,
            target_quat1,
            self.tighten_cap_grip_open,
            pos2,
            pos2,
            quat2,
            quat2,
            self.tighten_tube_grip_hold,
            self.spin_align_steps,
        )
        close = hold_with_gripper(
            target1,
            target_quat1,
            pos2,
            quat2,
            cap_grip=self.tighten_cap_grip_close,
            tube_grip=self.tighten_tube_grip_hold,
            steps=self.cap_seat_close_steps,
        )
        return torch.cat([align, close[:, 1:]], dim=1)

    def _plan_done(self, obs, idxs):
        pos1, quat1, pos2, quat2 = self._get_dual_eef_pose(obs, idxs)
        return hold_with_gripper(
            pos1,
            quat1,
            pos2,
            quat2,
            cap_grip=self.tighten_cap_grip_open,
            tube_grip=self.tighten_tube_grip_hold,
            steps=10,
        )
