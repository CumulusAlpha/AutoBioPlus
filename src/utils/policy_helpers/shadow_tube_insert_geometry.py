"""ShadowTubeInsertGeometry: extracted geometry behavior for shadow_tube_insert."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ShadowTubeInsertGeometry:
    """Provide geometry methods used by the task policy."""

    def _ensure_insert_targets(self, obs) -> None:
        targets_ready = (
            self._insert_targets_valid is not None
            and self._tube_to_palm_grasp is not None
            and self._target_tube_pose_w is not None
            and self._target_palm_above_w is not None
            and self._target_tube_pre_insert_w is not None
            and self._target_palm_pre_insert_w is not None
            and self._target_palm_insert_w is not None
            and self._insert_targets_valid.shape[0] == self.num_envs
        )
        if targets_ready and bool(torch.all(self._insert_targets_valid)):
            return

        tube_pose_w = self._extract_tube_pose_w(obs)
        palm_pose_w = self._get_current_palm_pose_w(obs)
        if palm_pose_w is None:
            raise RuntimeError("Cannot build insert target before palm pose is known.")

        tube_to_palm_grasp = self._compose_pose_wxyz(
            self._invert_pose_wxyz(tube_pose_w),
            palm_pose_w,
        )

        slot_pose_w = self._extract_pose_w(obs, "slot1_site_pose_w")
        insert_quat_w = self._align_local_z_to_target_z_wxyz(
            tube_pose_w[:, 3:7],
            slot_pose_w[:, 3:7],
        )
        target_tube_pose_w = self._sample_rack_hole_tube_pose_w(
            slot_pose_w,
            insert_quat_w,
        )
        target_tube_above_move_w = target_tube_pose_w.clone()
        target_tube_above_move_w[:, 2] += self.pre_insert_clearance
        target_tube_above_move_w[:, 3:7] = tube_pose_w[:, 3:7]

        target_tube_pre_insert_w = target_tube_pose_w.clone()
        target_tube_pre_insert_w[:, 2] += self.pre_insert_clearance

        target_palm_above_w = self._compose_pose_wxyz(
            target_tube_above_move_w,
            tube_to_palm_grasp,
        )
        target_palm_pre_insert_w = self._compose_pose_wxyz(
            target_tube_pre_insert_w,
            tube_to_palm_grasp,
        )
        target_palm_insert_w = self._compose_pose_wxyz(
            target_tube_pose_w,
            tube_to_palm_grasp,
        )

        if not targets_ready:
            self._tube_to_palm_grasp = tube_to_palm_grasp
            self._target_tube_pose_w = target_tube_pose_w
            self._target_tube_pre_insert_w = target_tube_pre_insert_w
            self._target_palm_above_w = target_palm_above_w
            self._target_palm_pre_insert_w = target_palm_pre_insert_w
            self._target_palm_insert_w = target_palm_insert_w
            self._insert_targets_valid = torch.ones(
                (self.num_envs,),
                device=self.device,
                dtype=torch.bool,
            )
            return

        invalid = ~self._insert_targets_valid
        if invalid.any():
            self._tube_to_palm_grasp[invalid] = tube_to_palm_grasp[invalid]
            self._target_tube_pose_w[invalid] = target_tube_pose_w[invalid]
            self._target_tube_pre_insert_w[invalid] = target_tube_pre_insert_w[invalid]
            self._target_palm_above_w[invalid] = target_palm_above_w[invalid]
            self._target_palm_pre_insert_w[invalid] = target_palm_pre_insert_w[invalid]
            self._target_palm_insert_w[invalid] = target_palm_insert_w[invalid]
            self._insert_targets_valid[invalid] = True

    def _sample_rack_hole_tube_pose_w(
        self,
        slot_pose_w: torch.Tensor,
        tube_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        origin_pos = slot_pose_w[:, :3]
        slot_quat = F.normalize(slot_pose_w[:, 3:7], dim=-1)
        row_dir = self._quat_rotate_vec_wxyz(
            slot_quat,
            torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.num_envs, 1),
        )
        col_dir = self._quat_rotate_vec_wxyz(
            slot_quat,
            torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.num_envs, 1),
        )

        if self._target_rack_indices is None:
            raise RuntimeError(
                "Target rack indices must be sampled before building insert targets."
            )
        row_idx = self._target_rack_indices[:, 0:1]
        col_idx = self._target_rack_indices[:, 1:2]
        target_pos = (
            origin_pos
            + row_idx.to(origin_pos.dtype) * self.rack_row_gap * row_dir
            + col_idx.to(origin_pos.dtype) * self.rack_col_gap * col_dir
        )
        target_pos[:, 2] += self.rack_hole_height + self.insert_depth_offset

        tube_quat = F.normalize(tube_quat_w, dim=-1)
        return torch.cat((target_pos, tube_quat), dim=-1)

    def _resolve_palm_body_id(self) -> int | None:
        sim = self.env.unwrapped.sim
        mj_model = sim.mj_model
        for body_name in ("hand:rh_palm", "rh_palm", "robot/hand:rh_palm"):
            try:
                return int(mj_model.body(body_name).id)
            except Exception:
                continue
        for body_id in range(mj_model.nbody):
            try:
                body_name = mj_model.body(body_id).name
            except Exception:
                continue
            if body_name.endswith("hand:rh_palm") or body_name.endswith("rh_palm"):
                return int(body_id)
        return None

    def _get_current_palm_pose_w(self, obs=None) -> torch.Tensor | None:
        if obs is not None:
            try:
                return self._extract_pose_w(obs, "palm_body_pose_w")
            except Exception:
                pass

        if self._palm_body_id is None:
            self._palm_body_id = self._resolve_palm_body_id()
        if self._palm_body_id is None:
            return None

        sim = self.env.unwrapped.sim
        mj_data = sim.mj_data
        pos = torch.as_tensor(
            mj_data.xpos[self._palm_body_id],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        quat = torch.as_tensor(
            mj_data.xquat[self._palm_body_id],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        pose = torch.cat((pos, F.normalize(quat, dim=-1)), dim=-1)
        return pose.repeat(self.num_envs, 1)

    def _extract_tube_pose_w(self, obs) -> torch.Tensor:
        return self._extract_pose_w(obs, "tube_body_pose_w")

    def _extract_pose_w(self, obs, key: str) -> torch.Tensor:
        pose = None

        try:
            if key in obs:
                pose = obs[key]
        except Exception:
            pass

        if pose is None:
            try:
                if (key, key) in obs:
                    pose = obs[(key, key)]
            except Exception:
                pass

        if pose is not None and isinstance(pose, dict):
            pose = pose.get(key, pose)

        if pose is None:
            raise KeyError(
                f"Observation missing '{key}'. "
                f"Expected key '{key}' or ('{key}', '{key}')."
            )

        if not isinstance(pose, torch.Tensor):
            pose = torch.as_tensor(pose, device=self.device, dtype=torch.float32)
        else:
            pose = pose.to(device=self.device, dtype=torch.float32)

        if pose.ndim == 3:
            pose = pose[:, 0, :]
        elif pose.ndim == 2:
            pass
        elif pose.ndim == 1 and pose.shape[0] == 7:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
        else:
            raise ValueError(f"Unexpected {key} shape: {tuple(pose.shape)}")

        if pose.shape[1] != 7:
            raise ValueError(f"{key} last dim must be 7, got {pose.shape[1]}")
        return pose

    def quat_mul_wxyz(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    def _quat_conjugate_wxyz(self, q: torch.Tensor) -> torch.Tensor:
        out = q.clone()
        out[:, 1:] = -out[:, 1:]
        return out

    def _quat_rotate_vec_wxyz(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_xyz = q[:, 1:]
        t = 2.0 * torch.cross(q_xyz, v, dim=-1)
        return v + q[:, :1] * t + torch.cross(q_xyz, t, dim=-1)

    def _align_local_z_to_target_z_wxyz(
        self,
        source_quat: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> torch.Tensor:
        source_quat = F.normalize(source_quat, dim=-1)
        target_quat = F.normalize(target_quat, dim=-1)
        local_z = torch.zeros(
            (source_quat.shape[0], 3),
            device=source_quat.device,
            dtype=source_quat.dtype,
        )
        local_z[:, 2] = 1.0
        source_z_w = self._quat_rotate_vec_wxyz(source_quat, local_z)
        target_z_w = self._quat_rotate_vec_wxyz(target_quat, local_z)

        dot = torch.sum(source_z_w * target_z_w, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        cross = torch.cross(source_z_w, target_z_w, dim=-1)
        align_quat = torch.cat((1.0 + dot, cross), dim=-1)

        opposite = dot[:, 0] < -1.0 + 1e-6
        if opposite.any():
            basis_idx = torch.argmin(torch.abs(source_z_w[opposite]), dim=-1)
            basis = F.one_hot(basis_idx, num_classes=3).to(source_z_w.dtype)
            rot_axis = F.normalize(
                torch.cross(source_z_w[opposite], basis, dim=-1),
                dim=-1,
            )
            align_quat[opposite, 0] = 0.0
            align_quat[opposite, 1:4] = rot_axis

        align_quat = F.normalize(align_quat, dim=-1)
        return F.normalize(self.quat_mul_wxyz(align_quat, source_quat), dim=-1)

    def _invert_pose_wxyz(self, pose: torch.Tensor) -> torch.Tensor:
        q_inv = self._quat_conjugate_wxyz(F.normalize(pose[:, 3:7], dim=-1))
        p_inv = self._quat_rotate_vec_wxyz(q_inv, -pose[:, :3])
        return torch.cat((p_inv, q_inv), dim=-1)

    def _compose_pose_wxyz(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        p_lhs = lhs[:, :3]
        q_lhs = F.normalize(lhs[:, 3:7], dim=-1)
        p_rhs = rhs[:, :3]
        q_rhs = F.normalize(rhs[:, 3:7], dim=-1)

        p_out = p_lhs + self._quat_rotate_vec_wxyz(q_lhs, p_rhs)
        q_out = F.normalize(self.quat_mul_wxyz(q_lhs, q_rhs), dim=-1)
        return torch.cat((p_out, q_out), dim=-1)
