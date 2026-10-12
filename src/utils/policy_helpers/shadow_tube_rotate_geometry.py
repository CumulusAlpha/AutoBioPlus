"""ShadowTubeRotateGeometry: extracted geometry behavior for shadow_tube_rotate."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.utils.policy_helpers.shadow_tube_rotate_common import (
    TUBE_ROTATION_AXIS_LOCAL,
    random_sphere_torch,
)
from src.utils.kinematics import batched_interpolate_linear
from src.utils.quat_util import (
    axis_angle_to_quat,
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_mul,
    quat_rotate_vec,
)


class ShadowTubeRotateGeometry:
    """Provide geometry methods used by the task policy."""

    def _sync_cap_reference_pose(
        self, obs, env_mask: torch.Tensor | None = None
    ) -> None:
        object_pose_w = self._object_pose_from_cap_obs(obs).detach().clone()
        all_idxs = torch.arange(self.num_envs, device=self.device)
        tube_pose_w = self._current_tube_pose_w(obs, all_idxs)
        cap_local_pose = (
            compose_pose_wxyz(
                pose_inv_wxyz(tube_pose_w),
                object_pose_w,
            )
            .detach()
            .clone()
        )

        if env_mask is None:
            self._initial_object_pose_w = object_pose_w
            self._initial_cap_local_pose = cap_local_pose
        else:
            if (
                self._initial_object_pose_w is None
                or self._initial_object_pose_w.shape[0] != self.num_envs
            ):
                self._initial_object_pose_w = object_pose_w
            else:
                self._initial_object_pose_w[env_mask] = object_pose_w[env_mask]
            if (
                self._initial_cap_local_pose is None
                or self._initial_cap_local_pose.shape[0] != self.num_envs
            ):
                self._initial_cap_local_pose = cap_local_pose
            else:
                self._initial_cap_local_pose[env_mask] = cap_local_pose[env_mask]

        self._invalidate_cap_open_angle(env_mask)

    def _target_tube_holder_palm_pose_w(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        self._ensure_left_tube_target_pose(obs)
        attached_tube_pose_w = self._extract_pose_w(obs, "attached_tube_body_pose_w")[
            idxs
        ]
        left_palm_pose_w = self._get_current_tube_holder_palm_pose_w(obs)[idxs]
        tube_to_palm_pose = compose_pose_wxyz(
            pose_inv_wxyz(attached_tube_pose_w), left_palm_pose_w
        )
        return compose_pose_wxyz(self._left_tube_target_pose_w[idxs], tube_to_palm_pose)

    def _randomize_left_tube_target_enabled(self) -> bool:
        return self.left_tube_target_random_diameter > 0.0

    def _ensure_left_tube_target_pose(self, obs) -> None:
        if (
            self._left_tube_target_base_pose_w is None
            or self._left_tube_target_base_pose_w.shape[0] != self.num_envs
        ):
            self._left_tube_target_base_pose_w = self._build_left_tube_target_base_pose(
                obs
            )
        if (
            self._left_tube_target_pose_w is None
            or self._left_tube_target_pose_w.shape[0] != self.num_envs
        ):
            self._left_tube_target_pose_w = self._sample_left_tube_target_pose(
                self._left_tube_target_base_pose_w
            )

    def _reset_left_tube_target_pose_for_envs(self, obs, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        base_pose = self._build_left_tube_target_base_pose(obs)
        if (
            self._left_tube_target_base_pose_w is None
            or self._left_tube_target_base_pose_w.shape[0] != self.num_envs
        ):
            self._left_tube_target_base_pose_w = base_pose.detach().clone()
        else:
            self._left_tube_target_base_pose_w[env_ids] = (
                base_pose[env_ids].detach().clone()
            )

        sampled_pose = self._sample_left_tube_target_pose(base_pose)
        if (
            self._left_tube_target_pose_w is None
            or self._left_tube_target_pose_w.shape[0] != self.num_envs
        ):
            self._left_tube_target_pose_w = sampled_pose.detach().clone()
        else:
            self._left_tube_target_pose_w[env_ids] = (
                sampled_pose[env_ids].detach().clone()
            )

    def _build_left_tube_target_base_pose(self, obs) -> torch.Tensor:
        if self._configured_left_tube_target_pose_w is not None:
            target_pose = (
                self._configured_left_tube_target_pose_w.unsqueeze(0)
                .repeat(self.num_envs, 1)
                .detach()
                .clone()
            )
            target_pose[:, :3] += self._env_origins_like(target_pose)
            return target_pose
        target_pose = self._extract_pose_w(obs, "tube_body_pose_w").detach().clone()
        target_pose[:, 0] += self.left_tube_target_x_offset
        return target_pose

    def _env_origins_like(self, pose: torch.Tensor) -> torch.Tensor:
        env_origins = getattr(self.env.scene, "env_origins", None)
        if env_origins is None:
            return torch.zeros((pose.shape[0], 3), device=pose.device, dtype=pose.dtype)
        return env_origins[: pose.shape[0]].to(device=pose.device, dtype=pose.dtype)

    def _sample_left_tube_target_pose(self, base_pose_w: torch.Tensor) -> torch.Tensor:
        target_pose = base_pose_w.detach().clone()
        if self._randomize_left_tube_target_enabled():
            target_pose[:, :3] = random_sphere_torch(
                target_pose[:, :3],
                self.left_tube_target_random_diameter,
                target_pose.shape[0],
            )
        return target_pose

    def _current_tube_pose_w(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        try:
            return self._extract_pose_w(obs, "tube_body_pose_w")[idxs]
        except KeyError:
            return self._spin_corrected_object_pose_w(obs, idxs)

    def _maybe_print_initial_tube_pose(self, tube_pose_w: torch.Tensor) -> None:
        if self._printed_initial_tube_pose:
            return
        pose = tube_pose_w.detach().cpu().numpy()
        print(
            "Initial tube_body_pose_w "
            "(xyz + quat wxyz, before left_tube_target_x_offset):"
        )
        if pose.shape[0] == 1:
            print(f"  left_tube_target_pose_w={pose[0].tolist()}")
        else:
            print(f"  env0 left_tube_target_pose_w={pose[0].tolist()}")
            print(f"  all_env_left_tube_target_pose_w={pose.tolist()}")
        self._printed_initial_tube_pose = True

    def _cap_opened_mask(self, obs, env_mask: torch.Tensor) -> torch.Tensor:
        spin_angle = self._cap_relative_spin_angle_rad(
            obs,
            update_unwrap=True,
            env_mask=env_mask,
        )
        if spin_angle is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        enough_rotate_steps = self.idx >= self.min_rotate_open_steps
        return enough_rotate_steps & (spin_angle <= self.cap_open_angle_rad)

    def _invalidate_cap_open_angle(
        self,
        env_mask: torch.Tensor | None = None,
    ) -> None:
        if (
            self._cap_open_angle_valid is None
            or self._cap_open_angle_valid.shape[0] != self.num_envs
        ):
            self._cap_open_angle_valid = torch.zeros(
                (self.num_envs,),
                device=self.device,
                dtype=torch.bool,
            )
        elif env_mask is None:
            self._cap_open_angle_valid.fill_(False)
        else:
            self._cap_open_angle_valid[env_mask] = False

    def _unwrap_cap_open_angle(
        self,
        angle_wrapped: torch.Tensor,
        env_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self._last_cap_open_angle_wrapped is None
            or self._cap_open_angle_unwrapped is None
            or self._last_cap_open_angle_wrapped.shape != angle_wrapped.shape
        ):
            self._last_cap_open_angle_wrapped = angle_wrapped.detach().clone()
            self._cap_open_angle_unwrapped = angle_wrapped.detach().clone()
            self._cap_open_angle_valid = torch.zeros_like(
                angle_wrapped, dtype=torch.bool
            )

        delta = angle_wrapped - self._last_cap_open_angle_wrapped
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        continued_angle = self._cap_open_angle_unwrapped + delta
        updated_angle = torch.where(
            self._cap_open_angle_valid,
            continued_angle,
            angle_wrapped,
        )
        self._cap_open_angle_unwrapped[env_mask] = updated_angle[env_mask]
        self._last_cap_open_angle_wrapped[env_mask] = angle_wrapped[env_mask]
        self._cap_open_angle_valid[env_mask] = True
        return torch.where(
            self._cap_open_angle_valid,
            self._cap_open_angle_unwrapped,
            angle_wrapped,
        )

    def _tube_axis_w(self, tube_pose_w: torch.Tensor) -> torch.Tensor:
        local_axis = (
            torch.tensor(
                TUBE_ROTATION_AXIS_LOCAL, device=self.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .repeat(tube_pose_w.shape[0], 1)
        )
        return F.normalize(
            self._quat_rotate_vec_wxyz(
                F.normalize(tube_pose_w[:, 3:7], dim=-1),
                local_axis,
            ),
            dim=-1,
        )

    def _object_pose_from_cap_obs(self, obs) -> torch.Tensor:
        cap_pose_w = self._extract_pose_w(obs, self.pose_obs_key)
        cap_pose_w = self._flip_cap_pose_xoy(cap_pose_w)
        offset = self.cap_to_object_pose.unsqueeze(0).repeat(cap_pose_w.shape[0], 1)
        return self._compose_pose_wxyz(cap_pose_w, offset)

    def _spin_corrected_object_pose_w(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_object_pose_w = self._object_pose_from_cap_obs(obs)[idxs]
        if self._initial_object_pose_w is None:
            return current_object_pose_w
        initial_object_pose_w = self._initial_object_pose_w[idxs]
        spin_angle = self._object_spin_angle_rad(
            initial_object_pose_w, current_object_pose_w
        )
        local_axis = (
            torch.tensor(
                TUBE_ROTATION_AXIS_LOCAL, device=self.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .repeat(idxs.numel(), 1)
        )
        axis_w = F.normalize(
            self._quat_rotate_vec_wxyz(
                F.normalize(current_object_pose_w[:, 3:7], dim=-1),
                local_axis,
            ),
            dim=-1,
        )
        correction_quat = self._axis_angle_to_quat_wxyz(axis_w, -spin_angle)
        corrected_quat = F.normalize(
            self.quat_mul_wxyz(correction_quat, current_object_pose_w[:, 3:7]),
            dim=-1,
        )
        return torch.cat((current_object_pose_w[:, :3], corrected_quat), dim=-1)

    def _object_spin_angle_rad(
        self,
        initial_object_pose_w: torch.Tensor,
        current_object_pose_w: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = current_object_pose_w.shape[0]
        local_x = (
            torch.tensor((1.0, 0.0, 0.0), device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        local_axis = (
            torch.tensor(
                TUBE_ROTATION_AXIS_LOCAL, device=self.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        initial_quat = F.normalize(initial_object_pose_w[:, 3:7], dim=-1)
        current_quat = F.normalize(current_object_pose_w[:, 3:7], dim=-1)
        axis_w = F.normalize(
            self._quat_rotate_vec_wxyz(current_quat, local_axis), dim=-1
        )
        initial_x_w = self._project_onto_axis_plane(
            self._quat_rotate_vec_wxyz(initial_quat, local_x),
            axis_w,
        )
        current_x_w = self._project_onto_axis_plane(
            self._quat_rotate_vec_wxyz(current_quat, local_x),
            axis_w,
        )
        sin_angle = torch.sum(
            axis_w * torch.cross(initial_x_w, current_x_w, dim=-1), dim=-1
        )
        cos_angle = torch.sum(initial_x_w * current_x_w, dim=-1).clamp(-1.0, 1.0)
        return torch.atan2(sin_angle, cos_angle)

    def _project_onto_axis_plane(
        self, vec_w: torch.Tensor, axis_w: torch.Tensor
    ) -> torch.Tensor:
        projected = vec_w - axis_w * torch.sum(vec_w * axis_w, dim=-1, keepdim=True)
        return F.normalize(projected, dim=-1)

    def _flip_cap_pose_xoy(self, cap_pose_w: torch.Tensor) -> torch.Tensor:
        if not self.flip_cap_xoy:
            return cap_pose_w
        flip_pose = self.cap_xoy_flip_pose.unsqueeze(0).repeat(cap_pose_w.shape[0], 1)
        return self._compose_pose_wxyz(cap_pose_w, flip_pose)

    def _local_mocap_pose_to_palm_pose_w(
        self,
        object_pose_w: torch.Tensor,
        local_mocap_pose: torch.Tensor,
    ) -> torch.Tensor:
        mocap_pose_w = self._compose_pose_wxyz(object_pose_w, local_mocap_pose)
        offset = self.mocap_to_palm_offset.unsqueeze(0).repeat(mocap_pose_w.shape[0], 1)
        palm_pose_w = self._compose_pose_wxyz(mocap_pose_w, offset)
        palm_pose_w[:, 2] += self.z_offset
        return palm_pose_w

    def _build_rotate_pose_traj_w(
        self,
        start_pose_w: torch.Tensor,
        object_pose_w: torch.Tensor,
        rotate_angle_rad: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = start_pose_w.shape[0]
        rotate_steps = max(
            1,
            math.ceil(
                self.rotate_steps
                * float(torch.max(rotate_angle_rad).detach().cpu())
                / float(self.rotate_angle_rad)
                - 1e-5
            ),
        )
        center = object_pose_w[:, :3]
        object_quat = F.normalize(object_pose_w[:, 3:7], dim=-1)
        local_axis = (
            torch.tensor(
                TUBE_ROTATION_AXIS_LOCAL, device=self.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        axis = F.normalize(self._quat_rotate_vec_wxyz(object_quat, local_axis), dim=-1)
        progress = torch.linspace(
            0.0,
            1.0,
            rotate_steps + 1,
            device=self.device,
            dtype=torch.float32,
        )
        angles = -rotate_angle_rad.unsqueeze(1) * progress.unsqueeze(0)
        rot_quat = self._axis_angle_to_quat_wxyz(
            axis.unsqueeze(1).repeat(1, rotate_steps + 1, 1).reshape(-1, 3),
            angles.reshape(-1),
        )
        offset = start_pose_w[:, :3] - center
        rotate_pos = center.unsqueeze(1) + self._quat_rotate_vec_wxyz(
            rot_quat,
            offset.unsqueeze(1).repeat(1, rotate_steps + 1, 1).reshape(-1, 3),
        ).reshape(batch_size, rotate_steps + 1, 3)
        rotate_quat = F.normalize(
            self.quat_mul_wxyz(
                rot_quat,
                start_pose_w[:, 3:7]
                .unsqueeze(1)
                .repeat(1, rotate_steps + 1, 1)
                .reshape(-1, 4),
            ),
            dim=-1,
        ).reshape(batch_size, rotate_steps + 1, 4)
        return torch.cat((rotate_pos, rotate_quat), dim=-1)

    def _adaptive_rotate_angle_rad(
        self,
        current_spin_angle_rad: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        normal_angle = torch.full(
            (batch_size,),
            float(self.rotate_angle_rad),
            device=self.device,
            dtype=torch.float32,
        )
        if current_spin_angle_rad is None:
            return normal_angle

        projected_angle = current_spin_angle_rad - normal_angle
        use_final_angle = (projected_angle <= self.final_rotate_trigger_angle_rad) & (
            current_spin_angle_rad > self.final_rotate_target_angle_rad
        )
        final_angle = (
            current_spin_angle_rad - self.final_rotate_target_angle_rad
        ).clamp_min(0.0)
        return torch.where(use_final_angle, final_angle, normal_angle)

    def _make_step_pose_traj(
        self,
        start_pose_w: torch.Tensor,
        target_pose_w: torch.Tensor,
        hold_steps: int,
        max_pos_step: float | None = None,
        max_rot_step: float | None = None,
    ) -> torch.Tensor:
        if max_pos_step is None:
            max_pos_step = self.pose_max_pos_step
        if max_rot_step is None:
            max_rot_step = self.pose_max_rot_step
        pos_distance = torch.linalg.norm(
            target_pose_w[:, :3] - start_pose_w[:, :3],
            dim=-1,
        )
        start_quat = F.normalize(start_pose_w[:, 3:7], dim=-1)
        target_quat = F.normalize(target_pose_w[:, 3:7], dim=-1)
        quat_dot = torch.sum(start_quat * target_quat, dim=-1).abs().clamp(-1.0, 1.0)
        rot_distance = 2.0 * torch.acos(quat_dot)
        pos_steps = torch.ceil(pos_distance / max_pos_step)
        rot_steps = torch.ceil(rot_distance / max_rot_step)
        num_steps = max(1, int(torch.max(torch.maximum(pos_steps, rot_steps)).item()))
        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos=start_pose_w[:, :3],
            end_pos=target_pose_w[:, :3],
            start_quat=start_pose_w[:, 3:7],
            end_quat=target_pose_w[:, 3:7],
            num_steps=num_steps,
        )
        traj = torch.cat((pos_traj, quat_traj), dim=-1)
        hold = target_pose_w.unsqueeze(1).repeat(1, max(1, hold_steps), 1)
        return torch.cat((traj, hold), dim=1)

    def _pack_action_traj(
        self, pose_traj: torch.Tensor, action_tail: torch.Tensor
    ) -> torch.Tensor:
        tail = action_tail[:, None, :].repeat(1, pose_traj.shape[1], 1)
        return torch.cat((pose_traj, tail), dim=-1)

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
                f"Observation missing '{key}'. ShadowTubeRotatePolicy only anchors to cap pose."
            )
        if not isinstance(pose, torch.Tensor):
            pose = torch.as_tensor(pose, device=self.device, dtype=torch.float32)
        else:
            pose = pose.to(device=self.device, dtype=torch.float32)
        if pose.ndim == 3:
            pose = pose[:, 0, :]
        elif pose.ndim == 1 and pose.shape[0] == 7:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
        elif pose.ndim != 2:
            raise ValueError(f"Unexpected {key} shape: {tuple(pose.shape)}")
        if pose.shape[1] != 7:
            raise ValueError(f"{key} last dim must be 7, got {pose.shape[1]}")
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _get_current_palm_pose_w(self, obs) -> torch.Tensor:
        return self._extract_pose_w(obs, "palm_body_pose_w")

    def _get_current_tube_holder_palm_pose_w(self, obs) -> torch.Tensor:
        return self._extract_pose_w(obs, "tube_holder_palm_body_pose_w")

    def quat_mul_wxyz(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return quat_mul(q1, q2)

    def _axis_angle_to_quat_wxyz(
        self, axis: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        return axis_angle_to_quat(axis, angle, B=axis.shape[0])

    def _quat_rotate_vec_wxyz(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return quat_rotate_vec(q, v)

    def _compose_pose_wxyz(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return compose_pose_wxyz(lhs, rhs)
