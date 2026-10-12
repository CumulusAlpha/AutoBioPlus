from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mjlab.actuator.actuator import TransmissionType

from src.utils.policy_helpers.shadow_tube_rotate_common import (
    CAP_FINAL_ROTATE_TARGET_ANGLE_RAD as CAP_FINAL_ROTATE_TARGET_ANGLE_RAD,
    CAP_FINAL_ROTATE_TRIGGER_ANGLE_RAD as CAP_FINAL_ROTATE_TRIGGER_ANGLE_RAD,
    CAP_LIFT_ANGLE_DEG as CAP_LIFT_ANGLE_DEG,
    CAP_LIFT_ANGLE_RAD as CAP_LIFT_ANGLE_RAD,
    CAP_LIFT_TURNS as CAP_LIFT_TURNS,
    DEFAULT_GRASP_PAYLOAD as DEFAULT_GRASP_PAYLOAD,
    LEFT_TUBE_TARGET_X_OFFSET as LEFT_TUBE_TARGET_X_OFFSET,
    TUBE_ROTATION_AXIS_LOCAL as TUBE_ROTATION_AXIS_LOCAL,
    ShadowTubeRotateState as ShadowTubeRotateState,
    random_sphere_torch as random_sphere_torch,
)
from src.utils.policy_helpers.shadow_tube_rotate_geometry import (
    ShadowTubeRotateGeometry,
)
from src.utils.policy_helpers.shadow_tube_rotate_replay import (
    ShadowTubeRotateReplay,
)
from src.utils.policy_helpers.shadow_tube_rotate_viewer import (
    ShadowTubeRotateViewer,
)
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
)


class ShadowTubeRotatePolicy(
    ShadowTubeRotateReplay, ShadowTubeRotateGeometry, ShadowTubeRotateViewer
):
    """UR5e+Shadow cap policy that anchors grasp targets directly to cap pose."""

    ShadowTubeRotateState = ShadowTubeRotateState
    _PHASE_KEYS = (
        "pregrasp_act_target",
        "grasp_act_target",
        "squeeze_act_target",
    )

    def __init__(
        self,
        action_shape,
        device,
        env,
        npy_path: str | None = None,
        interp_steps: int = 5,
        qpos_interp_steps: int = 5,
        tendon_interp_steps: int = 5,
        loop: bool = False,
        pose_obs_key: str = "tube_cap_pose_w",
        slot_pose_obs_key: str = "slot1_site_pose_w",
        z_offset: float = -0.005,
        rotate_steps: int = 90,
        flip_cap_xoy: bool = False,
        left_tube_target_pose_w: Sequence[float]
        | np.ndarray
        | torch.Tensor
        | None = None,
        left_tube_target_x_offset: float = LEFT_TUBE_TARGET_X_OFFSET,
        left_tube_target_random_diameter: float = 0.0,
        server=None,
    ) -> None:
        del slot_pose_obs_key
        self.device = device
        self.env = env
        self.server = server
        self.num_envs, self.act_dim = action_shape
        self.pose_dim = 7
        self.expected_row_dim = self.act_dim
        if self.expected_row_dim < self.pose_dim:
            raise ValueError(
                f"ShadowTubeRotatePolicy expects at least 7 action dims, got {self.act_dim}."
            )

        self.pose_obs_key = pose_obs_key
        self.z_offset = float(z_offset)
        self.interp_steps = int(interp_steps)
        self.pose_max_pos_step = 0.024
        self.pose_max_rot_step = 0.36
        self.pregrasp_max_pos_step = 0.003
        self.pregrasp_max_rot_step = 0.045
        self.qpos_interp_steps = max(1, int(qpos_interp_steps))
        self.tendon_interp_steps = max(1, int(tendon_interp_steps))
        self.loop = bool(loop)
        self.rotate_steps = max(1, int(rotate_steps))
        self.rotate_angle_rad = 0.25 * torch.pi
        self.flip_cap_xoy = bool(flip_cap_xoy)
        self.left_tube_target_x_offset = float(left_tube_target_x_offset)
        self.left_tube_target_random_diameter = float(left_tube_target_random_diameter)
        self._configured_left_tube_target_pose_w: torch.Tensor | None = None
        if left_tube_target_pose_w is not None:
            target_pose = torch.as_tensor(
                left_tube_target_pose_w,
                device=self.device,
                dtype=torch.float32,
            ).flatten()
            if target_pose.numel() != 7:
                raise ValueError(
                    "left_tube_target_pose_w must contain xyz + quat(wxyz), "
                    f"got {target_pose.numel()} values."
                )
            target_pose[3:7] = F.normalize(target_pose[3:7], dim=-1)
            target_pose[0] += self.left_tube_target_x_offset
            self._configured_left_tube_target_pose_w = target_pose

        self.cap_to_object_pose = torch.tensor(
            [0.0, 0.0, 0.00617, 1.0, 0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        # Optional compatibility transform for grasp data saved in the opposite
        # cap frame. Keep disabled for parity with the floating Shadow cap setup.
        self.cap_xoy_flip_pose = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        self.mocap_to_palm_offset = torch.tensor(
            [0.0, 0.0, 0.03400001, 1.0, 0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )

        self.pregrasp_pos_tol = 0.004
        self.pregrasp_rot_tol = 0.08
        self.pregrasp_hold_steps = 15
        self.replay_end_hold_steps = 20
        self.lift_height = 0.02
        self.lift_hold_steps = 5
        self.cap_open_turns = CAP_LIFT_TURNS
        self.cap_open_angle_rad = CAP_LIFT_ANGLE_RAD
        self.final_rotate_trigger_angle_rad = CAP_FINAL_ROTATE_TRIGGER_ANGLE_RAD
        self.final_rotate_target_angle_rad = CAP_FINAL_ROTATE_TARGET_ANGLE_RAD
        self.min_rotate_open_steps = 0
        self.left_place_hold_steps = 20
        self.post_pos_tol = 0.005
        self.post_rot_tol = 0.08

        self._joint_mask, self._tendon_mask = self._build_action_masks()
        self._tube_holder_slices = self._build_tube_holder_action_slices()
        self._prev_action: torch.Tensor | None = None
        self._tube_holder_hold_action_targets: dict[str, torch.Tensor] | None = None
        del npy_path
        self._loaded_target_source = "embedded DEFAULT_GRASP_PAYLOAD"
        self._replay_traj_local = self._build_trajectory()
        self._pregrasp_action_tail = self._replay_traj_local[0, 7:].clone()
        self._grasp_action_tail = self._replay_traj_local[-1, 7:].clone()

        self.traj: torch.Tensor | None = None
        self.traj_len: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None
        self.stage: torch.Tensor | None = None
        self._slow_pregrasp: torch.Tensor | None = None
        self._initial_object_pose_w: torch.Tensor | None = None
        self._initial_cap_local_pose: torch.Tensor | None = None
        self._last_cap_open_angle_wrapped: torch.Tensor | None = None
        self._cap_open_angle_unwrapped: torch.Tensor | None = None
        self._cap_open_angle_valid: torch.Tensor | None = None
        self._last_rotate_opened_mask: torch.Tensor | None = None
        self._left_tube_target_base_pose_w: torch.Tensor | None = None
        self._left_tube_target_pose_w: torch.Tensor | None = None
        self._replay_traj_world: torch.Tensor | None = None
        self._pending_env_reset: torch.Tensor | None = None
        self._reset_smoothing_envs: torch.Tensor | None = None
        self._printed_initial_tube_pose = False

        self.state_target_palm_frame = None
        self.current_palm_frame = None
        self.current_left_palm_frame = None
        self.state_target_left_palm_frame = None
        self.current_attached_tube_frame = None
        self.stage_text = None
        if self.server is not None:
            self.state_target_palm_frame = self.server.scene.add_frame(
                name="/shadow_tube_rotate/state_target_palm",
                axes_length=0.08,
                axes_radius=0.004,
                origin_radius=0.01,
            )
            self.current_palm_frame = self.server.scene.add_frame(
                name="/shadow_tube_rotate/current_palm",
                axes_length=0.06,
                axes_radius=0.003,
                origin_radius=0.008,
            )
            self.current_left_palm_frame = self.server.scene.add_frame(
                name="/shadow_tube_rotate/current_left_palm",
                axes_length=0.06,
                axes_radius=0.003,
                origin_radius=0.008,
            )
            self.state_target_left_palm_frame = self.server.scene.add_frame(
                name="/shadow_tube_rotate/state_target_left_palm",
                axes_length=0.08,
                axes_radius=0.004,
                origin_radius=0.01,
            )
            self.current_attached_tube_frame = self.server.scene.add_frame(
                name="/shadow_tube_rotate/current_attached_tube",
                axes_length=0.08,
                axes_radius=0.004,
                origin_radius=0.01,
            )
            with self.server.gui.add_folder("Shadow Cap Policy"):
                self.stage_text = self.server.gui.add_text(
                    label="ShadowTubeRotateState",
                    initial_value="UNINITIALIZED",
                    multiline=True,
                    disabled=True,
                )

        print(
            f"Loaded ShadowTubeRotatePolicy grasp trajectory: {self._replay_traj_local.shape[0]} steps, "
            f"row_dim={self._replay_traj_local.shape[1]}, source={self._loaded_target_source}, "
            f"pose_obs_key={self.pose_obs_key}, rotate_steps={self.rotate_steps}, "
            f"z_offset={self.z_offset}, flip_cap_xoy={self.flip_cap_xoy}, "
            f"cap_open_turns={self.cap_open_turns:.3f}, "
            f"cap_open_angle_rad={self.cap_open_angle_rad:.6f}, "
            f"left_tube_target_x_offset={self.left_tube_target_x_offset}, "
            f"left_tube_target_random_diameter={self.left_tube_target_random_diameter}"
        )

    def reset(self) -> None:
        self._prev_action = None
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.stage = None
        self._slow_pregrasp = None
        self._initial_object_pose_w = None
        self._initial_cap_local_pose = None
        self._last_cap_open_angle_wrapped = None
        self._cap_open_angle_unwrapped = None
        self._cap_open_angle_valid = None
        self._last_rotate_opened_mask = None
        self._left_tube_target_base_pose_w = None
        self._left_tube_target_pose_w = None
        self._replay_traj_world = None
        self._pending_env_reset = None
        self._reset_smoothing_envs = None
        self._tube_holder_hold_action_targets = None
        if self.stage_text is not None:
            self.stage_text.value = "UNINITIALIZED"

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        if self._prev_action is not None:
            self._prev_action[env_ids] = 0.0
        if self.traj is not None:
            self.traj[env_ids] = 0.0
        if self.traj_len is not None:
            self.traj_len[env_ids] = 0
        if self.idx is not None:
            self.idx[env_ids] = 0
        if self.stage is not None:
            self.stage[env_ids] = ShadowTubeRotateState.LEFT_PLACE.value
        if self._slow_pregrasp is not None:
            self._slow_pregrasp[env_ids] = False
        if self._initial_object_pose_w is not None:
            self._initial_object_pose_w[env_ids] = 0.0
        if self._initial_cap_local_pose is not None:
            self._initial_cap_local_pose[env_ids] = 0.0
        self._invalidate_cap_open_angle(env_ids)
        if self._last_rotate_opened_mask is not None:
            self._last_rotate_opened_mask[env_ids] = False
        if self._left_tube_target_base_pose_w is not None:
            self._left_tube_target_base_pose_w[env_ids] = 0.0
        if self._left_tube_target_pose_w is not None:
            self._left_tube_target_pose_w[env_ids] = 0.0
        if self._replay_traj_world is not None:
            self._replay_traj_world[env_ids] = 0.0

        pending = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if (
            self._pending_env_reset is not None
            and self._pending_env_reset.shape[0] == self.num_envs
        ):
            pending |= self._pending_env_reset
        pending[env_ids] = True
        self._pending_env_reset = pending

        reset_smoothing = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        if (
            self._reset_smoothing_envs is not None
            and self._reset_smoothing_envs.shape[0] == self.num_envs
        ):
            reset_smoothing |= self._reset_smoothing_envs
        reset_smoothing[env_ids] = True
        self._reset_smoothing_envs = reset_smoothing

    def __call__(self, obs):
        if (
            self._initial_object_pose_w is None
            or self._initial_object_pose_w.shape[0] != self.num_envs
        ):
            self._sync_cap_reference_pose(obs)
            all_idxs = torch.arange(self.num_envs, device=self.device)
            tube_pose_w = self._current_tube_pose_w(obs, all_idxs)
            self._maybe_print_initial_tube_pose(tube_pose_w)

        if self._pending_env_reset is not None and bool(
            torch.any(self._pending_env_reset)
        ):
            reset_mask = self._pending_env_reset
            reset_idxs = torch.where(reset_mask)[0]
            self._sync_cap_reference_pose(obs, reset_mask)
            self._reset_left_tube_target_pose_for_envs(obs, reset_idxs)
            self._reset_tube_holder_hold_targets_for_envs(obs, reset_idxs)
            self._pending_env_reset = None

        if self.traj is None:
            self.init_buffers()

        need_plan = self.traj_len == 0
        if need_plan.any():
            self._plan_stage_batch(obs, need_plan)

        rotate = self.stage == ShadowTubeRotateState.ROTATE.value
        if bool(torch.any(rotate)):
            rotate_opened = rotate & self._cap_opened_mask(obs, rotate)
            if bool(torch.any(rotate_opened)):
                self._last_rotate_opened_mask = rotate_opened.detach().clone()
                self._advance_stage(rotate_opened)
                self._plan_stage_batch(obs, rotate_opened)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        target_action = self.traj[batch_idx, self.idx]
        action = self._smooth_action(target_action)
        action = self._apply_tube_holder_action(action, obs)

        done_mask = self.update_traj_progress(obs, target_action)
        if done_mask.any():
            self._cache_tube_holder_done_targets(done_mask, target_action, obs)
            self._advance_stage(done_mask)
            self._plan_stage_batch(obs, done_mask)

        self._update_state_target_palm_frame()
        self._update_current_palm_frame(obs)
        self._update_current_left_palm_frame(obs)
        self._update_state_target_left_palm_frame(obs)
        self._update_current_attached_tube_frame(obs)
        self._update_stage_text(obs)
        return action

    def init_buffers(self) -> None:
        max_len = max(
            self._replay_traj_local.shape[0] + self.replay_end_hold_steps, 512
        )
        self.traj = torch.zeros(
            (self.num_envs, max_len, self.expected_row_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.traj_len = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.stage = torch.full(
            (self.num_envs,),
            ShadowTubeRotateState.LEFT_PLACE.value,
            device=self.device,
            dtype=torch.long,
        )
        self._slow_pregrasp = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )

    def _plan_stage_batch(self, obs, plan_mask: torch.Tensor) -> None:
        stage_table = {
            ShadowTubeRotateState.LEFT_PLACE.value: self._plan_left_place,
            ShadowTubeRotateState.REPLAY.value: self._plan_replay,
            ShadowTubeRotateState.ROTATE.value: self._plan_rotate,
            ShadowTubeRotateState.LIFT.value: self._plan_lift,
            ShadowTubeRotateState.CAP_LIFT.value: self._plan_cap_lift,
            ShadowTubeRotateState.HOLD.value: self._plan_hold,
        }
        for stage_value, plan_fn in stage_table.items():
            stage_mask = plan_mask & (self.stage == stage_value)
            if not stage_mask.any():
                continue
            idxs = torch.where(stage_mask)[0]
            self.apply_plan(idxs, plan_fn(obs, idxs))

        pregrasp_mask = plan_mask & (
            self.stage == ShadowTubeRotateState.PREGRASP.value
        )
        for slow in (False, True):
            speed_mask = pregrasp_mask & (self._slow_pregrasp == slow)
            if not speed_mask.any():
                continue
            idxs = torch.where(speed_mask)[0]
            self.apply_plan(idxs, self._plan_pregrasp(obs, idxs, slow=slow))

    def apply_plan(self, idxs: torch.Tensor, traj: torch.Tensor) -> None:
        length = traj.shape[1]
        self.ensure_traj_capacity(length)
        self.traj[idxs] = 0.0
        self.traj[idxs, :length] = traj
        self.traj_len[idxs] = length
        self.idx[idxs] = 0

    def ensure_traj_capacity(self, length: int) -> None:
        if self.traj is not None and self.traj.shape[1] >= length:
            return
        old_traj = self.traj
        new_len = max(length, 2 * old_traj.shape[1] if old_traj is not None else length)
        self.traj = torch.zeros(
            (self.num_envs, new_len, self.expected_row_dim),
            device=self.device,
            dtype=torch.float32,
        )
        if old_traj is not None:
            self.traj[:, : old_traj.shape[1]] = old_traj

    def _plan_pregrasp(
        self,
        obs,
        idxs: torch.Tensor,
        *,
        slow: bool,
    ) -> torch.Tensor:
        replay = self._world_replay_traj(obs, idxs)
        if self._replay_traj_world is None:
            self._replay_traj_world = torch.zeros(
                (self.num_envs, replay.shape[1], self.expected_row_dim),
                device=self.device,
                dtype=torch.float32,
            )
        self._replay_traj_world[idxs] = replay
        target_pose_w = replay[:, 0, :7]
        current_pose_w = self._get_current_palm_pose_w(obs)[idxs]
        step_limits = (
            {
                "max_pos_step": self.pregrasp_max_pos_step,
                "max_rot_step": self.pregrasp_max_rot_step,
            }
            if slow
            else {}
        )
        pose_traj = self._make_step_pose_traj(
            current_pose_w,
            target_pose_w,
            hold_steps=self.pregrasp_hold_steps,
            **step_limits,
        )
        action_tail = self._pregrasp_action_tail.unsqueeze(0).repeat(idxs.numel(), 1)
        return self._pack_action_traj(pose_traj, action_tail)

    def _plan_left_place(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        left_palm_pose_w = self._get_current_tube_holder_palm_pose_w(obs)[idxs]
        target_left_palm_pose_w = self._target_tube_holder_palm_pose_w(obs, idxs)
        left_pose_traj = self._make_step_pose_traj(
            left_palm_pose_w,
            target_left_palm_pose_w,
            hold_steps=self.left_place_hold_steps,
        )

        right_pose_traj = self._get_current_palm_pose_w(obs)[idxs]
        right_pose_traj = right_pose_traj.unsqueeze(1).repeat(
            1, left_pose_traj.shape[1], 1
        )
        action_tail = self._pregrasp_action_tail.unsqueeze(0).repeat(idxs.numel(), 1)
        rows = self._pack_action_traj(right_pose_traj, action_tail)
        return self._write_tube_holder_targets(
            rows, self._tube_holder_plan_targets(obs, left_pose_traj, idxs)
        )

    def _plan_replay(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        del obs
        if self._replay_traj_world is None:
            raise RuntimeError(
                "World replay trajectory must be cached during PREGRASP planning."
            )
        rows = self._replay_traj_world[idxs].clone()
        if self.loop:
            return rows
        hold = rows[:, -1:, :].repeat(1, self.replay_end_hold_steps, 1)
        return torch.cat((rows, hold), dim=1)

    def _plan_rotate(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        object_pose_w = self._spin_corrected_object_pose_w(obs, idxs)
        start_pose_w = self._get_current_palm_pose_w(obs)[idxs]
        spin_angle = self._cap_relative_spin_angle_rad(obs, update_unwrap=False)
        rotate_angle_rad = self._adaptive_rotate_angle_rad(
            None if spin_angle is None else spin_angle[idxs],
            idxs.numel(),
        )
        pose_traj = self._build_rotate_pose_traj_w(
            start_pose_w,
            object_pose_w,
            rotate_angle_rad,
        )
        return self._pack_action_traj(
            pose_traj,
            self._grasp_action_tail.unsqueeze(0).repeat(idxs.numel(), 1),
        )

    def _plan_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        pose_traj = self._plan_lift_pose_traj(obs, idxs, slow=True)
        return self._pack_action_traj(
            pose_traj,
            self._pregrasp_action_tail.unsqueeze(0).repeat(idxs.numel(), 1),
        )

    def _plan_cap_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        pose_traj = self._plan_lift_pose_traj(obs, idxs, slow=False)
        return self._pack_action_traj(
            pose_traj,
            self._grasp_action_tail.unsqueeze(0).repeat(idxs.numel(), 1),
        )

    def _plan_hold(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        pose_traj = self._get_current_palm_pose_w(obs)[idxs].unsqueeze(1)
        return self._pack_action_traj(
            pose_traj,
            self._grasp_action_tail.unsqueeze(0).repeat(idxs.numel(), 1),
        )

    def _plan_lift_pose_traj(
        self,
        obs,
        idxs: torch.Tensor,
        *,
        slow: bool,
    ) -> torch.Tensor:
        start_pose_w = self._get_current_palm_pose_w(obs)[idxs]
        tube_pose_w = self._current_tube_pose_w(obs, idxs)
        tube_to_palm_pose = compose_pose_wxyz(pose_inv_wxyz(tube_pose_w), start_pose_w)
        target_tube_pose_w = tube_pose_w.clone()
        target_tube_pose_w[:, :3] += self._tube_axis_w(tube_pose_w) * self.lift_height
        target_pose_w = compose_pose_wxyz(target_tube_pose_w, tube_to_palm_pose)
        step_limits = (
            {
                "max_pos_step": self.pregrasp_max_pos_step,
                "max_rot_step": self.pregrasp_max_rot_step,
            }
            if slow
            else {}
        )
        pose_traj = self._make_step_pose_traj(
            start_pose_w,
            target_pose_w,
            hold_steps=self.lift_hold_steps,
            **step_limits,
        )
        return pose_traj

    def _world_replay_traj(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        object_pose_w = self._spin_corrected_object_pose_w(obs, idxs)
        rows = self._replay_traj_local.unsqueeze(0).repeat(idxs.numel(), 1, 1)
        local_pose = rows[:, :, :7].reshape(-1, 7)
        object_pose = (
            object_pose_w.unsqueeze(1).repeat(1, rows.shape[1], 1).reshape(-1, 7)
        )
        palm_pose = self._local_mocap_pose_to_palm_pose_w(object_pose, local_pose)
        rows[:, :, :7] = palm_pose.reshape(idxs.numel(), rows.shape[1], 7)
        return rows

    def update_traj_progress(self, obs, target_action: torch.Tensor) -> torch.Tensor:
        last = self.idx == (self.traj_len - 1)
        replay = self.stage == ShadowTubeRotateState.REPLAY.value
        rotate = self.stage == ShadowTubeRotateState.ROTATE.value
        hold = self.stage == ShadowTubeRotateState.HOLD.value
        pose_close = self.target_action_is_reached(obs, target_action)
        # Intermediate waypoints define a time-parameterized trajectory. Waiting
        # for every point to be reached turns smooth motion into stop-and-go
        # tracking; only the final waypoint needs closed-loop convergence.
        advance = ~last
        self.idx = torch.minimum(self.idx + advance.long(), self.traj_len - 1)
        rotate_opened = torch.zeros_like(rotate)
        if bool(torch.any(rotate)):
            rotate_opened = rotate & self._cap_opened_mask(obs, rotate)
        self._last_rotate_opened_mask = rotate_opened.detach().clone()
        done = torch.where(replay, last, last & pose_close)
        done = torch.where(rotate, rotate_opened | (last & pose_close), done)
        done = torch.where(hold, torch.zeros_like(done), done)
        return done & (self.traj_len > 0)

    def _advance_stage(self, done_mask: torch.Tensor) -> None:
        next_stage = self.stage.clone()
        next_stage[
            done_mask & (self.stage == ShadowTubeRotateState.LEFT_PLACE.value)
        ] = ShadowTubeRotateState.PREGRASP.value
        next_stage[
            done_mask & (self.stage == ShadowTubeRotateState.PREGRASP.value)
        ] = ShadowTubeRotateState.REPLAY.value
        replay_done = done_mask & (self.stage == ShadowTubeRotateState.REPLAY.value)
        next_stage[replay_done] = ShadowTubeRotateState.ROTATE.value
        rotate_done = done_mask & (self.stage == ShadowTubeRotateState.ROTATE.value)
        rotate_opened = (
            self._last_rotate_opened_mask
            if self._last_rotate_opened_mask is not None
            else torch.zeros_like(done_mask)
        )
        next_stage[rotate_done & rotate_opened] = (
            ShadowTubeRotateState.CAP_LIFT.value
        )
        rotate_failed = rotate_done & (~rotate_opened)
        next_stage[rotate_failed] = ShadowTubeRotateState.LIFT.value
        lift_done = done_mask & (self.stage == ShadowTubeRotateState.LIFT.value)
        next_stage[lift_done] = ShadowTubeRotateState.PREGRASP.value
        self._slow_pregrasp[lift_done] = True
        next_stage[
            done_mask & (self.stage == ShadowTubeRotateState.CAP_LIFT.value)
        ] = ShadowTubeRotateState.HOLD.value
        if bool(torch.any(replay_done)):
            self._invalidate_cap_open_angle(replay_done)
        self.stage = next_stage
        self.traj_len[done_mask] = 0
        self.idx[done_mask] = 0

    def target_action_is_reached(
        self, obs, target_action: torch.Tensor
    ) -> torch.Tensor:
        current_pose_w = self._get_current_palm_pose_w(obs)
        target_pose_w = target_action[:, :7]
        left_place = self.stage == ShadowTubeRotateState.LEFT_PLACE.value
        site_pose_slice = self._tube_holder_slices.get("site_pose")
        if site_pose_slice is not None and bool(torch.any(left_place)):
            left_pose_w = self._get_current_tube_holder_palm_pose_w(obs)
            current_pose_w = current_pose_w.clone()
            target_pose_w = target_pose_w.clone()
            current_pose_w[left_place] = left_pose_w[left_place]
            target_pose_w[left_place] = target_action[left_place, site_pose_slice]
        pos_err = torch.linalg.norm(
            current_pose_w[:, :3] - target_pose_w[:, :3], dim=-1
        )
        qa = F.normalize(current_pose_w[:, 3:7], dim=-1)
        qb = F.normalize(target_pose_w[:, 3:7], dim=-1)
        rot_err = 2.0 * torch.acos(torch.sum(qa * qb, dim=-1).abs().clamp(-1.0, 1.0))
        pregrasp = (self.stage == ShadowTubeRotateState.PREGRASP.value) | left_place
        pos_tol = torch.where(
            pregrasp,
            torch.full_like(pos_err, self.pregrasp_pos_tol),
            torch.full_like(pos_err, self.post_pos_tol),
        )
        rot_tol = torch.where(
            pregrasp,
            torch.full_like(rot_err, self.pregrasp_rot_tol),
            torch.full_like(rot_err, self.post_rot_tol),
        )
        return (pos_err <= pos_tol) & (rot_err <= rot_tol)

    def _build_action_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_mask = torch.zeros(self.act_dim, dtype=torch.bool, device=self.device)
        tendon_mask = torch.zeros(self.act_dim, dtype=torch.bool, device=self.device)
        idx = 0
        for term in self.env.action_manager._terms.values():
            dim = term.action_dim
            transmission_type = getattr(term.cfg, "transmission_type", None)
            if transmission_type == TransmissionType.JOINT:
                joint_mask[idx : idx + dim] = True
            elif transmission_type == TransmissionType.TENDON:
                tendon_mask[idx : idx + dim] = True
            idx += dim
        return joint_mask, tendon_mask

    def _build_tube_holder_action_slices(self) -> dict[str, slice]:
        slices = {}
        idx = 0
        for name, term in self.env.action_manager._terms.items():
            dim = term.action_dim
            if getattr(term.cfg, "entity_name", None) != "tube_holder_robot":
                idx += dim
                continue
            transmission_type = getattr(term.cfg, "transmission_type", None)
            if transmission_type == TransmissionType.JOINT:
                slices["joint"] = slice(idx, idx + dim)
            elif transmission_type == TransmissionType.TENDON:
                slices["tendon"] = slice(idx, idx + dim)
            else:
                slices["site_pose"] = slice(idx, idx + dim)
            idx += dim
        return slices

    def _apply_tube_holder_action(self, action: torch.Tensor, obs) -> torch.Tensor:
        if not self._tube_holder_slices:
            return action

        action = action.clone()
        hold_targets = self._get_tube_holder_hold_action_targets(obs)
        left_place = self.stage == ShadowTubeRotateState.LEFT_PLACE.value
        site_pose_slice = self._tube_holder_slices.get("site_pose")
        if site_pose_slice is not None and bool(torch.any(~left_place)):
            action[~left_place, site_pose_slice] = hold_targets["site_pose"][
                ~left_place
            ]

        joint_slice = self._tube_holder_slices.get("joint")
        if joint_slice is not None:
            action[:, joint_slice] = hold_targets["joint"]

        tendon_slice = self._tube_holder_slices.get("tendon")
        if tendon_slice is not None:
            action[:, tendon_slice] = hold_targets["tendon"]

        self._prev_action = action.clone()
        return action

    def _tube_holder_plan_targets(
        self,
        obs,
        site_pose_traj: torch.Tensor,
        idxs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hold_targets = self._get_tube_holder_hold_action_targets(obs)
        targets: dict[str, torch.Tensor] = {}
        if "site_pose" in self._tube_holder_slices:
            targets["site_pose"] = site_pose_traj
        if "joint" in self._tube_holder_slices:
            targets["joint"] = hold_targets["joint"][idxs]
        if "tendon" in self._tube_holder_slices:
            targets["tendon"] = hold_targets["tendon"][idxs]
        return targets

    def _write_tube_holder_targets(
        self,
        action: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self._tube_holder_slices:
            return action

        action = action.clone()
        is_traj = action.ndim == 3
        for key, values in targets.items():
            action_slice = self._tube_holder_slices.get(key)
            if action_slice is None:
                continue
            if is_traj and values.ndim == 2:
                values = values.unsqueeze(1).repeat(1, action.shape[1], 1)
            action[..., action_slice] = values
        return action

    def _cache_tube_holder_done_targets(
        self,
        done_mask: torch.Tensor,
        target_action: torch.Tensor,
        obs,
    ) -> None:
        left_done = done_mask & (
            self.stage == ShadowTubeRotateState.LEFT_PLACE.value
        )
        if not bool(torch.any(left_done)):
            return
        self._sync_cap_reference_pose(obs, left_done)
        site_pose_slice = self._tube_holder_slices.get("site_pose")
        if site_pose_slice is None:
            return
        hold_targets = self._get_tube_holder_hold_action_targets(obs)
        hold_targets["site_pose"][left_done] = (
            target_action[left_done, site_pose_slice].detach().clone()
        )

    def _get_tube_holder_hold_action_targets(self, obs) -> dict[str, torch.Tensor]:
        if (
            self._tube_holder_hold_action_targets is not None
            and next(iter(self._tube_holder_hold_action_targets.values())).shape[0]
            == self.num_envs
        ):
            return self._tube_holder_hold_action_targets

        targets: dict[str, torch.Tensor] = {}
        site_pose_slice = self._tube_holder_slices.get("site_pose")
        if site_pose_slice is not None:
            targets["site_pose"] = (
                self._get_current_tube_holder_palm_pose_w(obs).detach().clone()
            )

        joint_slice = self._tube_holder_slices.get("joint")
        if joint_slice is not None:
            tube_holder_qpos = self._extract_qpos(obs, "tube_holder_qpos")
            joint_dim = joint_slice.stop - joint_slice.start
            targets["joint"] = tube_holder_qpos[:, -joint_dim:].detach().clone()

        tendon_slice = self._tube_holder_slices.get("tendon")
        if tendon_slice is not None:
            tendon_dim = tendon_slice.stop - tendon_slice.start
            targets["tendon"] = self._extract_tube_holder_tendon_len(tendon_dim)

        self._tube_holder_hold_action_targets = targets
        return targets

    def _reset_tube_holder_hold_targets_for_envs(
        self, obs, env_ids: torch.Tensor
    ) -> None:
        if self._tube_holder_hold_action_targets is None or env_ids.numel() == 0:
            return

        site_pose_slice = self._tube_holder_slices.get("site_pose")
        if (
            site_pose_slice is not None
            and "site_pose" in self._tube_holder_hold_action_targets
        ):
            site_pose = self._get_current_tube_holder_palm_pose_w(obs).detach().clone()
            self._tube_holder_hold_action_targets["site_pose"][env_ids] = site_pose[
                env_ids
            ]

        joint_slice = self._tube_holder_slices.get("joint")
        if joint_slice is not None and "joint" in self._tube_holder_hold_action_targets:
            tube_holder_qpos = self._extract_qpos(obs, "tube_holder_qpos")
            joint_dim = joint_slice.stop - joint_slice.start
            self._tube_holder_hold_action_targets["joint"][env_ids] = (
                tube_holder_qpos[:, -joint_dim:].detach().clone()[env_ids]
            )

        tendon_slice = self._tube_holder_slices.get("tendon")
        if (
            tendon_slice is not None
            and "tendon" in self._tube_holder_hold_action_targets
        ):
            tendon_dim = tendon_slice.stop - tendon_slice.start
            tendon_len = self._extract_tube_holder_tendon_len(tendon_dim)
            self._tube_holder_hold_action_targets["tendon"][env_ids] = tendon_len[
                env_ids
            ]

    def _extract_tube_holder_tendon_len(self, tendon_dim: int) -> torch.Tensor:
        tendon_len = None
        try:
            entity = self.env.scene["tube_holder_robot"]
            tendon_len = getattr(entity.data, "tendon_len", None)
        except Exception:
            tendon_len = None

        if tendon_len is None:
            return torch.zeros(
                (self.num_envs, tendon_dim), device=self.device, dtype=torch.float32
            )

        tendon_len = tendon_len.to(device=self.device, dtype=torch.float32)
        if tendon_len.ndim == 1:
            tendon_len = tendon_len.unsqueeze(0).repeat(self.num_envs, 1)
        elif tendon_len.ndim == 3:
            tendon_len = tendon_len[:, 0, :]
        elif tendon_len.ndim != 2:
            raise ValueError(
                f"Unexpected tube_holder_robot tendon_len shape: {tuple(tendon_len.shape)}"
            )
        return tendon_len[:, :tendon_dim].detach().clone()

    def _smooth_action(self, raw_act: torch.Tensor) -> torch.Tensor:
        has_joint_smoothing = self.qpos_interp_steps > 1 and bool(
            torch.any(self._joint_mask)
        )
        has_tendon_smoothing = self.tendon_interp_steps > 1 and bool(
            torch.any(self._tendon_mask)
        )
        if not has_joint_smoothing and not has_tendon_smoothing:
            self._prev_action = raw_act.clone()
            self._reset_smoothing_envs = None
            return raw_act
        if self._prev_action is None:
            self._prev_action = raw_act.clone()
            self._reset_smoothing_envs = None
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
        if (
            self._reset_smoothing_envs is not None
            and self._reset_smoothing_envs.shape[0] == raw_act.shape[0]
        ):
            smoothed[self._reset_smoothing_envs] = raw_act[self._reset_smoothing_envs]
            self._reset_smoothing_envs = None
        self._prev_action = smoothed.clone()
        return smoothed


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    from src.tasks.config.shadow_tube_rotate.env_cfgs import (
        SHADOW_CAP_TUBE_BODY_INIT_POS,
        SHADOW_CAP_TUBE_BODY_INIT_ROT,
    )

    del replay_path
    return ShadowTubeRotatePolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        z_offset=-0.005,
        left_tube_target_pose_w=(
            SHADOW_CAP_TUBE_BODY_INIT_POS[0],
            SHADOW_CAP_TUBE_BODY_INIT_POS[1] + 0.12,
            SHADOW_CAP_TUBE_BODY_INIT_POS[2] + 0.15,
            *SHADOW_CAP_TUBE_BODY_INIT_ROT,
        ),
        left_tube_target_random_diameter=0.05,
        server=server,
    )
