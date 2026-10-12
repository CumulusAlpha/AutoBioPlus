from __future__ import annotations

from enum import Enum

import mujoco
import torch
import viser

from src.utils.kinematics import batched_interpolate_linear, qpos_linear_interpolate
from src.utils.observations import observation_tensor
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_mul,
)

from .shadow_pipette_manipulate import (
    ShadowPipetteManipulatePolicy,
    pipette_vertical_quat_wxyz,
)


class ShadowPipetteInsertState(Enum):
    MOVE_TO_PREGRASP = 0
    PREGRASP = 1
    GRASP = 2
    GRASP_HOLD = 3
    LIFT = 4
    MOVE_ABOVE_TIP = 5
    LOWER_TO_TIP = 6
    ATTACH_TIP = 7
    LIFT_WITH_TIP = 8


class ShadowPipetteInsertPolicy(ShadowPipetteManipulatePolicy):
    """Shadow policy that grasps a pipette and inserts a tip using EE pose actions."""

    def __init__(
        self,
        action_shape,
        device,
        env,
        interp_steps: int = 20,
        qpos_interp_steps: int = 10,
        tendon_interp_steps: int = 10,
        server: viser.ViserServer = None,
    ) -> None:
        super().__init__(
            action_shape=action_shape,
            device=device,
            env=env,
            interp_steps=interp_steps,
            qpos_interp_steps=qpos_interp_steps,
            tendon_interp_steps=tendon_interp_steps,
            server=server,
        )
        self.cell_dish_frame = None
        if self.stage_text is not None:
            self.stage_text.value = "RESET"
        if self.act_dim != self.expected_action_dim:
            raise ValueError(
                "ShadowPipetteInsertPolicy expects action_dim="
                f"{self.expected_action_dim} (7D EE pose + {self.hand_action_dim} hand actions), "
                f"got {self.act_dim}."
            )

        self.move_above_tip_steps = 50
        self.lower_to_tip_steps = 90
        self.attach_tip_hold_steps = 20
        self.lift_with_tip_steps = 40
        self.tip_approach_height = 0.08
        self.tip_attach_depth = -0.002
        self.lower_xy_gain = 0.25
        self.lower_max_xy_step = 0.001
        self.lower_xy_stop_tol = 0.0005
        self.lower_max_z_step = 0.0015
        self.lower_xy_correction_stop_height = 0.003
        self.tip_lift_height = 0.12
        self.pose_waypoint_pos_tol = 0.008
        self._pipette_attach_pose_w: torch.Tensor | None = None
        self._tip_attached = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._tip_weld_eq_id = self._resolve_mj_id(
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "pipette_tip_weld",
        )
        self._latest_obs = None
        print(
            "Loaded shadow tip policy: "
            f"interp_steps={self.interp_steps}, "
            f"pregrasp_move_steps={self.pregrasp_move_steps}, "
            f"move_to_pregrasp_hold_steps={self.move_to_pregrasp_hold_steps}, "
            f"grasp_hold_steps={self.grasp_hold_steps}, "
            f"lift_steps={self.lift_steps}, lift_height={self.lift_height}, "
            f"pos_tol={self.pos_tol}, rot_tol={self.rot_tol}, "
            f"pregrasp_pos_tol={self.pregrasp_pos_tol}, "
            f"pregrasp_rot_tol={self.pregrasp_rot_tol}, "
            f"qpos_interp_steps={self.qpos_interp_steps}, "
            f"tendon_interp_steps={self.tendon_interp_steps}, "
            f"move_above_tip_steps={self.move_above_tip_steps}, "
            f"lower_to_tip_steps={self.lower_to_tip_steps}, "
            f"attach_tip_hold_steps={self.attach_tip_hold_steps}, "
            f"lift_with_tip_steps={self.lift_with_tip_steps}, "
            f"action_dim={self.act_dim}"
        )

    def __call__(self, obs):
        self._latest_obs = obs
        if self.traj is None:
            self.init_buffers()

        need_plan = self.traj_len == 0
        if need_plan.any():
            self.plan_state_batch(obs, need_plan)

        self._refresh_error_aware_lower_to_tip_targets(obs)
        early_attach_mask = self._maybe_attach_tip_during_lower(obs)
        if early_attach_mask.any():
            self.plan_state_batch(obs, early_attach_mask)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        raw_action = self.traj[batch_idx, self.idx]

        done_mask = self.update_traj_progress(obs, raw_action)
        if done_mask.any():
            self.advance_state(done_mask)
            self.plan_state_batch(obs, done_mask)

        self._update_stage_text()
        self._update_cell_dish_frame(obs)

        return raw_action

    def reset(self) -> None:
        super().reset()
        self._tip_attached.zero_()

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if self.traj is not None:
            self.traj[env_ids] = 0.0
        if self.traj_len is not None:
            self.traj_len[env_ids] = 0
        if self.idx is not None:
            self.idx[env_ids] = 0
        if self.state is not None:
            self.state[env_ids] = ShadowPipetteInsertState.MOVE_TO_PREGRASP.value
        self._tip_attached[env_ids] = False

    def _resolve_mj_id(self, obj_type, name: str) -> int:
        for candidate in (name, f"robot/{name}"):
            obj_id = mujoco.mj_name2id(self.env.sim.mj_model, obj_type, candidate)
            if obj_id >= 0:
                return int(obj_id)
        raise ValueError(f"MuJoCo model does not contain {name!r} or 'robot/{name}'.")

    def init_buffers(self) -> None:
        max_len = (
            max(
                self.pregrasp_move_steps + self.move_to_pregrasp_hold_steps,
                self.interp_steps,
                self.grasp_hold_steps,
                self.lift_steps,
                self.move_above_tip_steps,
                self.lower_to_tip_steps,
                self.attach_tip_hold_steps,
                self.lift_with_tip_steps,
            )
            + 1
        )
        self.traj = torch.zeros(
            (self.num_envs, max_len, self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.traj_len = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.state = torch.full(
            (self.num_envs,),
            ShadowPipetteInsertState.MOVE_TO_PREGRASP.value,
            device=self.device,
            dtype=torch.long,
        )
        self._tip_attached = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )

    def plan_state_batch(self, obs, plan_mask: torch.Tensor) -> None:
        state_table = {
            ShadowPipetteInsertState.MOVE_TO_PREGRASP.value: self._plan_move_to_pregrasp,
            ShadowPipetteInsertState.PREGRASP.value: self._plan_pregrasp,
            ShadowPipetteInsertState.GRASP.value: self._plan_grasp,
            ShadowPipetteInsertState.GRASP_HOLD.value: self._plan_grasp_hold,
            ShadowPipetteInsertState.LIFT.value: self._plan_lift,
            ShadowPipetteInsertState.MOVE_ABOVE_TIP.value: self._plan_move_above_tip,
            ShadowPipetteInsertState.LOWER_TO_TIP.value: self._plan_lower_to_tip,
            ShadowPipetteInsertState.ATTACH_TIP.value: self._plan_attach_tip,
            ShadowPipetteInsertState.LIFT_WITH_TIP.value: self._plan_lift_with_tip,
        }

        for state_value, plan_fn in state_table.items():
            state_mask = plan_mask & (self.state == state_value)
            if not state_mask.any():
                continue
            idxs = torch.where(state_mask)[0]
            self.apply_plan(idxs, plan_fn(obs, idxs))

    def update_traj_progress(self, obs, target_action: torch.Tensor) -> torch.Tensor:
        last = self.idx == (self.traj_len - 1)
        pose_close = self.pose_target_is_reached(obs, target_action, last)
        close = pose_close
        lower_mask = self.state == ShadowPipetteInsertState.LOWER_TO_TIP.value
        close = torch.where(lower_mask, torch.zeros_like(close), close)
        advance = close & (~last)
        self.idx = torch.minimum(self.idx + advance.long(), self.traj_len - 1)
        return (
            last
            & close
            & (self.traj_len > 0)
            & (self.state != ShadowPipetteInsertState.LIFT_WITH_TIP.value)
        )

    def pose_target_is_reached(
        self,
        obs,
        target_action: torch.Tensor,
        last: torch.Tensor,
    ) -> torch.Tensor:
        palm_pose_w = self._get_current_control_pose_w(obs)
        target_pose_w = target_action[:, : self.pose_dim]
        pos_err = torch.linalg.norm(palm_pose_w[:, :3] - target_pose_w[:, :3], dim=-1)

        qa = torch.nn.functional.normalize(palm_pose_w[:, 3:7], dim=-1)
        qb = torch.nn.functional.normalize(target_pose_w[:, 3:7], dim=-1)
        dot = torch.sum(qa * qb, dim=-1).abs().clamp(-1.0, 1.0)
        rot_err = 2.0 * torch.acos(dot)
        self._latest_pos_err = pos_err.detach().clone()
        self._latest_rot_err = rot_err.detach().clone()

        pos_tol = torch.full_like(pos_err, float(self.pos_tol))
        rot_tol = torch.full_like(rot_err, float(self.rot_tol))
        pregrasp = self.state == ShadowPipetteInsertState.PREGRASP.value
        intermediate = ~last
        pos_tol = torch.where(
            intermediate,
            torch.full_like(pos_tol, float(self.pose_waypoint_pos_tol)),
            pos_tol,
        )
        pos_tol = torch.where(
            pregrasp & last,
            torch.full_like(pos_tol, float(self.pregrasp_pos_tol)),
            pos_tol,
        )
        rot_tol = torch.where(
            pregrasp & last,
            torch.full_like(rot_tol, float(self.pregrasp_rot_tol)),
            rot_tol,
        )
        require_rot_close = last | pregrasp
        pose_close = torch.where(
            require_rot_close,
            (pos_err <= pos_tol) & (rot_err <= rot_tol),
            pos_err <= pos_tol,
        )
        return pose_close

    def advance_state(self, done_mask: torch.Tensor) -> None:
        new_state = self.state.clone()
        new_state[
            done_mask
            & (self.state == ShadowPipetteInsertState.MOVE_TO_PREGRASP.value)
        ] = ShadowPipetteInsertState.PREGRASP.value
        new_state[
            done_mask & (self.state == ShadowPipetteInsertState.PREGRASP.value)
        ] = ShadowPipetteInsertState.GRASP.value
        new_state[
            done_mask & (self.state == ShadowPipetteInsertState.GRASP.value)
        ] = ShadowPipetteInsertState.GRASP_HOLD.value
        new_state[
            done_mask & (self.state == ShadowPipetteInsertState.GRASP_HOLD.value)
        ] = ShadowPipetteInsertState.LIFT.value
        new_state[
            done_mask & (self.state == ShadowPipetteInsertState.LIFT.value)
        ] = ShadowPipetteInsertState.MOVE_ABOVE_TIP.value
        new_state[
            done_mask
            & (self.state == ShadowPipetteInsertState.MOVE_ABOVE_TIP.value)
        ] = ShadowPipetteInsertState.LOWER_TO_TIP.value
        new_state[
            done_mask & (self.state == ShadowPipetteInsertState.LOWER_TO_TIP.value)
        ] = ShadowPipetteInsertState.ATTACH_TIP.value
        attach_done = done_mask & (
            self.state == ShadowPipetteInsertState.ATTACH_TIP.value
        )
        if attach_done.any():
            self._tip_attached[attach_done] = True
        new_state[attach_done] = ShadowPipetteInsertState.LIFT_WITH_TIP.value

        self.state = new_state
        self.traj_len[done_mask] = 0
        self.idx[done_mask] = 0

    def _tip_weld_active(self) -> torch.Tensor:
        return self.env.sim.data.eq_active[:, self._tip_weld_eq_id].to(
            device=self.device,
            dtype=torch.bool,
        )

    def _format_stage_text(self) -> str:
        if self.state is None or self.state.numel() == 0:
            return "UNINITIALIZED"

        stage_values = self.state.detach().cpu().tolist()
        stage_names = [
            ShadowPipetteInsertState(int(value)).name for value in stage_values
        ]
        env0_stage = stage_names[0]
        if len(stage_names) == 1 or all(name == env0_stage for name in stage_names):
            return self._format_pose_error_text(env0_stage)

        summary = []
        for state in ShadowPipetteInsertState:
            count = stage_names.count(state.name)
            if count > 0:
                summary.append(f"{state.name}: {count}")
        return self._format_pose_error_text(
            f"env0: {env0_stage}\n" + "\n".join(summary)
        )

    def _format_pose_error_text(self, stage_text: str) -> str:
        if self._latest_pos_err is None:
            return stage_text

        pos_err = float(self._latest_pos_err.detach().cpu()[0].item())
        rot_err = float(self._latest_rot_err.detach().cpu()[0].item())
        return "\n".join(
            (
                stage_text,
                f"env0 EE pos_err: {pos_err:.5f} m",
                f"env0 EE rot_err: {rot_err:.5f} rad",
            )
        )

    def _plan_move_above_tip(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = self._tip_aligned_palm_pose(obs, idxs, self.tip_approach_height)
        self._update_target_frame(target_pose_w, idxs)
        hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.move_above_tip_steps,
        )

    def _plan_lower_to_tip(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = self._tip_aligned_palm_pose(obs, idxs, -self.tip_attach_depth)
        self._update_target_frame(target_pose_w, idxs)
        hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.lower_to_tip_steps,
        )

    def _refresh_error_aware_lower_to_tip_targets(self, obs) -> None:
        lower_mask = self.state == ShadowPipetteInsertState.LOWER_TO_TIP.value
        if not lower_mask.any():
            return

        idxs = torch.where(lower_mask)[0]
        current_palm = self._get_current_control_pose_w(obs)[idxs]
        pipette_attach = self._get_pipette_attach_pose_w(obs)[idxs]
        pipette_pose = self._get_pipette_pose_w(obs)[idxs]
        tip_target = self._get_tip_target_pose_w(obs)[idxs]

        xy_error = tip_target[:, :2] - pipette_attach[:, :2]
        xy_norm = torch.linalg.norm(xy_error, dim=-1)
        z_gap = pipette_attach[:, 2] - tip_target[:, 2]

        xy_step = torch.zeros_like(xy_error)
        correct_xy = (xy_norm > float(self.lower_xy_stop_tol)) & (
            z_gap > float(self.lower_xy_correction_stop_height)
        )
        if correct_xy.any():
            scaled_xy = xy_error[correct_xy] * float(self.lower_xy_gain)
            scaled_norm = torch.linalg.norm(scaled_xy, dim=-1, keepdim=True).clamp_min(
                1e-8
            )
            max_scale = torch.clamp(
                float(self.lower_max_xy_step) / scaled_norm, max=1.0
            )
            xy_step[correct_xy] = scaled_xy * max_scale

        desired_attach_z = tip_target[:, 2] - float(self.tip_attach_depth)
        z_error = desired_attach_z - pipette_attach[:, 2]
        z_step = torch.clamp(z_error, min=-float(self.lower_max_z_step), max=0.0)

        target_pose_w = current_palm.clone()
        target_pose_w[:, :2] += xy_step
        target_pose_w[:, 2] += z_step
        target_pose_w[:, 3:7] = self._verticalized_palm_quat(pipette_pose, current_palm)
        self._update_target_frame(target_pose_w, idxs)

        hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        self.traj[idxs, self.idx[idxs], : self.pose_dim] = target_pose_w
        self.traj[idxs, self.idx[idxs], self.pose_dim :] = hand_action

    def _maybe_attach_tip_during_lower(self, obs) -> torch.Tensor:
        attach_mask = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        lower_mask = (self.state == ShadowPipetteInsertState.LOWER_TO_TIP.value) & (
            ~self._tip_attached
        )
        if not lower_mask.any():
            return attach_mask

        attached_by_environment = lower_mask & self._tip_weld_active()
        if not attached_by_environment.any():
            return attach_mask

        self._tip_attached[attached_by_environment] = True
        self.state[attached_by_environment] = (
            ShadowPipetteInsertState.ATTACH_TIP.value
        )
        self.traj_len[attached_by_environment] = 0
        self.idx[attached_by_environment] = 0
        attach_mask[attached_by_environment] = True
        return attach_mask

    def _plan_attach_tip(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        hold_pose_w = self._get_current_control_pose_w(obs)[idxs]
        self._update_target_frame(hold_pose_w, idxs)
        hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            hold_pose_w,
            hold_pose_w,
            hand_action,
            hand_action,
            self.attach_tip_hold_steps,
        )

    def _plan_lift_with_tip(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = start_pose_w.clone()
        target_pose_w[:, 2] += self.tip_lift_height
        self._update_target_frame(target_pose_w, idxs)
        hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.lift_with_tip_steps,
        )

    def plan_action_traj(
        self,
        start_pose_w: torch.Tensor,
        target_pose_w: torch.Tensor,
        start_hand_action: torch.Tensor,
        target_hand_action: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos=start_pose_w[:, :3],
            end_pos=target_pose_w[:, :3],
            start_quat=start_pose_w[:, 3:7],
            end_quat=target_pose_w[:, 3:7],
            num_steps=num_steps,
        )
        pose_traj = torch.cat((pos_traj, quat_traj), dim=-1)
        hand_traj = qpos_linear_interpolate(
            start_hand_action,
            target_hand_action,
            num_steps,
        )
        return torch.cat((pose_traj, hand_traj), dim=-1)

    def _tip_aligned_palm_pose(
        self,
        obs,
        idxs: torch.Tensor,
        attach_site_z_offset: float | torch.Tensor,
    ) -> torch.Tensor:
        current_palm = self._get_current_control_pose_w(obs)[idxs]
        pipette_attach = self._get_pipette_attach_pose_w(obs)[idxs]
        pipette_pose = self._get_pipette_pose_w(obs)[idxs]
        loose_tip_target = self._get_tip_target_pose_w(obs)[idxs]
        attach_delta = loose_tip_target[:, :3] - pipette_attach[:, :3]
        if torch.is_tensor(attach_site_z_offset):
            attach_delta[:, 2] += attach_site_z_offset.to(
                device=self.device,
                dtype=attach_delta.dtype,
            )
        else:
            attach_delta[:, 2] += float(attach_site_z_offset)
        target = current_palm.clone()
        target[:, :3] = current_palm[:, :3] + attach_delta
        target[:, 3:7] = self._verticalized_palm_quat(pipette_pose, current_palm)
        return target

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
        return torch.nn.functional.normalize(
            quat_mul(relative_pose_w[:, 3:7], current_palm_pose_w[:, 3:7]),
            dim=-1,
        )

    def _get_pipette_attach_pose_w(self, obs) -> torch.Tensor:
        return self._obs_pose(obs, "pipette_tip_attach_site_pose_w")

    def _get_tip_target_pose_w(self, obs) -> torch.Tensor:
        return self._obs_pose(obs, "tip_weld_site_pose_w")

    def _obs_pose(self, obs, key: str) -> torch.Tensor:
        return observation_tensor(
            obs, key, self.device, self.num_envs, select_first=True
        )[:, :7]


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return ShadowPipetteInsertPolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        server=server,
    )
