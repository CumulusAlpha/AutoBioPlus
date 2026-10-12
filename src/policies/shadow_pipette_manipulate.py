from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import viser
from mjlab.actuator.actuator import TransmissionType

from assets.mjcf.robot.ur5e_shadow import (
    UR5E_SHADOW_HAND_JOINT_ORDER,
    UR5E_SHADOW_HAND_TENDON,
)
from src.utils.policy_helpers.shadow_pipette_manipulate_common import (
    BUTTON_PRESSED_QPOS as BUTTON_PRESSED_QPOS,
    GRASP_QPOS as GRASP_QPOS,
    GRASP_QUAT_WXYZ as GRASP_QUAT_WXYZ,
    GRASP_TENDON as GRASP_TENDON,
    MOCAP_TO_PALM_POS as MOCAP_TO_PALM_POS,
    MOCAP_TO_PALM_QUAT as MOCAP_TO_PALM_QUAT,
    PREGRASP_QUAT_WXYZ as PREGRASP_QUAT_WXYZ,
    SHADOW_INIT_REL_POS as SHADOW_INIT_REL_POS,
    SHADOW_INIT_REL_QUAT as SHADOW_INIT_REL_QUAT,
    START_REL_POS as START_REL_POS,
    START_REL_QUAT as START_REL_QUAT,
    ShadowPipetteManipulateState as ShadowPipetteManipulateState,
    pipette_vertical_quat_wxyz as pipette_vertical_quat_wxyz,
)
from src.utils.policy_helpers.shadow_pipette_manipulate_geometry import (
    ShadowPipetteGeometry,
)
from src.utils.policy_helpers.shadow_pipette_manipulate_viewer import (
    ShadowPipetteViewer,
)
from src.utils.kinematics import batched_interpolate_linear, qpos_linear_interpolate
from src.utils.observations import observation_tensor
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_mul,
)


class ShadowPipetteManipulatePolicy(ShadowPipetteGeometry, ShadowPipetteViewer):
    """Pipette-relative pregrasp and grasp policy with an explicit state machine."""

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
        self.device = device
        self.env = env
        self.server = server
        self.num_envs, self.act_dim = action_shape
        self.pose_dim = 7
        self.interp_steps = max(1, int(interp_steps))
        self.default_motion_steps = 40
        self.pregrasp_move_steps = self.default_motion_steps
        self.move_to_pregrasp_hold_steps = 15
        self.grasp_hold_steps = self.default_motion_steps
        self.lift_steps = self.default_motion_steps
        self.move_to_cell_dish_steps = 30
        self.adjust_pipette_pose_steps = self.default_motion_steps
        self.lower_to_cell_dish_steps = self.default_motion_steps
        self.manipulate_button_steps = self.default_motion_steps
        self.post_button_lift_steps = self.default_motion_steps
        self.lift_height = 0.15
        self.post_button_lift_height = 0.10
        self.move_to_cell_dish_tip_clearance = 0.08
        self.lower_to_cell_dish_tip_clearance = 0.03
        self.pos_tol = 0.05
        self.rot_tol = 0.05
        self.pregrasp_pos_tol = 0.03
        self.pregrasp_rot_tol = 0.05
        self.move_to_cell_dish_pos_tol = 0.02
        self.move_to_cell_dish_rot_tol = 0.02
        self.adjust_pipette_pose_pos_tol = 0.015
        self.adjust_pipette_pose_rot_tol = 0.02
        self.lower_to_cell_dish_pos_tol = 0.08
        self.lower_to_cell_dish_rot_tol = 0.08
        self.qpos_interp_steps = max(1, int(qpos_interp_steps))
        self.tendon_interp_steps = max(1, int(tendon_interp_steps))

        self.robot_handle = env.scene["robot"]
        self.hand_action_dim = len(UR5E_SHADOW_HAND_JOINT_ORDER) + len(
            UR5E_SHADOW_HAND_TENDON
        )
        self.expected_action_dim = self.pose_dim + self.hand_action_dim

        self.pregrasp_quat_wxyz = F.normalize(
            torch.tensor(PREGRASP_QUAT_WXYZ, device=self.device, dtype=torch.float32),
            dim=0,
        )
        self.grasp_quat_wxyz = F.normalize(
            torch.tensor(GRASP_QUAT_WXYZ, device=self.device, dtype=torch.float32),
            dim=0,
        )
        self.start_rel_pose = torch.tensor(
            [*START_REL_POS, *START_REL_QUAT],
            device=self.device,
            dtype=torch.float32,
        )
        self.start_rel_pose[3:7] = F.normalize(self.start_rel_pose[3:7], dim=0)
        self.shadow_init_rel_pose = torch.tensor(
            [*SHADOW_INIT_REL_POS, *SHADOW_INIT_REL_QUAT],
            device=self.device,
            dtype=torch.float32,
        )
        self.shadow_init_rel_pose[3:7] = F.normalize(
            self.shadow_init_rel_pose[3:7], dim=0
        )
        self.mocap_to_palm_pose = torch.tensor(
            [*MOCAP_TO_PALM_POS, *MOCAP_TO_PALM_QUAT],
            device=self.device,
            dtype=torch.float32,
        )
        self.mocap_to_palm_pose[3:7] = F.normalize(self.mocap_to_palm_pose[3:7], dim=0)
        self.grasp_hand_action = self._build_grasp_hand_action()
        self.button_pressed_hand_action = self._build_button_pressed_hand_action()
        self.initial_hand_joint_action = self._build_initial_hand_joint_action()

        self._joint_mask, self._tendon_mask = self._build_action_masks()
        self._prev_action: torch.Tensor | None = None
        self._latest_pos_err: torch.Tensor | None = None
        self._latest_rot_err: torch.Tensor | None = None
        self.log_grasp_rel_pose = (
            os.getenv("DEXBIOLAB_LOG_PIPETTE_GRASP_REL", "0") == "1"
        )
        self._logged_grasp_rel_pose = False

        self.traj: torch.Tensor | None = None
        self.traj_len: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None
        self.state: torch.Tensor | None = None

        self.target_frame = None
        self.cell_dish_frame = None
        self.cryovial_frame = None
        self.grasp_offset_sliders = {}
        self.stage_text = None
        self.success_debug_text = None
        if self.server is not None:
            self.target_frame = self.server.scene.add_frame(
                name="/shadow_pipette_manipulate/target",
                axes_length=0.05,
                axes_radius=0.005,
                origin_radius=0.01,
            )
            self.cell_dish_frame = self.server.scene.add_frame(
                name="/shadow_pipette_manipulate/cell_dish",
                axes_length=0.08,
                axes_radius=0.004,
                origin_radius=0.012,
            )
            self.cryovial_frame = self.server.scene.add_frame(
                name="/shadow_pipette_manipulate/cryovial",
                axes_length=0.08,
                axes_radius=0.004,
                origin_radius=0.012,
            )
            with self.server.gui.add_folder("Manipulate Pipette Policy"):
                self.grasp_offset_sliders = {
                    axis: self.server.gui.add_slider(
                        label=f"grasp target {axis} offset (m)",
                        min=-0.03,
                        max=0.03,
                        step=0.001,
                        initial_value=0.0,
                    )
                    for axis in ("x", "y", "z")
                }
                self.stage_text = self.server.gui.add_text(
                    label="Stage",
                    initial_value="UNINITIALIZED",
                    multiline=True,
                    disabled=True,
                )
                self.success_debug_text = self.server.gui.add_text(
                    label="env 0 check_success",
                    initial_value="Waiting for first environment step...",
                    multiline=True,
                    disabled=True,
                )

        print(
            "Loaded manipulate pipette policy: "
            f"interp_steps={self.interp_steps}, "
            f"pregrasp_move_steps={self.pregrasp_move_steps}, "
            f"move_to_pregrasp_hold_steps={self.move_to_pregrasp_hold_steps}, "
            f"grasp_hold_steps={self.grasp_hold_steps}, "
            f"lift_steps={self.lift_steps}, lift_height={self.lift_height}, "
            f"move_to_cell_dish_tip_clearance={self.move_to_cell_dish_tip_clearance}, "
            f"lower_to_cell_dish_tip_clearance={self.lower_to_cell_dish_tip_clearance}, "
            f"move_to_cell_dish_steps={self.move_to_cell_dish_steps}, "
            f"adjust_pipette_pose_steps={self.adjust_pipette_pose_steps}, "
            f"lower_to_cell_dish_steps={self.lower_to_cell_dish_steps}, "
            f"manipulate_button_steps={self.manipulate_button_steps}, "
            f"post_button_lift_steps={self.post_button_lift_steps}, "
            f"post_button_lift_height={self.post_button_lift_height}, "
            f"pos_tol={self.pos_tol}, rot_tol={self.rot_tol}, "
            f"pregrasp_pos_tol={self.pregrasp_pos_tol}, "
            f"pregrasp_rot_tol={self.pregrasp_rot_tol}, "
            f"move_to_cell_dish_pos_tol={self.move_to_cell_dish_pos_tol}, "
            f"move_to_cell_dish_rot_tol={self.move_to_cell_dish_rot_tol}, "
            f"adjust_pipette_pose_pos_tol={self.adjust_pipette_pose_pos_tol}, "
            f"adjust_pipette_pose_rot_tol={self.adjust_pipette_pose_rot_tol}, "
            f"lower_to_cell_dish_pos_tol={self.lower_to_cell_dish_pos_tol}, "
            f"lower_to_cell_dish_rot_tol={self.lower_to_cell_dish_rot_tol}, "
            f"qpos_interp_steps={self.qpos_interp_steps}, "
            f"tendon_interp_steps={self.tendon_interp_steps}, "
            f"action_dim={self.act_dim}"
        )

    def reset(self) -> None:
        self._prev_action = None
        self._latest_pos_err = None
        self._latest_rot_err = None
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.state = None
        if self.stage_text is not None:
            self.stage_text.value = "RESET"
        if self.success_debug_text is not None:
            self.success_debug_text.value = "RESET"

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
            self.state[env_ids] = (
                ShadowPipetteManipulateState.MOVE_TO_PREGRASP.value
            )
        if self._prev_action is not None:
            self._prev_action[env_ids] = 0.0
        if 0 in env_ids.detach().cpu().tolist() and self.success_debug_text is not None:
            self.success_debug_text.value = "RESET"

    def __call__(self, obs):
        if self.traj is None:
            self.init_buffers()

        need_plan = self.traj_len == 0
        if need_plan.any():
            self.plan_state_batch(obs, need_plan)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        raw_action = self.traj[batch_idx, self.idx]
        action = raw_action

        done_mask = self.update_traj_progress(obs, raw_action)
        if done_mask.any():
            defer_adjust_plan = done_mask & (
                self.state == ShadowPipetteManipulateState.MOVE_TO_CELL_DISH.value
            )
            self.advance_state(done_mask)
            immediate_plan = done_mask & (~defer_adjust_plan)
            if immediate_plan.any():
                self.plan_state_batch(obs, immediate_plan)

        self._update_stage_text()
        self._update_success_debug_text()
        self._update_cell_dish_frame(obs)
        self._update_cryovial_frame(obs)

        return action

    def init_buffers(self) -> None:
        max_len = (
            max(
                self.pregrasp_move_steps,
                self.interp_steps,
                self.grasp_hold_steps,
                self.lift_steps,
                self.move_to_cell_dish_steps,
                self.adjust_pipette_pose_steps,
                self.lower_to_cell_dish_steps,
                self.manipulate_button_steps,
                self.post_button_lift_steps,
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
            ShadowPipetteManipulateState.MOVE_TO_PREGRASP.value,
            device=self.device,
            dtype=torch.long,
        )

    def plan_state_batch(self, obs, plan_mask: torch.Tensor) -> None:
        state_table = {
            ShadowPipetteManipulateState.MOVE_TO_PREGRASP.value: self._plan_move_to_pregrasp,
            ShadowPipetteManipulateState.PREGRASP.value: self._plan_pregrasp,
            ShadowPipetteManipulateState.GRASP.value: self._plan_grasp,
            ShadowPipetteManipulateState.GRASP_HOLD.value: self._plan_grasp_hold,
            ShadowPipetteManipulateState.LIFT.value: self._plan_lift,
            ShadowPipetteManipulateState.MOVE_TO_CELL_DISH.value: self._plan_move_to_cell_dish,
            ShadowPipetteManipulateState.ADJUST_PIPETTE_POSE.value: self._plan_adjust_pipette_pose,
            ShadowPipetteManipulateState.LOWER_TO_CELL_DISH.value: self._plan_lower_to_cell_dish,
            ShadowPipetteManipulateState.MANIPULATE_BUTTON.value: self._plan_manipulate_button,
            ShadowPipetteManipulateState.POST_BUTTON_LIFT.value: self._plan_post_button_lift,
        }

        for state_value, plan_fn in state_table.items():
            state_mask = plan_mask & (self.state == state_value)
            if not state_mask.any():
                continue
            idxs = torch.where(state_mask)[0]
            self.apply_plan(idxs, plan_fn(obs, idxs))

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
        new_len = max(length, 2 * old_traj.shape[1])
        self.traj = torch.zeros(
            (self.num_envs, new_len, self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.traj[:, : old_traj.shape[1]] = old_traj

    def update_traj_progress(self, obs, target_action: torch.Tensor) -> torch.Tensor:
        last = self.idx == (self.traj_len - 1)
        pose_close = self.target_action_is_reached(obs, target_action, last)
        advance = pose_close & (~last)
        self.idx = torch.minimum(self.idx + advance.long(), self.traj_len - 1)
        done = (
            last
            & pose_close
            & (self.traj_len > 0)
            & (self.state != ShadowPipetteManipulateState.POST_BUTTON_LIFT.value)
        )

        return done

    def advance_state(self, done_mask: torch.Tensor) -> None:
        new_state = self.state.clone()
        move_done = done_mask & (
            self.state == ShadowPipetteManipulateState.MOVE_TO_PREGRASP.value
        )
        pregrasp_done = done_mask & (
            self.state == ShadowPipetteManipulateState.PREGRASP.value
        )
        grasp_done = done_mask & (
            self.state == ShadowPipetteManipulateState.GRASP.value
        )
        grasp_hold_done = done_mask & (
            self.state == ShadowPipetteManipulateState.GRASP_HOLD.value
        )
        lift_done = done_mask & (
            self.state == ShadowPipetteManipulateState.LIFT.value
        )
        move_to_cell_dish_done = done_mask & (
            self.state == ShadowPipetteManipulateState.MOVE_TO_CELL_DISH.value
        )
        adjust_pipette_pose_done = done_mask & (
            self.state == ShadowPipetteManipulateState.ADJUST_PIPETTE_POSE.value
        )
        lower_to_cell_dish_done = done_mask & (
            self.state == ShadowPipetteManipulateState.LOWER_TO_CELL_DISH.value
        )
        manipulate_button_done = done_mask & (
            self.state == ShadowPipetteManipulateState.MANIPULATE_BUTTON.value
        )

        new_state[move_done] = ShadowPipetteManipulateState.PREGRASP.value
        new_state[pregrasp_done] = ShadowPipetteManipulateState.GRASP.value
        new_state[grasp_done] = ShadowPipetteManipulateState.GRASP_HOLD.value
        new_state[grasp_hold_done] = ShadowPipetteManipulateState.LIFT.value
        new_state[lift_done] = ShadowPipetteManipulateState.MOVE_TO_CELL_DISH.value
        new_state[move_to_cell_dish_done] = (
            ShadowPipetteManipulateState.ADJUST_PIPETTE_POSE.value
        )
        new_state[adjust_pipette_pose_done] = (
            ShadowPipetteManipulateState.LOWER_TO_CELL_DISH.value
        )
        new_state[lower_to_cell_dish_done] = (
            ShadowPipetteManipulateState.MANIPULATE_BUTTON.value
        )
        new_state[manipulate_button_done] = (
            ShadowPipetteManipulateState.POST_BUTTON_LIFT.value
        )

        self.state = new_state
        self.traj_len[done_mask] = 0
        self.idx[done_mask] = 0

    def _plan_move_to_pregrasp(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        pipette_pose_w = self._sample_pipette_pose_w(obs, idxs)
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = self._pregrasp_pose_from_pipette(pipette_pose_w)
        self._update_target_frame(target_pose_w, idxs)
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = start_hand_action.clone()
        joint_dim = len(UR5E_SHADOW_HAND_JOINT_ORDER)
        target_hand_action[:, :joint_dim] = self.initial_hand_joint_action.unsqueeze(0)
        traj = self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            target_hand_action,
            self.pregrasp_move_steps,
        )
        return self._append_hold_steps(traj, self.move_to_pregrasp_hold_steps)

    def _plan_pregrasp(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        pipette_pose_w = self._sample_pipette_pose_w(obs, idxs)
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = self._grasp_pose_from_pipette(pipette_pose_w)
        self._update_target_frame(target_pose_w, idxs)
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            start_hand_action,
            self.interp_steps,
        )

    def _plan_grasp(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        self._update_target_frame(start_pose_w, idxs)
        self._log_current_grasp_rel_pose(obs, idxs, start_pose_w)
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            start_pose_w,
            start_hand_action,
            target_hand_action,
            self.interp_steps,
        )

    def _plan_grasp_hold(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        hold_pose_w = self._get_current_control_pose_w(obs)[idxs]
        self._update_target_frame(hold_pose_w, idxs)
        hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            hold_pose_w,
            hold_pose_w,
            hand_action,
            hand_action,
            self.grasp_hold_steps,
        )

    def _plan_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = start_pose_w.clone()
        target_pose_w[:, 2] += self.lift_height
        self._update_target_frame(target_pose_w, idxs)
        target_hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            target_hand_action,
            target_hand_action,
            self.lift_steps,
        )

    def _plan_move_to_cell_dish(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        cryovial_pose_w = self._get_cryovial_pose_w(obs)[idxs]
        pipette_pose_w = self._sample_pipette_pose_w(obs, idxs)
        target_tip_pose_w = self._get_pipette_tip_pose_w(obs)[idxs].clone()
        target_tip_pose_w[:, :2] = cryovial_pose_w[:, :2]
        target_tip_pose_w[:, 2] = (
            cryovial_pose_w[:, 2] + self.move_to_cell_dish_tip_clearance
        )
        target_pose_w = self._tip_aligned_palm_pose_w(
            obs,
            idxs,
            target_tip_pose_w,
            self._verticalized_palm_quat(pipette_pose_w, start_pose_w),
        )
        self._update_target_frame(target_pose_w, idxs)
        start_hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        target_hand_action = self.button_pressed_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            target_hand_action,
            self.move_to_cell_dish_steps,
        )

    def _plan_adjust_pipette_pose(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        cryovial_pose_w = self._get_cryovial_pose_w(obs)[idxs]
        target_tip_pose_w = self._get_pipette_tip_pose_w(obs)[idxs].clone()
        target_tip_pose_w[:, :2] = cryovial_pose_w[:, :2]
        pipette_pose_w = self._sample_pipette_pose_w(obs, idxs)
        target_pipette_pose_w = pipette_pose_w.clone()
        target_pipette_pose_w[:, 3:7] = pipette_vertical_quat_wxyz(
            pipette_pose_w[:, 3:7]
        )
        pipette_to_target_delta_w = compose_pose_wxyz(
            target_pipette_pose_w,
            pose_inv_wxyz(pipette_pose_w),
        )
        target_tip_pose_w[:, 3:7] = F.normalize(
            quat_mul(pipette_to_target_delta_w[:, 3:7], target_tip_pose_w[:, 3:7]),
            dim=-1,
        )
        target_pose_w = self._tip_aligned_palm_pose_w(obs, idxs, target_tip_pose_w)
        self._update_target_frame(target_pose_w, idxs)
        hand_action = self.button_pressed_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.adjust_pipette_pose_steps,
        )

    def _plan_lower_to_cell_dish(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        cell_dish_pose_w = self._get_cell_dish_pose_w(obs)[idxs]
        target_tip_pose_w = self._get_pipette_tip_pose_w(obs)[idxs].clone()
        target_tip_pose_w[:, :3] = cell_dish_pose_w[:, :3]
        target_tip_pose_w[:, 2] += self.lower_to_cell_dish_tip_clearance
        target_pose_w = self._tip_aligned_palm_pose_w(obs, idxs, target_tip_pose_w)
        self._update_target_frame(target_pose_w, idxs)
        hand_action = self.button_pressed_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.lower_to_cell_dish_steps,
        )

    def _plan_manipulate_button(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        hold_pose_w = self._get_current_control_pose_w(obs)[idxs]
        self._update_target_frame(hold_pose_w, idxs)
        start_hand_action = self.button_pressed_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        target_hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            hold_pose_w,
            hold_pose_w,
            start_hand_action,
            target_hand_action,
            self.manipulate_button_steps,
        )

    def _plan_post_button_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = start_pose_w.clone()
        target_pose_w[:, 2] += self.post_button_lift_height
        self._update_target_frame(target_pose_w, idxs)
        hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.post_button_lift_steps,
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
        hand_traj = qpos_linear_interpolate(
            start_hand_action,
            target_hand_action,
            num_steps,
        )
        return torch.cat((pos_traj, quat_traj, hand_traj), dim=-1)

    def _append_hold_steps(self, traj: torch.Tensor, hold_steps: int) -> torch.Tensor:
        if hold_steps <= 0:
            return traj

        hold = traj[:, -1:, :].repeat(1, int(hold_steps), 1)
        return torch.cat((traj, hold), dim=1)

    def target_action_is_reached(
        self,
        obs,
        target_action: torch.Tensor,
        last: torch.Tensor,
    ) -> torch.Tensor:
        palm_pose_w = self._get_current_control_pose_w(obs)
        pos_err = torch.linalg.norm(palm_pose_w[:, :3] - target_action[:, :3], dim=-1)

        qa = F.normalize(palm_pose_w[:, 3:7], dim=-1)
        qb = F.normalize(target_action[:, 3:7], dim=-1)
        dot = torch.sum(qa * qb, dim=-1).abs().clamp(-1.0, 1.0)
        rot_err = 2.0 * torch.acos(dot)
        self._latest_pos_err = pos_err.detach().clone()
        self._latest_rot_err = rot_err.detach().clone()

        pos_tol, rot_tol = self._target_reach_tolerance()
        pos_close = pos_err <= pos_tol
        pose_close = pos_close & (rot_err <= rot_tol)
        move_to_pregrasp_hold = (
            self.state == ShadowPipetteManipulateState.MOVE_TO_PREGRASP.value
        ) & (self.idx >= self.pregrasp_move_steps)
        adjust_pipette_pose = (
            self.state == ShadowPipetteManipulateState.ADJUST_PIPETTE_POSE.value
        )
        require_rot_close = (
            last
            | (self.state == ShadowPipetteManipulateState.PREGRASP.value)
            | move_to_pregrasp_hold
            | adjust_pipette_pose
        )
        return torch.where(require_rot_close, pose_close, pos_close)

    def _target_reach_tolerance(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_tol = torch.full(
            (self.num_envs,),
            float(self.pos_tol),
            device=self.device,
            dtype=torch.float32,
        )
        rot_tol = torch.full_like(pos_tol, float(self.rot_tol))

        pregrasp = self.state == ShadowPipetteManipulateState.PREGRASP.value
        pos_tol = torch.where(
            pregrasp,
            torch.full_like(pos_tol, float(self.pregrasp_pos_tol)),
            pos_tol,
        )
        rot_tol = torch.where(
            pregrasp,
            torch.full_like(rot_tol, float(self.pregrasp_rot_tol)),
            rot_tol,
        )
        move_to_cell_dish = (
            self.state == ShadowPipetteManipulateState.MOVE_TO_CELL_DISH.value
        )
        pos_tol = torch.where(
            move_to_cell_dish,
            torch.full_like(pos_tol, float(self.move_to_cell_dish_pos_tol)),
            pos_tol,
        )
        rot_tol = torch.where(
            move_to_cell_dish,
            torch.full_like(rot_tol, float(self.move_to_cell_dish_rot_tol)),
            rot_tol,
        )
        adjust_pipette_pose = (
            self.state == ShadowPipetteManipulateState.ADJUST_PIPETTE_POSE.value
        )
        pos_tol = torch.where(
            adjust_pipette_pose,
            torch.full_like(pos_tol, float(self.adjust_pipette_pose_pos_tol)),
            pos_tol,
        )
        rot_tol = torch.where(
            adjust_pipette_pose,
            torch.full_like(rot_tol, float(self.adjust_pipette_pose_rot_tol)),
            rot_tol,
        )
        lower_to_cell_dish = (
            self.state == ShadowPipetteManipulateState.LOWER_TO_CELL_DISH.value
        )
        pos_tol = torch.where(
            lower_to_cell_dish,
            torch.full_like(pos_tol, float(self.lower_to_cell_dish_pos_tol)),
            pos_tol,
        )
        rot_tol = torch.where(
            lower_to_cell_dish,
            torch.full_like(rot_tol, float(self.lower_to_cell_dish_rot_tol)),
            rot_tol,
        )
        return pos_tol, rot_tol

    def _sample_pipette_pose_w(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        return self._get_pipette_pose_w(obs)[idxs].clone()

    def _build_grasp_hand_action(self) -> torch.Tensor:
        joint_targets = [GRASP_QPOS[name] for name in UR5E_SHADOW_HAND_JOINT_ORDER]
        tendon_targets = [GRASP_TENDON[name] for name in UR5E_SHADOW_HAND_TENDON]
        return torch.tensor(
            [*joint_targets, *tendon_targets],
            device=self.device,
            dtype=torch.float32,
        )

    def _build_button_pressed_hand_action(self) -> torch.Tensor:
        joint_targets = [
            BUTTON_PRESSED_QPOS[name] for name in UR5E_SHADOW_HAND_JOINT_ORDER
        ]
        tendon_targets = [GRASP_TENDON[name] for name in UR5E_SHADOW_HAND_TENDON]
        return torch.tensor(
            [*joint_targets, *tendon_targets],
            device=self.device,
            dtype=torch.float32,
        )

    def _build_initial_hand_joint_action(self) -> torch.Tensor:
        init_state = getattr(
            getattr(self.robot_handle, "cfg", None), "init_state", None
        )
        init_joint_pos = (
            getattr(init_state, "joint_pos", {}) if init_state is not None else {}
        )
        joint_targets = [
            float(init_joint_pos.get(joint_name, 0.0))
            for joint_name in UR5E_SHADOW_HAND_JOINT_ORDER
        ]
        return torch.tensor(joint_targets, device=self.device, dtype=torch.float32)

    def _get_current_control_pose_w(self, obs) -> torch.Tensor:
        return observation_tensor(
            obs, "palm_body_pose_w", self.device, self.num_envs, select_first=True
        )[:, :7]

    def _extract_current_hand_action(self, obs) -> torch.Tensor:
        hand_action = torch.zeros(
            (self.num_envs, self.hand_action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        qpos = observation_tensor(
            obs, "current_qpos", self.device, self.num_envs, select_first=True
        )
        joint_dim = len(UR5E_SHADOW_HAND_JOINT_ORDER)
        hand_action[:, :joint_dim] = qpos[:, -joint_dim:]

        tendon_len = self.robot_handle.data.tendon_len.to(
            device=self.device, dtype=torch.float32
        )
        if tendon_len.ndim == 1:
            tendon_len = tendon_len.unsqueeze(0).repeat(self.num_envs, 1)
        hand_action[:, joint_dim:] = tendon_len[:, : len(UR5E_SHADOW_HAND_TENDON)]
        return hand_action

    def _get_pipette_pose_w(self, obs) -> torch.Tensor:
        return observation_tensor(
            obs, "pipette_body_pose_w", self.device, self.num_envs, select_first=True
        )[:, :7]

    def _get_pipette_tip_pose_w(self, obs) -> torch.Tensor:
        try:
            return observation_tensor(
                obs,
                "pipette_tip_site_pose_w",
                self.device,
                self.num_envs,
                select_first=True,
            )[:, :7]
        except KeyError:
            pass

        pipette = self.env.scene["pipette"]
        pose = getattr(pipette.data, "site_pose_w", None)
        if pose is None:
            raise KeyError(
                "Observation missing 'pipette_tip_site_pose_w' and scene['pipette'] "
                "does not expose site_pose_w."
            )

        pose = pose.to(device=self.device, dtype=torch.float32)
        if pose.ndim == 2:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1, 1)
        return pose[:, 0, :7]

    def _get_cryovial_pose_w(self, obs) -> torch.Tensor:
        try:
            return observation_tensor(
                obs, "cryovial_pose_w", self.device, self.num_envs, select_first=True
            )[:, :7]
        except KeyError:
            pass

        try:
            cryovial = self.env.scene["cryovial"]
        except KeyError as exc:
            raise KeyError(
                "Observation missing 'cryovial_pose_w' and scene['cryovial'] does not exist."
            ) from exc

        pose = getattr(cryovial.data, "root_link_pose_w", None)
        if pose is None:
            raise KeyError(
                "Observation missing 'cryovial_pose_w' and scene['cryovial'] "
                "does not expose root_link_pose_w."
            )

        pose = pose.to(device=self.device, dtype=torch.float32)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
        return pose[:, :7]

    def _get_cell_dish_pose_w(self, obs) -> torch.Tensor:
        try:
            return observation_tensor(
                obs, "cryovial_pose_w", self.device, self.num_envs, select_first=True
            )[:, :7]
        except KeyError:
            pass

        try:
            return observation_tensor(
                obs,
                "cell_dish_body_pose_w",
                self.device,
                self.num_envs,
                select_first=True,
            )[:, :7]
        except KeyError:
            pass

        try:
            cell_dish = self.env.scene["cryovial"]
        except KeyError:
            try:
                cell_dish = self.env.scene["cell_dish"]
            except KeyError as exc:
                raise KeyError(
                    "Observation missing 'cryovial_pose_w' and 'cell_dish_body_pose_w', "
                    "and neither scene['cryovial'] nor scene['cell_dish'] exists."
                ) from exc
        if cell_dish is None:
            raise KeyError(
                "Observation missing 'cryovial_pose_w' and 'cell_dish_body_pose_w', "
                "and neither scene['cryovial'] nor scene['cell_dish'] exists."
            )
        pose = getattr(cell_dish.data, "root_link_pose_w", None)
        if pose is None:
            raise KeyError(
                "Observation missing 'cryovial_pose_w' and 'cell_dish_body_pose_w', "
                "and the scene target does not expose root_link_pose_w."
            )

        pose = pose.to(device=self.device, dtype=torch.float32)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
        return pose[:, :7]

    def _build_action_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_mask = torch.zeros(self.act_dim, dtype=torch.bool, device=self.device)
        tendon_mask = torch.zeros(self.act_dim, dtype=torch.bool, device=self.device)
        idx = 0
        action_manager = self.env.action_manager
        for term in action_manager._terms.values():
            dim = term.action_dim
            transmission_type = getattr(term.cfg, "transmission_type", None)
            if transmission_type == TransmissionType.JOINT:
                joint_mask[idx : idx + dim] = True
            elif transmission_type == TransmissionType.TENDON:
                tendon_mask[idx : idx + dim] = True
            idx += dim
        return joint_mask, tendon_mask

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


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return ShadowPipetteManipulatePolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        server=server,
    )
