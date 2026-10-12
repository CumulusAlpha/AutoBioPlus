from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from assets.mjcf.robot.ur5e_wuji import WUJI_JOINT_ORDER
from src.utils.policy_helpers.wuji_heatingplate_rotate_common import (
    BEAKER_GRASP_JOINTS as BEAKER_GRASP_JOINTS,
    BEAKER_GRASP_REL_POSE as BEAKER_GRASP_REL_POSE,
    BEAKER_PREGRASP_REL_POSE as BEAKER_PREGRASP_REL_POSE,
    BEAKER_RELEASE_RETREAT_STEPS as BEAKER_RELEASE_RETREAT_STEPS,
    BEAKER_RELEASE_Y_OFFSET as BEAKER_RELEASE_Y_OFFSET,
    BEAKER_TARGET_POSE_HEATING_PLATE as BEAKER_TARGET_POSE_HEATING_PLATE,
    BUTTON_GRASP_JOINTS as BUTTON_GRASP_JOINTS,
    BUTTON_GRASP_REL_POSE as BUTTON_GRASP_REL_POSE,
    BUTTON_HOME_GRASP_AXIS_ROTATION as BUTTON_HOME_GRASP_AXIS_ROTATION,
    BUTTON_PREGRASP_JOINTS as BUTTON_PREGRASP_JOINTS,
    BUTTON_PREGRASP_REL_POSE as BUTTON_PREGRASP_REL_POSE,
    HEATING_PLATE_KNOB_AXIS as HEATING_PLATE_KNOB_AXIS,
    WujiHeatingPlateRotateState as WujiHeatingPlateRotateState,
)
from src.utils.policy_helpers.wuji_heatingplate_rotate_diagnostics import (
    HeatingPlateDiagnostics,
)
from src.utils.kinematics import batched_interpolate_linear, qpos_linear_interpolate
from src.utils.observations import observation_tensor
from src.utils.quat_util import (
    axis_angle_to_quat,
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_mul,
    quat_rotate_vec,
)


class WujiHeatingPlateRotatePolicy(HeatingPlateDiagnostics):
    """Wuji policy that places a beaker, turns the plate knob, then returns it."""

    def __init__(
        self,
        action_shape,
        device,
        env,
        pregrasp_beaker_steps: int = 30,
        grasp_beaker_steps: int = 30,
        grasp_beaker_hold_steps: int = 20,
        lift_steps: int = 25,
        move_to_plate_steps: int = 45,
        release_beaker_steps: int = 25,
        pregrasp_button_steps: int = 45,
        grasp_button_steps: int = 30,
        grasp_button_hold_steps: int = 40,
        rotate_button_steps: int = 50,
        rotate_button_home_steps: int = 50,
        button_hover_steps: int = 50,
        button_hover_height: float = 0.10,
        button_safe_clearance: float = 0.20,
        lift_height: float = 0.20,
        beaker_pregrasp_clearance: float = 0.08,
        beaker_pregrasp_y_offset: float = 0.04,
        button_turn_angle: float = 1.57,
        button_qpos_tol: float = 0.05,
        pos_tol: float = 0.03,
        rot_tol: float = 0.12,
        server=None,
    ) -> None:
        self.device = device
        self.env = env
        self.num_envs, self.act_dim = action_shape
        self.hand_dim = len(WUJI_JOINT_ORDER)
        self.expected_action_dim = 7 + self.hand_dim
        if self.act_dim != self.expected_action_dim:
            raise ValueError(
                f"Expected action_dim={self.expected_action_dim} "
                f"(7D palm IK + {self.hand_dim} Wuji joints), got {self.act_dim}."
            )

        self.pregrasp_beaker_steps = max(1, int(pregrasp_beaker_steps))
        self.grasp_beaker_steps = max(2, int(grasp_beaker_steps))
        self.grasp_beaker_hold_steps = max(0, int(grasp_beaker_hold_steps))
        self.lift_steps = max(1, int(lift_steps))
        self.move_to_plate_steps = max(1, int(move_to_plate_steps))
        self.release_beaker_steps = max(1, int(release_beaker_steps))
        self.pregrasp_button_steps = max(2, int(pregrasp_button_steps))
        self.grasp_button_steps = max(2, int(grasp_button_steps))
        self.grasp_button_hold_steps = max(0, int(grasp_button_hold_steps))
        self.rotate_button_steps = max(1, int(rotate_button_steps))
        self.rotate_button_home_steps = max(1, int(rotate_button_home_steps))
        self.button_hover_steps = max(1, int(button_hover_steps))
        self.button_hover_height = float(button_hover_height)
        self.button_safe_clearance = float(button_safe_clearance)
        self.lift_height = float(lift_height)
        self.beaker_pregrasp_clearance = float(beaker_pregrasp_clearance)
        self.beaker_pregrasp_y_offset = float(beaker_pregrasp_y_offset)
        self.button_turn_angle = float(button_turn_angle)
        self.button_qpos_tol = float(button_qpos_tol)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)

        recorded_beaker_grasp = self._pose_tensor(BEAKER_GRASP_REL_POSE)
        half_turn_about_beaker = self._pose_tensor((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        self.beaker_grasp_rel_pose = compose_pose_wxyz(
            half_turn_about_beaker, recorded_beaker_grasp
        )
        self.button_grasp_rel_pose = self._pose_tensor(BUTTON_GRASP_REL_POSE)
        self.beaker_target_pose_plate = self._pose_tensor(
            BEAKER_TARGET_POSE_HEATING_PLATE
        )
        self.beaker_pregrasp_rel_pose = self._pose_tensor(BEAKER_PREGRASP_REL_POSE)
        self.button_pregrasp_rel_pose = self._pose_tensor(BUTTON_PREGRASP_REL_POSE)
        self.beaker_grasp_hand = self._joint_targets(
            BEAKER_GRASP_JOINTS, "beaker pose log"
        )
        self.button_grasp_hand = self._joint_targets(
            BUTTON_GRASP_JOINTS, "heating-plate button pose log"
        )
        self.button_pregrasp_hand = self._joint_targets(
            BUTTON_PREGRASP_JOINTS, "heating-plate button pregrasp pose log"
        )
        self.open_hand = torch.zeros(
            self.hand_dim, device=self.device, dtype=torch.float32
        )
        self.knob_axis = F.normalize(
            torch.tensor(
                HEATING_PLATE_KNOB_AXIS,
                device=self.device,
                dtype=torch.float32,
            ),
            dim=0,
        )
        self.button_home_grasp_rel_pose = self._rotate_local_pose_about_knob_axis(
            self.button_grasp_rel_pose,
            BUTTON_HOME_GRASP_AXIS_ROTATION,
        )

        self.beaker = env.scene["beaker"]
        self.heating_plate = env.scene["heating_plate"]
        knob_body_ids, _ = self.heating_plate.find_bodies(["knob"], preserve_order=True)
        if len(knob_body_ids) != 1:
            raise RuntimeError("Could not uniquely resolve heating plate knob body.")
        self._knob_body_id = int(knob_body_ids[0])
        self._knob_geom_ids: set[int] = set()
        self._wuji_collision_geom_ids: set[int] = set()
        self._geom_id_to_name: dict[int, str] = {}
        self._geom_id_to_body_name: dict[int, str] = {}
        self._knob_global_body_id: int | None = None
        self._init_contact_force_debug()

        self.traj: torch.Tensor | None = None
        self.traj_len: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None
        self.state: torch.Tensor | None = None
        self.complete: torch.Tensor | None = None
        self.success_debug_text = None
        self._active_turn_angle = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._active_knob_target_qpos = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._set_env_policy_complete(False)
        if server is not None:
            with server.gui.add_folder("Wuji Heating Plate Success Debug"):
                self.success_debug_text = server.gui.add_text(
                    label="env 0 check_success",
                    initial_value="Waiting for first environment step...",
                    multiline=True,
                    disabled=True,
                )

        print(
            "Loaded Wuji heating-plate rotate policy: "
            f"13 states, button_turn_angle={self.button_turn_angle:.3f} rad"
        )

    def reset(self) -> None:
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.state = None
        self.complete = None
        self._active_turn_angle[:] = 0.0
        self._active_knob_target_qpos[:] = 0.0
        self._set_env_policy_complete(False)
        if self.success_debug_text is not None:
            self.success_debug_text.value = "RESET"

    def reset_envs(self, env_ids) -> None:
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
            self.state[env_ids] = WujiHeatingPlateRotateState.PREGRASP_BEAKER.value
        if self.complete is not None:
            self.complete[env_ids] = False
        self._active_turn_angle[env_ids] = 0.0
        self._active_knob_target_qpos[env_ids] = 0.0
        self._set_env_policy_complete(False, env_ids)
        if 0 in env_ids.detach().cpu().tolist() and self.success_debug_text is not None:
            self.success_debug_text.value = "RESET"

    def __call__(self, obs):
        if self.traj is None:
            self.init_buffers()

        need_plan = (self.traj_len == 0) & (~self.complete)
        if need_plan.any():
            self.plan_state_batch(obs, need_plan)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        action = self.traj[batch_idx, self.idx]
        done = self._update_progress(obs, action)
        if done.any():
            self.advance_state(done)
            self.plan_state_batch(obs, done)
            transition_idxs = torch.where(done)[0]
            next_idx = torch.minimum(
                torch.ones_like(transition_idxs),
                self.traj_len[transition_idxs] - 1,
            )
            next_state = self.state[transition_idxs]
            entering_rotation = (
                next_state
                == WujiHeatingPlateRotateState.ROTATE_BUTTON_TO_TARGET.value
            ) | (next_state == WujiHeatingPlateRotateState.ROTATE_BUTTON_HOME.value)
            next_idx[entering_rotation] = 0
            self.idx[transition_idxs] = next_idx
            action[transition_idxs] = self.traj[transition_idxs, next_idx]

        self._set_env_policy_complete(self.complete)
        self._update_success_debug_text()
        return action

    def init_buffers(self) -> None:
        max_len = max(
            self.pregrasp_beaker_steps,
            self.grasp_beaker_steps + self.grasp_beaker_hold_steps + 1,
            self.lift_steps,
            self.move_to_plate_steps,
            self.release_beaker_steps + 10,
            self.pregrasp_button_steps + 1,
            self.grasp_button_steps + self.grasp_button_hold_steps + 1,
            self.rotate_button_steps + 1,
            self.rotate_button_home_steps + 1,
            self.button_hover_steps + self.release_beaker_steps + 1,
        )
        self.traj = torch.zeros(
            (self.num_envs, max_len, self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.traj_len = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.state = torch.full(
            (self.num_envs,),
            WujiHeatingPlateRotateState.PREGRASP_BEAKER.value,
            device=self.device,
            dtype=torch.long,
        )
        self.complete = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def plan_state_batch(self, obs, plan_mask: torch.Tensor) -> None:
        planners = {
            WujiHeatingPlateRotateState.PREGRASP_BEAKER.value: self._plan_pregrasp_beaker,
            WujiHeatingPlateRotateState.GRASP_BEAKER.value: self._plan_grasp_beaker,
            WujiHeatingPlateRotateState.LIFT.value: self._plan_lift,
            WujiHeatingPlateRotateState.MOVE_TO_HEATING_PLATE.value: self._plan_move_to_heating_plate,
            WujiHeatingPlateRotateState.RELEASE_BEAKER.value: self._plan_release_beaker,
            WujiHeatingPlateRotateState.PREGRASP_BUTTON.value: self._plan_pregrasp_button,
            WujiHeatingPlateRotateState.GRASP_BUTTON.value: self._plan_grasp_button,
            WujiHeatingPlateRotateState.ROTATE_BUTTON_TO_TARGET.value: self._plan_rotate_button_to_target,
            WujiHeatingPlateRotateState.RELEASE_BUTTON_HOVER.value: self._plan_release_button_hover,
            WujiHeatingPlateRotateState.PREGRASP_BUTTON_HOME.value: self._plan_pregrasp_button_home,
            WujiHeatingPlateRotateState.REGRASP_BUTTON.value: self._plan_regrasp_button_home,
            WujiHeatingPlateRotateState.ROTATE_BUTTON_HOME.value: self._plan_rotate_button_home,
            WujiHeatingPlateRotateState.POST_ROTATE_LIFT.value: self._plan_post_rotate_lift,
        }
        for state_value, planner in planners.items():
            state_mask = plan_mask & (self.state == state_value)
            if state_mask.any():
                idxs = torch.where(state_mask)[0]
                self.apply_plan(idxs, planner(obs, idxs))

    def apply_plan(self, idxs: torch.Tensor, trajectory: torch.Tensor) -> None:
        length = trajectory.shape[1]
        self._ensure_capacity(length)
        self.traj[idxs] = 0.0
        self.traj[idxs, :length] = trajectory
        self.traj_len[idxs] = length
        self.idx[idxs] = 0

    def _ensure_capacity(self, length: int) -> None:
        if self.traj.shape[1] >= length:
            return
        old = self.traj
        self.traj = torch.zeros(
            (self.num_envs, max(length, old.shape[1] * 2), self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.traj[:, : old.shape[1]] = old

    def _update_progress(self, obs, target_action: torch.Tensor) -> torch.Tensor:
        last = self.idx == self.traj_len - 1
        rotate = (
            self.state == WujiHeatingPlateRotateState.ROTATE_BUTTON_TO_TARGET.value
        ) | (self.state == WujiHeatingPlateRotateState.ROTATE_BUTTON_HOME.value)
        rotate_to_target = (
            self.state == WujiHeatingPlateRotateState.ROTATE_BUTTON_TO_TARGET.value
        )
        rotate_home = (
            self.state == WujiHeatingPlateRotateState.ROTATE_BUTTON_HOME.value
        )
        post_rotate_lift = (
            self.state == WujiHeatingPlateRotateState.POST_ROTATE_LIFT.value
        )
        pose_close = self._target_pose_is_reached(obs, target_action[:, :7])
        advance = (pose_close | rotate) & (~last) & (~self.complete)
        self.idx = torch.minimum(self.idx + advance.long(), self.traj_len - 1)

        qpos_reached = self._button_qpos_target_is_reached()
        rotate_target_done = (
            rotate_to_target & (qpos_reached | last) & (self.traj_len > 0)
        )
        rotate_home_done = rotate_home & (qpos_reached | last) & (self.traj_len > 0)
        post_rotate_lift_done = (
            post_rotate_lift & last & pose_close & (self.traj_len > 0)
        )
        self.complete[post_rotate_lift_done] = True
        return (
            (last & pose_close & (~rotate) & (self.traj_len > 0))
            | rotate_target_done
            | rotate_home_done
        ) & (~post_rotate_lift_done)

    def advance_state(self, done: torch.Tensor) -> None:
        transitions = {
            WujiHeatingPlateRotateState.PREGRASP_BEAKER.value: WujiHeatingPlateRotateState.GRASP_BEAKER.value,
            WujiHeatingPlateRotateState.GRASP_BEAKER.value: WujiHeatingPlateRotateState.LIFT.value,
            WujiHeatingPlateRotateState.LIFT.value: WujiHeatingPlateRotateState.MOVE_TO_HEATING_PLATE.value,
            WujiHeatingPlateRotateState.MOVE_TO_HEATING_PLATE.value: WujiHeatingPlateRotateState.RELEASE_BEAKER.value,
            WujiHeatingPlateRotateState.RELEASE_BEAKER.value: WujiHeatingPlateRotateState.PREGRASP_BUTTON.value,
            WujiHeatingPlateRotateState.PREGRASP_BUTTON.value: WujiHeatingPlateRotateState.GRASP_BUTTON.value,
            WujiHeatingPlateRotateState.GRASP_BUTTON.value: WujiHeatingPlateRotateState.ROTATE_BUTTON_TO_TARGET.value,
            WujiHeatingPlateRotateState.ROTATE_BUTTON_TO_TARGET.value: WujiHeatingPlateRotateState.RELEASE_BUTTON_HOVER.value,
            WujiHeatingPlateRotateState.RELEASE_BUTTON_HOVER.value: WujiHeatingPlateRotateState.PREGRASP_BUTTON_HOME.value,
            WujiHeatingPlateRotateState.PREGRASP_BUTTON_HOME.value: WujiHeatingPlateRotateState.REGRASP_BUTTON.value,
            WujiHeatingPlateRotateState.REGRASP_BUTTON.value: WujiHeatingPlateRotateState.ROTATE_BUTTON_HOME.value,
            WujiHeatingPlateRotateState.ROTATE_BUTTON_HOME.value: WujiHeatingPlateRotateState.POST_ROTATE_LIFT.value,
        }
        old_state = self.state.clone()
        for source, target in transitions.items():
            mask = done & (old_state == source)
            self.state[mask] = target
        self.traj_len[done] = 0
        self.idx[done] = 0

    def _plan_pregrasp_beaker(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        target_pose = self._pose_from_entity(
            self._beaker_pose(), self.beaker_pregrasp_rel_pose, idxs
        )
        target_pose[:, 1] += self.beaker_pregrasp_y_offset
        start_hand = self._current_hand(obs)[idxs]
        open_hand = self._repeat_hand(self.open_hand, idxs)
        return self._interpolate(
            start_pose,
            target_pose,
            start_hand,
            open_hand,
            self.pregrasp_beaker_steps,
        )

    def _plan_grasp_beaker(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        grasp_pose = self._pose_from_entity(
            self._beaker_pose(), self.beaker_grasp_rel_pose, idxs
        )
        open_hand = self._repeat_hand(self.open_hand, idxs)
        grasp_hand = self._repeat_hand(self.beaker_grasp_hand, idxs)
        approach_steps = max(1, self.grasp_beaker_steps // 2)
        close_steps = max(1, self.grasp_beaker_steps - approach_steps)
        approach = self._interpolate(
            start_pose, grasp_pose, open_hand, open_hand, approach_steps
        )
        close = self._interpolate(
            grasp_pose, grasp_pose, open_hand, grasp_hand, close_steps
        )
        hold_action = torch.cat((grasp_pose, grasp_hand), dim=-1).unsqueeze(1)
        hold = hold_action.repeat(1, self.grasp_beaker_hold_steps, 1)
        return torch.cat((approach, close[:, 1:], hold), dim=1)

    def _plan_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        target_pose = start_pose.clone()
        target_pose[:, 2] += self.lift_height
        hand = self._repeat_hand(self.beaker_grasp_hand, idxs)
        return self._interpolate(start_pose, target_pose, hand, hand, self.lift_steps)

    def _plan_move_to_heating_plate(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        plate_pose = self._heating_plate_pose()
        target_beaker_pose = compose_pose_wxyz(
            plate_pose[idxs],
            self.beaker_target_pose_plate.unsqueeze(0).repeat(idxs.numel(), 1),
        )
        target_pose = compose_pose_wxyz(
            target_beaker_pose,
            self.beaker_grasp_rel_pose.unsqueeze(0).repeat(idxs.numel(), 1),
        )
        above_pose = target_pose.clone()
        above_pose[:, 2] = start_pose[:, 2]
        hand = self._repeat_hand(self.beaker_grasp_hand, idxs)
        horizontal_steps = max(2, self.move_to_plate_steps * 2 // 3)
        descend_steps = max(2, self.move_to_plate_steps - horizontal_steps + 1)
        horizontal = self._interpolate(
            start_pose, above_pose, hand, hand, horizontal_steps
        )
        descend = self._interpolate(above_pose, target_pose, hand, hand, descend_steps)
        return torch.cat((horizontal, descend[:, 1:]), dim=1)

    def _plan_release_beaker(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        pose = self._current_palm_pose(obs)[idxs]
        start_hand = self._current_hand(obs)[idxs]
        open_hand = self._repeat_hand(self.open_hand, idxs)
        release = self._interpolate(
            pose, pose, start_hand, open_hand, self.release_beaker_steps
        )
        retreat_pose = pose.clone()
        retreat_pose[:, 1] += BEAKER_RELEASE_Y_OFFSET
        retreat = self._interpolate(
            pose,
            retreat_pose,
            open_hand,
            open_hand,
            BEAKER_RELEASE_RETREAT_STEPS,
        )
        return torch.cat((release, retreat[:, 1:]), dim=1)

    def _plan_pregrasp_button(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_knob_pose = self._knob_pose()
        target_pose = self._pose_from_entity(
            current_knob_pose, self.button_pregrasp_rel_pose, idxs
        )
        return self._plan_elevated_button_pregrasp(obs, idxs, target_pose)

    def _plan_pregrasp_button_home(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        current_knob_pose = self._knob_pose()
        target_pose = self._pose_from_entity(
            current_knob_pose, self.button_pregrasp_rel_pose, idxs
        )
        start_hand = self._current_hand(obs)[idxs]
        pregrasp_hand = self._repeat_hand(self.button_pregrasp_hand, idxs)
        return self._interpolate(
            start_pose,
            target_pose,
            start_hand,
            pregrasp_hand,
            self.pregrasp_button_steps,
        )

    def _plan_elevated_button_pregrasp(
        self, obs, idxs: torch.Tensor, target_pose: torch.Tensor
    ) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        safe_height = (
            torch.maximum(start_pose[:, 2], target_pose[:, 2])
            + self.button_safe_clearance
        )
        lifted_pose = start_pose.clone()
        lifted_pose[:, 2] = safe_height
        above_target_pose = target_pose.clone()
        above_target_pose[:, 2] = safe_height
        open_hand = self._repeat_hand(self.open_hand, idxs)
        pregrasp_hand = self._repeat_hand(self.button_pregrasp_hand, idxs)
        lift_steps = max(2, self.pregrasp_button_steps // 3)
        traverse_steps = max(2, self.pregrasp_button_steps // 3)
        descend_steps = max(
            2, self.pregrasp_button_steps - lift_steps - traverse_steps + 2
        )
        lift = self._interpolate(
            start_pose, lifted_pose, open_hand, open_hand, lift_steps
        )
        traverse = self._interpolate(
            lifted_pose,
            above_target_pose,
            open_hand,
            pregrasp_hand,
            traverse_steps,
        )
        descend = self._interpolate(
            above_target_pose,
            target_pose,
            pregrasp_hand,
            pregrasp_hand,
            descend_steps,
        )
        return torch.cat((lift, traverse[:, 1:], descend[:, 1:]), dim=1)

    def _plan_grasp_button(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        grasp_pose = self._pose_from_entity(
            self._knob_pose(), self.button_grasp_rel_pose, idxs
        )
        pregrasp_hand = self._repeat_hand(self.button_pregrasp_hand, idxs)
        grasp_hand = self._repeat_hand(self.button_grasp_hand, idxs)
        approach_steps = max(1, self.grasp_button_steps // 2)
        close_steps = max(1, self.grasp_button_steps - approach_steps)
        approach = self._interpolate(
            start_pose, grasp_pose, pregrasp_hand, pregrasp_hand, approach_steps
        )
        close = self._interpolate(
            grasp_pose, grasp_pose, pregrasp_hand, grasp_hand, close_steps
        )
        hold = (
            torch.cat((grasp_pose, grasp_hand), dim=-1)
            .unsqueeze(1)
            .repeat(1, self.grasp_button_hold_steps, 1)
        )
        return torch.cat((approach, close[:, 1:], hold), dim=1)

    def _plan_regrasp_button_home(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        current_knob_pose = self._knob_pose()
        grasp_pose = self._pose_from_entity(
            current_knob_pose, self.button_home_grasp_rel_pose, idxs
        )
        pregrasp_hand = self._repeat_hand(self.button_pregrasp_hand, idxs)
        grasp_hand = self._repeat_hand(self.button_grasp_hand, idxs)
        approach_steps = max(1, self.grasp_button_steps // 2)
        close_steps = max(1, self.grasp_button_steps - approach_steps)
        approach = self._interpolate(
            start_pose,
            grasp_pose,
            pregrasp_hand,
            pregrasp_hand,
            approach_steps,
        )
        close = self._interpolate(
            grasp_pose, grasp_pose, pregrasp_hand, grasp_hand, close_steps
        )
        hold = (
            torch.cat((grasp_pose, grasp_hand), dim=-1)
            .unsqueeze(1)
            .repeat(1, self.grasp_button_hold_steps, 1)
        )
        return torch.cat((approach, close[:, 1:], hold), dim=1)

    def _plan_rotate_button_to_target(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_knob_qpos = self._current_knob_qpos()[idxs]
        target_qpos = torch.full_like(current_knob_qpos, self.button_turn_angle)
        angles = target_qpos - current_knob_qpos
        self._active_knob_target_qpos[idxs] = target_qpos
        self._active_turn_angle[idxs] = angles
        return self._plan_rotate_button_by_angles(
            idxs,
            angles,
            self._current_palm_pose(obs)[idxs],
            self.rotate_button_steps,
        )

    def _plan_release_button_hover(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        hover_pose = start_pose.clone()
        hover_pose[:, 2] += self.button_hover_height
        start_hand = self._current_hand(obs)[idxs]
        open_hand = self._repeat_hand(self.open_hand, idxs)
        release_steps = max(2, self.release_beaker_steps)
        release = self._interpolate(
            start_pose, hover_pose, start_hand, open_hand, release_steps
        )
        hold = (
            torch.cat((hover_pose, open_hand), dim=-1)
            .unsqueeze(1)
            .repeat(1, self.button_hover_steps, 1)
        )
        return torch.cat((release, hold), dim=1)

    def _plan_rotate_button_home(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_knob_qpos = self._current_knob_qpos()[idxs]
        angles = -current_knob_qpos
        self._active_knob_target_qpos[idxs] = 0.0
        self._active_turn_angle[idxs] = angles
        return self._plan_rotate_button_by_angles(
            idxs,
            angles,
            self._current_palm_pose(obs)[idxs],
            self.rotate_button_home_steps,
        )

    def _plan_post_rotate_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._current_palm_pose(obs)[idxs]
        target_pose = start_pose.clone()
        target_pose[:, 2] += self.button_hover_height
        start_hand = self._current_hand(obs)[idxs]
        open_hand = self._repeat_hand(self.open_hand, idxs)
        return self._interpolate(
            start_pose,
            target_pose,
            start_hand,
            open_hand,
            self.lift_steps,
        )

    def _plan_rotate_button_by_angles(
        self,
        idxs: torch.Tensor,
        target_angles: torch.Tensor,
        start_pose: torch.Tensor,
        rotate_steps: int,
    ) -> torch.Tensor:
        knob_pose = self._knob_pose()[idxs]
        fractions = torch.linspace(
            0.0,
            1.0,
            rotate_steps + 1,
            device=self.device,
            dtype=torch.float32,
        )
        angles = fractions.unsqueeze(0) * target_angles.view(-1, 1)
        plate_quat = self._heating_plate_pose()[idxs, 3:7]
        world_axis = quat_rotate_vec(
            plate_quat,
            self.knob_axis.unsqueeze(0).repeat(idxs.numel(), 1),
        )
        rotation = torch.stack(
            [
                axis_angle_to_quat(world_axis, angles[:, step], idxs.numel())
                for step in range(angles.shape[1])
            ],
            dim=1,
        )
        num_steps = angles.shape[1]
        knob_traj = knob_pose.unsqueeze(1).repeat(1, num_steps, 1)
        knob_quat = knob_pose[:, 3:7].unsqueeze(1).repeat(1, num_steps, 1)
        knob_traj[:, :, 3:7] = F.normalize(quat_mul(rotation, knob_quat), dim=-1)
        local_grasp = compose_pose_wxyz(pose_inv_wxyz(knob_pose), start_pose)
        local_grasp = local_grasp.unsqueeze(1).repeat(1, num_steps, 1)
        palm_traj = compose_pose_wxyz(knob_traj, local_grasp)
        hand = self._repeat_hand(self.button_grasp_hand, idxs)
        hand_traj = hand.unsqueeze(1).repeat(1, num_steps, 1)
        return torch.cat((palm_traj, hand_traj), dim=-1)

    def _interpolate(
        self,
        start_pose: torch.Tensor,
        target_pose: torch.Tensor,
        start_hand: torch.Tensor,
        target_hand: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        pos, quat = batched_interpolate_linear(
            start_pos=start_pose[:, :3],
            end_pos=target_pose[:, :3],
            start_quat=F.normalize(start_pose[:, 3:7], dim=-1),
            end_quat=F.normalize(target_pose[:, 3:7], dim=-1),
            num_steps=steps,
        )
        hand = qpos_linear_interpolate(start_hand, target_hand, steps)
        return torch.cat((pos, quat, hand), dim=-1)

    def _target_pose_is_reached(self, obs, target_pose: torch.Tensor) -> torch.Tensor:
        current_pose = self._current_palm_pose(obs)
        pos_err = torch.linalg.norm(current_pose[:, :3] - target_pose[:, :3], dim=-1)
        qa = F.normalize(current_pose[:, 3:7], dim=-1)
        qb = F.normalize(target_pose[:, 3:7], dim=-1)
        dot = torch.sum(qa * qb, dim=-1).abs().clamp(-1.0, 1.0)
        rot_err = 2.0 * torch.acos(dot)
        return (pos_err <= self.pos_tol) & (rot_err <= self.rot_tol)

    def _current_palm_pose(self, obs) -> torch.Tensor:
        pose = observation_tensor(obs, "palm_body_pose_w", self.device, self.num_envs)[
            :, :7
        ]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _current_hand(self, obs) -> torch.Tensor:
        qpos = observation_tensor(obs, "current_qpos", self.device, self.num_envs)
        return qpos[:, -self.hand_dim :]

    def _current_knob_qpos(self) -> torch.Tensor:
        try:
            qpos = self.heating_plate.data.joint_pos.to(
                device=self.device, dtype=torch.float32
            )
            if qpos.ndim == 1:
                return qpos[:1].repeat(self.num_envs)
            return qpos[:, 0]
        except Exception:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _button_qpos_target_is_reached(self) -> torch.Tensor:
        qpos = self._current_knob_qpos()
        target = self._active_knob_target_qpos
        close = torch.abs(qpos - target) <= self.button_qpos_tol
        passed_negative_target = (target < 0.0) & (
            qpos <= target + self.button_qpos_tol
        )
        passed_positive_target = (target > 0.0) & (
            qpos >= target - self.button_qpos_tol
        )
        home_target = torch.abs(target) <= self.button_qpos_tol
        home_reached = home_target & (torch.abs(qpos) <= self.button_qpos_tol)
        return close | passed_negative_target | passed_positive_target | home_reached

    def _beaker_pose(self) -> torch.Tensor:
        return self._entity_root_pose(self.beaker, "beaker")

    def _heating_plate_pose(self) -> torch.Tensor:
        return self._entity_root_pose(self.heating_plate, "heating_plate")

    def _knob_pose(self) -> torch.Tensor:
        pose = self.heating_plate.data.body_link_pose_w[:, self._knob_body_id, :7]
        pose = pose.to(device=self.device, dtype=torch.float32).clone()
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _entity_root_pose(self, entity, name: str) -> torch.Tensor:
        pose = getattr(entity.data, "root_link_pose_w", None)
        if pose is None:
            raise RuntimeError(f"scene[{name!r}].data.root_link_pose_w is unavailable.")
        pose = pose.to(device=self.device, dtype=torch.float32).clone()
        if pose.ndim == 1:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
        pose = pose[:, :7]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _pose_from_entity(
        self, entity_pose: torch.Tensor, local_pose: torch.Tensor, idxs: torch.Tensor
    ) -> torch.Tensor:
        return compose_pose_wxyz(
            entity_pose[idxs], local_pose.unsqueeze(0).repeat(idxs.numel(), 1)
        )

    def _repeat_hand(self, hand: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        return hand.unsqueeze(0).repeat(idxs.numel(), 1)

    def _pose_tensor(self, values) -> torch.Tensor:
        pose = torch.tensor(values, device=self.device, dtype=torch.float32)
        pose[3:7] = F.normalize(pose[3:7], dim=0)
        return pose

    def _rotate_local_pose_about_knob_axis(
        self, pose: torch.Tensor, angle: float
    ) -> torch.Tensor:
        axis = self.knob_axis
        quat = axis_angle_to_quat(
            axis.unsqueeze(0),
            torch.tensor([angle], device=self.device, dtype=torch.float32),
            1,
        )[0]
        rotated_pose = pose.clone()
        rotated_pose[3:7] = F.normalize(
            quat_mul(quat.unsqueeze(0), pose[3:7].unsqueeze(0))[0],
            dim=0,
        )
        return rotated_pose

    def _joint_targets(self, targets: dict[str, float], source: str) -> torch.Tensor:
        values = []
        for expression in WUJI_JOINT_ORDER:
            value = next(
                (
                    target
                    for name, target in targets.items()
                    if re.fullmatch(expression, name)
                    or name.endswith(expression.removeprefix(".*"))
                ),
                None,
            )
            if value is None:
                raise KeyError(f"Missing Wuji joint {expression!r} in {source}.")
            values.append(value)
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _set_env_policy_complete(
        self, value, env_ids: torch.Tensor | None = None
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        flag = getattr(self.env, "wuji_heating_plate_rotate_policy_complete", None)
        if flag is None or flag.shape[0] != self.num_envs:
            flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if isinstance(value, torch.Tensor):
            value = value.to(device=self.device, dtype=torch.bool)
            flag[env_ids] = value[env_ids] if value.shape[0] == self.num_envs else value
        else:
            flag[env_ids] = bool(value)
        self.env.wuji_heating_plate_rotate_policy_complete = flag


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return WujiHeatingPlateRotatePolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        server=server,
    )
