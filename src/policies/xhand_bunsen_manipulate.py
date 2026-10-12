from __future__ import annotations

import math
import re
from enum import Enum

import torch
import torch.nn.functional as F

from assets.mjcf.robot.ur5e_xhand import XHAND_JOINT_ORDER
from src.tasks.config.xhand_bunsen_manipulate.env_cfgs import (
    BUNSEN_FIRE_COLOR_JOINT,
    BUNSEN_FIRE_IGNITION_JOINT,
    BUNSEN_SUCCESS_SILVER_ANGLE_RAD,
    BUNSEN_SUCCESS_STABILITY_STEPS,
    BUNSEN_SUCCESS_YELLOW_ANGLE_RAD,
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

XHAND_TARGET_ACTION = {
    "right_hand_thumb_bend_joint": 1.305,
    "right_hand_thumb_rota_joint1": 0.444,
    "right_hand_thumb_rota_joint2": 0.825,
    "right_hand_index_bend_joint": 0.0,
    "right_hand_index_joint1": 1.467,
    "right_hand_index_joint2": 0.312,
    "right_hand_mid_joint1": 1.330,
    "right_hand_mid_joint2": 0.829,
    "right_hand_ring_joint1": 1.349,
    "right_hand_ring_joint2": 0.880,
    "right_hand_pinky_joint1": 1.408,
    "right_hand_pinky_joint2": 0.880,
}

BUNSEN_KNOB_CENTER_LOCAL = (0.033980539, 0.0, -0.003359146)
BUNSEN_KNOB_BODY_QUAT_LOCAL = (0.707105, 0.0, 0.707108, 0.0)
BUNSEN_KNOB_JOINT_AXIS_BODY = (0.0, 0.0, 1.0)
YELLOW_ROTATION_CONTACT_GAIN = 1.45
SILVER_KNOB_CENTER_LOCAL = (0.0, 0.0, 0.03604168)
SILVER_KNOB_BODY_QUAT_LOCAL = (1.0, 0.0, 0.0, 0.0)
SILVER_KNOB_JOINT_AXIS_BODY = (0.0, 0.0, 1.0)

YELLOW_PREGRASP_REL_POS = (0.081431, -0.08828, 0.158086)
YELLOW_PREGRASP_REL_QUAT_WXYZ = (
    -1.489750e-04,
    1.429813e-04,
    -6.924442e-01,
    7.214714e-01,
)
YELLOW_PREGRASP_JOINTS = {
    "right_hand_thumb_bend_joint": 1.41700000,
    "right_hand_thumb_rota_joint1": 0.09700000,
    "right_hand_thumb_rota_joint2": 0.00800000,
    "right_hand_index_bend_joint": 0.00000000,
    "right_hand_index_joint1": 1.46700000,
    "right_hand_index_joint2": 0.31200000,
    "right_hand_mid_joint1": 1.33000000,
    "right_hand_mid_joint2": 0.82900000,
    "right_hand_ring_joint1": 1.34900000,
    "right_hand_ring_joint2": 0.88000000,
    "right_hand_pinky_joint1": 1.40800000,
    "right_hand_pinky_joint2": 0.88000000,
}

YELLOW_GRASP_REL_POS = (0.077703, -0.027142, 0.122754)
YELLOW_GRASP_REL_QUAT_WXYZ = (-1.813143e-04, 9.880453e-05, -4.785004e-01, 8.780873e-01)
YELLOW_GRASP_JOINTS = {
    "right_hand_thumb_bend_joint": 1.32300000,
    "right_hand_thumb_rota_joint1": 0.15000000,
    "right_hand_thumb_rota_joint2": 1.45900000,
    "right_hand_index_bend_joint": 0.00000000,
    "right_hand_index_joint1": 1.29100000,
    "right_hand_index_joint2": 0.89900000,
    "right_hand_mid_joint1": 1.33000000,
    "right_hand_mid_joint2": 0.82900000,
    "right_hand_ring_joint1": 1.34900000,
    "right_hand_ring_joint2": 0.88000000,
    "right_hand_pinky_joint1": 1.40800000,
    "right_hand_pinky_joint2": 0.88000000,
}

SILVER_PREGRASP_REL_POS = (0.123153, -0.08914, 0.136722)
SILVER_PREGRASP_REL_QUAT_WXYZ = (0.275668, 0.264841, -0.639795, 0.666729)
SILVER_PREGRASP_JOINTS = {
    "right_hand_thumb_bend_joint": 1.30500000,
    "right_hand_thumb_rota_joint1": 0.44400000,
    "right_hand_thumb_rota_joint2": -0.17500000,
    "right_hand_index_bend_joint": 0.00000000,
    "right_hand_index_joint1": 1.19100000,
    "right_hand_index_joint2": 0.00000000,
    "right_hand_mid_joint1": 1.33000000,
    "right_hand_mid_joint2": 0.81000000,
    "right_hand_ring_joint1": 1.34900000,
    "right_hand_ring_joint2": 1.92000000,
    "right_hand_pinky_joint1": 1.40800000,
    "right_hand_pinky_joint2": 1.83900000,
}

SILVER_GRASP_REL_POS = (0.079525, -0.090952, 0.093055)
SILVER_GRASP_REL_QUAT_WXYZ = (0.275668, 0.264841, -0.639795, 0.666729)
SILVER_GRASP_JOINTS = {
    "right_hand_thumb_bend_joint": 1.30500000,
    "right_hand_thumb_rota_joint1": 0.44400000,
    "right_hand_thumb_rota_joint2": 0.82500000,
    "right_hand_index_bend_joint": 0.00000000,
    "right_hand_index_joint1": 1.46700000,
    "right_hand_index_joint2": 0.31200000,
    "right_hand_mid_joint1": 1.33000000,
    "right_hand_mid_joint2": 0.81000000,
    "right_hand_ring_joint1": 1.34900000,
    "right_hand_ring_joint2": 1.92000000,
    "right_hand_pinky_joint1": 1.40800000,
    "right_hand_pinky_joint2": 1.83900000,
}


class XHandBunsenManipulateState(Enum):
    PREGRASP = 0
    GRASP = 1
    ROTATE = 2
    RETREAT = 3
    PREGRASP_SILVER = 4
    GRASP_SILVER = 5
    ROTATE_SILVER = 6
    DONE = 7


class XHandBunsenManipulatePolicy:
    """Parallel UR5e+XHand state policy: pregrasp, grasp, then rotate the yellow knob."""

    def __init__(
        self,
        action_shape,
        device,
        env,
        pregrasp_steps: int = 80,
        grasp_steps: int = 50,
        rotate_steps: int = 90,
        retreat_hold_steps: int = 10,
        retreat_release_steps: int = 60,
        retreat_steps: int = 80,
        silver_pregrasp_steps: int = 80,
        silver_grasp_steps: int = 50,
        silver_rotate_steps: int = 90,
        turn_angle_rad: float = BUNSEN_SUCCESS_YELLOW_ANGLE_RAD,
        silver_turn_angle_rad: float = 0.75 * BUNSEN_SUCCESS_SILVER_ANGLE_RAD,
        pos_tol: float = 0.015,
        rot_tol: float = 0.08,
        silver_grasp_pos_tol: float = 0.03,
        silver_grasp_rot_tol: float = 0.12,
        rel_pose_file: str | None = None,
        server=None,
    ) -> None:
        self.server = server
        self.device = device
        self.env = env
        self.env.xhand_bunsen_require_policy_complete = True
        self.num_envs, self.act_dim = action_shape
        self.pose_dim = 7
        self.hand_dim = len(XHAND_JOINT_ORDER)
        self.expected_action_dim = self.pose_dim + self.hand_dim
        if self.act_dim != self.expected_action_dim:
            raise ValueError(
                f"Expected action_dim={self.expected_action_dim} "
                f"(7D UR5e IK target + {self.hand_dim} XHand qpos), got {self.act_dim}. "
                "Use a DifferentialIKActionCfg UR5e action before the XHand action."
            )

        self.pregrasp_steps = max(1, int(pregrasp_steps))
        self.grasp_steps = max(1, int(grasp_steps))
        self.rotate_steps = max(1, int(rotate_steps))
        self.retreat_hold_steps = max(0, int(retreat_hold_steps))
        self.retreat_release_steps = max(0, int(retreat_release_steps))
        self.retreat_steps = max(1, int(retreat_steps))
        self.silver_pregrasp_steps = max(1, int(silver_pregrasp_steps))
        self.silver_grasp_steps = max(1, int(silver_grasp_steps))
        self.silver_rotate_steps = max(1, int(silver_rotate_steps))
        self.turn_angle_rad = float(turn_angle_rad)
        self.silver_turn_angle_rad = float(silver_turn_angle_rad)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)
        self.silver_grasp_pos_tol = float(silver_grasp_pos_tol)
        self.silver_grasp_rot_tol = float(silver_grasp_rot_tol)

        if rel_pose_file is not None:
            print(
                "Ignoring rel_pose_file; XHand bunsen grasp poses are hardcoded in the policy."
            )

        self.pregrasp_local_pose = self._make_pose_tensor(
            YELLOW_PREGRASP_REL_POS, YELLOW_PREGRASP_REL_QUAT_WXYZ
        )
        self.grasp_local_pose = self._make_pose_tensor(
            YELLOW_GRASP_REL_POS, YELLOW_GRASP_REL_QUAT_WXYZ
        )
        self.silver_pregrasp_local_pose = self._make_pose_tensor(
            SILVER_PREGRASP_REL_POS,
            SILVER_PREGRASP_REL_QUAT_WXYZ,
        )
        self.silver_grasp_local_pose = self._make_pose_tensor(
            SILVER_GRASP_REL_POS, SILVER_GRASP_REL_QUAT_WXYZ
        )
        self.pregrasp_hand_action = self._build_action_from_joint_targets(
            YELLOW_PREGRASP_JOINTS,
            "hardcoded yellow pregrasp pose",
        )
        self.grasp_hand_action = self._build_target_action(YELLOW_GRASP_JOINTS)
        self.zero_hand_action = torch.zeros(
            self.hand_dim, device=self.device, dtype=torch.float32
        )
        self.silver_pregrasp_hand_action = self._build_action_from_joint_targets(
            SILVER_PREGRASP_JOINTS,
            "hardcoded silver pregrasp pose",
        )
        self.silver_grasp_hand_action = self._build_action_from_joint_targets(
            SILVER_GRASP_JOINTS,
            "hardcoded silver grasp pose",
        )

        self.robot_handle = env.scene["robot"]
        self.traj: torch.Tensor | None = None
        self.traj_len: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None
        self.state: torch.Tensor | None = None
        self.complete: torch.Tensor | None = None
        self.success_debug_text = None
        self.yellow_knob_joint_id = self._resolve_bunsen_joint_id(
            BUNSEN_FIRE_IGNITION_JOINT
        )
        self.silver_knob_joint_id = self._resolve_bunsen_joint_id(
            BUNSEN_FIRE_COLOR_JOINT
        )
        if self.server is not None:
            with self.server.gui.add_folder("XHand Bunsen Success Debug"):
                self.success_debug_text = self.server.gui.add_text(
                    label="env 0 success check",
                    initial_value=(
                        "Waiting for first environment step...\n"
                        "Success = yellow_open_now AND silver_blue_now\n"
                        f"Must hold for {BUNSEN_SUCCESS_STABILITY_STEPS} consecutive steps."
                    ),
                    multiline=True,
                    disabled=True,
                )
        print(
            "Loaded XHand bunsen rotate policy: "
            f"pregrasp_steps={self.pregrasp_steps}, grasp_steps={self.grasp_steps}, "
            f"rotate_steps={self.rotate_steps}, turn_angle_rad={self.turn_angle_rad:.6f}, "
            f"retreat_hold_steps={self.retreat_hold_steps}, "
            f"retreat_release_steps={self.retreat_release_steps}, retreat_steps={self.retreat_steps}, "
            f"silver_pregrasp_steps={self.silver_pregrasp_steps}, silver_grasp_steps={self.silver_grasp_steps}, "
            f"silver_rotate_steps={self.silver_rotate_steps}, silver_turn_angle_rad={self.silver_turn_angle_rad:.6f}, "
            f"pos_tol={self.pos_tol}, rot_tol={self.rot_tol}, "
            f"silver_grasp_pos_tol={self.silver_grasp_pos_tol}, "
            f"silver_grasp_rot_tol={self.silver_grasp_rot_tol}, action_dim={self.act_dim}"
        )
        print("Using hardcoded XHand bunsen pregrasp/grasp poses.")

    def reset(self) -> None:
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.state = None
        self.complete = None
        self._set_env_policy_complete(False)
        if self.success_debug_text is not None:
            self.success_debug_text.value = (
                "RESET\n"
                "Success = yellow_open_now AND silver_blue_now\n"
                f"Must hold for {BUNSEN_SUCCESS_STABILITY_STEPS} consecutive steps."
            )

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
            self.state[env_ids] = XHandBunsenManipulateState.PREGRASP.value
        if self.complete is not None:
            self.complete[env_ids] = False
        self._set_env_policy_complete(False, env_ids)
        if 0 in env_ids.detach().cpu().tolist() and self.success_debug_text is not None:
            self.success_debug_text.value = (
                "RESET\n"
                "Success = yellow_open_now AND silver_blue_now\n"
                f"Must hold for {BUNSEN_SUCCESS_STABILITY_STEPS} consecutive steps."
            )

    def __call__(self, obs):
        if self.traj is None:
            self.init_buffers()

        need_plan = self.traj_len == 0
        if need_plan.any():
            self.plan_state_batch(obs, need_plan)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        action = self.traj[batch_idx, self.idx]
        done_mask = self.update_traj_progress(obs, action)
        if done_mask.any():
            self.advance_state(done_mask)
            self.plan_state_batch(obs, done_mask)

        self._set_env_policy_complete(self.complete)
        self._update_success_debug_text()
        return action

    def init_buffers(self) -> None:
        max_len = (
            max(
                self.pregrasp_steps,
                self.grasp_steps,
                self.rotate_steps,
                self.retreat_steps
                + self.retreat_hold_steps
                + self.retreat_release_steps,
                self.silver_pregrasp_steps,
                self.silver_grasp_steps,
                self.silver_rotate_steps,
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
            XHandBunsenManipulateState.PREGRASP.value,
            device=self.device,
            dtype=torch.long,
        )
        self.complete = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._set_env_policy_complete(self.complete)

    def plan_state_batch(self, obs, plan_mask: torch.Tensor) -> None:
        state_table = {
            XHandBunsenManipulateState.PREGRASP.value: self._plan_pregrasp,
            XHandBunsenManipulateState.GRASP.value: self._plan_grasp,
            XHandBunsenManipulateState.ROTATE.value: self._plan_rotate,
            XHandBunsenManipulateState.RETREAT.value: self._plan_retreat,
            XHandBunsenManipulateState.PREGRASP_SILVER.value: self._plan_pregrasp_silver,
            XHandBunsenManipulateState.GRASP_SILVER.value: self._plan_grasp_silver,
            XHandBunsenManipulateState.ROTATE_SILVER.value: self._plan_rotate_silver,
            XHandBunsenManipulateState.DONE.value: self._plan_done,
        }
        for state_value, plan_fn in state_table.items():
            state_mask = plan_mask & (self.state == state_value)
            if state_mask.any():
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
        pose_close = self._target_pose_is_reached(obs, target_action[:, :7])
        silver_grasp_state = (
            self.state == XHandBunsenManipulateState.GRASP_SILVER.value
        )
        if silver_grasp_state.any():
            relaxed_pose_close = self._target_pose_is_reached(
                obs,
                target_action[:, :7],
                pos_tol=self.silver_grasp_pos_tol,
                rot_tol=self.silver_grasp_rot_tol,
            )
            pose_close = torch.where(silver_grasp_state, relaxed_pose_close, pose_close)
        yellow_rotate_state = self.state == XHandBunsenManipulateState.ROTATE.value
        rotate_state = yellow_rotate_state | (
            self.state == XHandBunsenManipulateState.ROTATE_SILVER.value
        )
        yellow_angle = self.env.scene["bunsen"].data.joint_pos[
            :, self.yellow_knob_joint_id
        ]
        yellow_target_reached = yellow_rotate_state & (
            yellow_angle >= BUNSEN_SUCCESS_YELLOW_ANGLE_RAD
        )
        stage_complete = last | yellow_target_reached
        done_state = self.state == XHandBunsenManipulateState.DONE.value
        complete_done = done_state & last & (self.traj_len > 0)
        if complete_done.any():
            self.complete[complete_done] = True
        advance = (pose_close | rotate_state) & (~stage_complete)
        self.idx = torch.minimum(self.idx + advance.long(), self.traj_len - 1)
        return (
            stage_complete
            & (pose_close | rotate_state)
            & (self.traj_len > 0)
            & (~done_state)
        )

    def _set_env_policy_complete(
        self, value, env_ids: torch.Tensor | None = None
    ) -> None:
        if self.env is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        flag = getattr(self.env, "xhand_bunsen_manipulate_policy_complete", None)
        if flag is None or flag.shape[0] != self.num_envs:
            flag = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        if isinstance(value, bool):
            flag[env_ids] = value
        else:
            value = torch.as_tensor(value, device=self.device, dtype=torch.bool)
            if value.numel() == self.num_envs:
                flag[env_ids] = value[env_ids]
            elif value.numel() == env_ids.numel():
                flag[env_ids] = value
            else:
                raise ValueError(
                    f"Cannot set xhand_bunsen_manipulate_policy_complete for {env_ids.numel()} envs from value shape {tuple(value.shape)}."
                )
        self.env.xhand_bunsen_manipulate_policy_complete = flag

    def advance_state(self, done_mask: torch.Tensor) -> None:
        new_state = self.state.clone()
        pregrasp_done = done_mask & (
            self.state == XHandBunsenManipulateState.PREGRASP.value
        )
        grasp_done = done_mask & (
            self.state == XHandBunsenManipulateState.GRASP.value
        )
        rotate_done = done_mask & (
            self.state == XHandBunsenManipulateState.ROTATE.value
        )
        retreat_done = done_mask & (
            self.state == XHandBunsenManipulateState.RETREAT.value
        )
        silver_pregrasp_done = done_mask & (
            self.state == XHandBunsenManipulateState.PREGRASP_SILVER.value
        )
        silver_grasp_done = done_mask & (
            self.state == XHandBunsenManipulateState.GRASP_SILVER.value
        )
        silver_rotate_done = done_mask & (
            self.state == XHandBunsenManipulateState.ROTATE_SILVER.value
        )
        new_state[pregrasp_done] = XHandBunsenManipulateState.GRASP.value
        new_state[grasp_done] = XHandBunsenManipulateState.ROTATE.value
        new_state[rotate_done] = XHandBunsenManipulateState.RETREAT.value
        new_state[retreat_done] = XHandBunsenManipulateState.PREGRASP_SILVER.value
        new_state[silver_pregrasp_done] = (
            XHandBunsenManipulateState.GRASP_SILVER.value
        )
        new_state[silver_grasp_done] = (
            XHandBunsenManipulateState.ROTATE_SILVER.value
        )
        new_state[silver_rotate_done] = XHandBunsenManipulateState.DONE.value
        self.state = new_state
        self.traj_len[done_mask] = 0
        self.idx[done_mask] = 0

    def _plan_pregrasp(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = self._pose_from_bunsen_local(self.pregrasp_local_pose, idxs)
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = self.pregrasp_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            target_hand_action,
            self.pregrasp_steps,
        )

    def _plan_grasp(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = self._pose_from_bunsen_local(self.grasp_local_pose, idxs)
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            target_hand_action,
            self.grasp_steps,
        )

    def _plan_rotate(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        bunsen_pose_w = self._get_bunsen_pose_w()[idxs]
        start_local_pose = compose_pose_wxyz(pose_inv_wxyz(bunsen_pose_w), start_pose_w)
        current_angle = self.env.scene["bunsen"].data.joint_pos[
            idxs, self.yellow_knob_joint_id
        ]
        turn_angle = (
            BUNSEN_SUCCESS_YELLOW_ANGLE_RAD - current_angle
        ) * YELLOW_ROTATION_CONTACT_GAIN
        pose_traj_w = self._build_rotate_pose_traj_w(
            bunsen_pose_w,
            start_local_pose,
            turn_angle_rad=turn_angle,
        )[:, 6:]
        hand_action = self.grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        hand_traj = hand_action.unsqueeze(1).repeat(1, pose_traj_w.shape[1], 1)
        return torch.cat((pose_traj_w, hand_traj), dim=-1)

    def _plan_retreat(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        silver_pregrasp_pose_w = self._pose_from_bunsen_local(
            self.silver_pregrasp_local_pose, idxs
        )
        return self._plan_retreat_like(obs, idxs, target_z=silver_pregrasp_pose_w[:, 2])

    def _plan_retreat_like(
        self, obs, idxs: torch.Tensor, target_z: torch.Tensor | None = None
    ) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = start_pose_w.clone()
        target_pose_w[:, 0] -= 0.05
        if target_z is None:
            target_pose_w[:, 2] += 0.2
        else:
            target_pose_w[:, 2] = target_z.to(
                device=self.device, dtype=target_pose_w.dtype
            )
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        zero_hand_action = self.zero_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)

        move_traj = self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            zero_hand_action,
            zero_hand_action,
            self.retreat_steps,
        )
        if self.retreat_release_steps <= 0:
            return move_traj

        if self.retreat_hold_steps > 0:
            hold_action = torch.cat(
                (start_pose_w, start_hand_action), dim=-1
            ).unsqueeze(1)
            hold_traj = hold_action.repeat(1, self.retreat_hold_steps, 1)
        else:
            hold_traj = None

        open_hand_traj = self.plan_action_traj(
            start_pose_w,
            start_pose_w,
            start_hand_action,
            zero_hand_action,
            self.retreat_release_steps,
        )
        parts = [open_hand_traj, move_traj[:, 1:]]
        if hold_traj is not None:
            parts.insert(0, hold_traj)
        return torch.cat(parts, dim=1)

    def _plan_pregrasp_silver(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = self._pose_from_bunsen_local(
            self.silver_pregrasp_local_pose, idxs
        )
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = self.silver_pregrasp_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            target_hand_action,
            self.silver_pregrasp_steps,
        )

    def _plan_grasp_silver(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        target_pose_w = self._pose_from_bunsen_local(self.silver_grasp_local_pose, idxs)
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = self.silver_grasp_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            target_hand_action,
            self.silver_grasp_steps,
        )

    def _plan_rotate_silver(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_current_control_pose_w(obs)[idxs]
        bunsen_pose_w = self._get_bunsen_pose_w()[idxs]
        start_local_pose = compose_pose_wxyz(pose_inv_wxyz(bunsen_pose_w), start_pose_w)
        pose_traj_w = self._build_rotate_pose_traj_w(
            bunsen_pose_w,
            start_local_pose,
            center_local=SILVER_KNOB_CENTER_LOCAL,
            body_quat_local=SILVER_KNOB_BODY_QUAT_LOCAL,
            joint_axis_body=SILVER_KNOB_JOINT_AXIS_BODY,
            turn_angle_rad=self.silver_turn_angle_rad,
            rotate_steps=self.silver_rotate_steps,
        )
        hand_action = self.silver_grasp_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        hand_traj = hand_action.unsqueeze(1).repeat(1, pose_traj_w.shape[1], 1)
        return torch.cat((pose_traj_w, hand_traj), dim=-1)

    def _plan_done(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        return self._plan_retreat_like(obs, idxs)

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
            start_quat=F.normalize(start_pose_w[:, 3:7], dim=-1),
            end_quat=F.normalize(target_pose_w[:, 3:7], dim=-1),
            num_steps=num_steps,
        )
        hand_traj = qpos_linear_interpolate(
            start_hand_action, target_hand_action, num_steps
        )
        return torch.cat((pos_traj, quat_traj, hand_traj), dim=-1)

    def _build_rotate_pose_traj_w(
        self,
        bunsen_pose_w: torch.Tensor,
        start_local_pose: torch.Tensor,
        center_local: tuple[float, float, float] = BUNSEN_KNOB_CENTER_LOCAL,
        body_quat_local: tuple[
            float, float, float, float
        ] = BUNSEN_KNOB_BODY_QUAT_LOCAL,
        joint_axis_body: tuple[float, float, float] = BUNSEN_KNOB_JOINT_AXIS_BODY,
        turn_angle_rad: float | torch.Tensor | None = None,
        rotate_steps: int | None = None,
    ) -> torch.Tensor:
        batch = start_local_pose.shape[0]
        dtype = start_local_pose.dtype
        turn_angle_rad = (
            self.turn_angle_rad if turn_angle_rad is None else turn_angle_rad
        )
        rotate_steps = self.rotate_steps if rotate_steps is None else int(rotate_steps)
        center = torch.tensor(center_local, device=self.device, dtype=dtype)
        knob_body_quat = torch.tensor(body_quat_local, device=self.device, dtype=dtype)
        knob_axis_body = torch.tensor(joint_axis_body, device=self.device, dtype=dtype)
        axis = quat_rotate_vec(
            F.normalize(knob_body_quat, dim=0).unsqueeze(0),
            knob_axis_body.unsqueeze(0),
        )[0]
        axis = F.normalize(axis, dim=0)

        progress = torch.linspace(
            0.0,
            1.0,
            rotate_steps + 1,
            device=self.device,
            dtype=dtype,
        )
        end_angle = torch.as_tensor(turn_angle_rad, device=self.device, dtype=dtype)
        if end_angle.ndim == 0:
            end_angle = end_angle.repeat(batch)
        else:
            end_angle = end_angle.reshape(batch)
        angles = end_angle.unsqueeze(1) * progress.unsqueeze(0)
        axes = axis.reshape(1, 1, 3).expand(batch, rotate_steps + 1, 3)
        rot_quat = axis_angle_to_quat(
            axes.reshape(-1, 3),
            angles.reshape(-1),
            batch * (rotate_steps + 1),
        ).reshape(batch, rotate_steps + 1, 4)

        offset = start_local_pose[:, :3] - center.unsqueeze(0)
        turn_pos = center.unsqueeze(0).unsqueeze(1) + quat_rotate_vec(
            rot_quat.reshape(-1, 4),
            offset.unsqueeze(1).repeat(1, rotate_steps + 1, 1).reshape(-1, 3),
        ).reshape(batch, rotate_steps + 1, 3)
        turn_quat = F.normalize(
            quat_mul(
                rot_quat,
                start_local_pose[:, 3:7].unsqueeze(1).repeat(1, rotate_steps + 1, 1),
            ),
            dim=-1,
        )
        local_traj = torch.cat((turn_pos, turn_quat), dim=-1)
        return compose_pose_wxyz(
            bunsen_pose_w.unsqueeze(1).repeat(1, rotate_steps + 1, 1),
            local_traj,
        )

    def _target_pose_is_reached(
        self,
        obs,
        target_pose_w: torch.Tensor,
        pos_tol: float | None = None,
        rot_tol: float | None = None,
    ) -> torch.Tensor:
        pos_tol = self.pos_tol if pos_tol is None else float(pos_tol)
        rot_tol = self.rot_tol if rot_tol is None else float(rot_tol)
        current_pose_w = self._get_current_control_pose_w(obs)
        pos_err = torch.linalg.norm(
            current_pose_w[:, :3] - target_pose_w[:, :3], dim=-1
        )
        qa = F.normalize(current_pose_w[:, 3:7], dim=-1)
        qb = F.normalize(target_pose_w[:, 3:7], dim=-1)
        dot = torch.sum(qa * qb, dim=-1).abs().clamp(-1.0, 1.0)
        rot_err = 2.0 * torch.acos(dot)
        return (pos_err <= pos_tol) & (rot_err <= rot_tol)

    def _pose_from_bunsen_local(
        self, local_pose: torch.Tensor, idxs: torch.Tensor
    ) -> torch.Tensor:
        bunsen_pose_w = self._get_bunsen_pose_w()[idxs]
        local = local_pose.unsqueeze(0).repeat(idxs.numel(), 1)
        return compose_pose_wxyz(bunsen_pose_w, local)

    def _make_pose_tensor(
        self, pos: tuple[float, float, float], quat: tuple[float, float, float, float]
    ) -> torch.Tensor:
        pose = torch.tensor([*pos, *quat], device=self.device, dtype=torch.float32)
        pose[3:7] = F.normalize(pose[3:7], dim=0)
        return pose

    def _build_target_action(
        self, joint_target_overrides: dict[str, float] | None = None
    ) -> torch.Tensor:
        if joint_target_overrides:
            return self._build_action_from_joint_targets(
                joint_target_overrides, "hardcoded yellow grasp pose"
            )
        return self._build_action_from_joint_targets(
            XHAND_TARGET_ACTION, "floating XHand config"
        )

    def _build_action_from_joint_targets(
        self,
        joint_target_by_name: dict[str, float],
        source_name: str,
    ) -> torch.Tensor:
        target = torch.tensor(
            [
                self._target_for_joint_name(name, joint_target_by_name, source_name)
                for name in XHAND_JOINT_ORDER
            ],
            device=self.device,
            dtype=torch.float32,
        )
        if target.numel() != self.hand_dim:
            raise ValueError(
                f"Expected {self.hand_dim} XHand targets, got {target.numel()}."
            )
        return target

    def _target_for_joint_name(
        self,
        joint_name: str,
        joint_target_by_name: dict[str, float],
        source_name: str,
    ) -> float:
        if joint_name in joint_target_by_name:
            return joint_target_by_name[joint_name]
        pattern = re.compile(joint_name)
        for target_name, target_value in joint_target_by_name.items():
            if pattern.fullmatch(target_name):
                return target_value
        suffix = joint_name.removeprefix(".*")
        for target_name, target_value in joint_target_by_name.items():
            if target_name.endswith(suffix):
                return target_value
        raise KeyError(
            f"No XHand target action configured for joint pattern {joint_name!r} in {source_name}."
        )

    def _extract_current_hand_action(self, obs) -> torch.Tensor:
        qpos = observation_tensor(obs, "current_qpos", self.device, self.num_envs)
        return qpos[:, -self.hand_dim :]

    def _get_current_control_pose_w(self, obs) -> torch.Tensor:
        pose = observation_tensor(obs, "palm_body_pose_w", self.device, self.num_envs)[
            :, :7
        ]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _get_bunsen_pose_w(self) -> torch.Tensor:
        bunsen = self.env.scene["bunsen"]
        pose = getattr(bunsen.data, "root_link_pose_w", None)
        if pose is None:
            raise RuntimeError("scene['bunsen'].data.root_link_pose_w is unavailable.")
        pose = pose.to(device=self.device, dtype=torch.float32)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
        pose = pose[:, :7]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _resolve_bunsen_joint_id(self, joint_name: str) -> int:
        bunsen = self.env.scene["bunsen"]
        joint_ids, resolved_names = bunsen.find_joints([joint_name])
        ids = joint_ids.tolist() if hasattr(joint_ids, "tolist") else list(joint_ids)
        names = list(resolved_names) if resolved_names is not None else []
        for joint_id, resolved_name in zip(ids, names, strict=False):
            if resolved_name == joint_name or str(resolved_name).endswith(joint_name):
                return int(joint_id)
        if len(ids) == 1:
            return int(ids[0])
        raise RuntimeError(f"Could not resolve Bunsen joint {joint_name!r}.")

    def _update_success_debug_text(self) -> None:
        if self.success_debug_text is None:
            return

        joint_pos = self.env.scene["bunsen"].data.joint_pos
        yellow_angle = float(joint_pos[0, self.yellow_knob_joint_id].item())
        silver_angle = float(joint_pos[0, self.silver_knob_joint_id].item())
        yellow_seen = self._env_flag("xhand_bunsen_yellow_open_seen")
        silver_seen = self._env_flag("xhand_bunsen_silver_blue_seen")
        success_seen = self._env_flag("xhand_bunsen_success_seen")
        yellow_open = yellow_angle >= BUNSEN_SUCCESS_YELLOW_ANGLE_RAD
        silver_blue = silver_angle <= BUNSEN_SUCCESS_SILVER_ANGLE_RAD
        object_complete = yellow_open and silver_blue
        policy_complete = self._env_flag("xhand_bunsen_manipulate_policy_complete")
        stable_steps = getattr(self.env, "xhand_bunsen_stable_steps", None)
        stable_count = int(stable_steps[0].item()) if stable_steps is not None else 0
        success = (
            object_complete
            and policy_complete
            and stable_count >= BUNSEN_SUCCESS_STABILITY_STEPS
        )

        def result(value: bool) -> str:
            return "PASS" if value else "WAIT"

        self.success_debug_text.value = (
            f"yellow_open_now: {result(yellow_open)} | "
            f"{yellow_angle:+.4f} >= {BUNSEN_SUCCESS_YELLOW_ANGLE_RAD:+.4f} rad "
            f"({math.degrees(yellow_angle):+.1f} >= "
            f"{math.degrees(BUNSEN_SUCCESS_YELLOW_ANGLE_RAD):+.1f} deg)\n"
            f"yellow_open_seen: {result(yellow_seen)} | latched for this episode\n"
            f"silver_blue_now: {result(silver_blue)} | "
            f"{silver_angle:+.4f} <= {BUNSEN_SUCCESS_SILVER_ANGLE_RAD:+.4f} rad "
            f"({math.degrees(silver_angle):+.1f} <= "
            f"{math.degrees(BUNSEN_SUCCESS_SILVER_ANGLE_RAD):+.1f} deg)\n"
            f"silver_blue_seen: {result(silver_seen)} | latched for this episode\n"
            f"current_valve_state: {result(object_complete)} | yellow_open_now AND silver_blue_now\n"
            f"policy_complete: {result(policy_complete)} | full motion and retreat finished\n"
            f"stable_steps: {stable_count}/{BUNSEN_SUCCESS_STABILITY_STEPS}\n"
            f"full_success: {result(success)} | current state held consecutively\n"
            f"success_seen: {result(success_seen)} | latched full success"
        )

    def _env_flag(self, name: str) -> bool:
        value = getattr(self.env, name, None)
        return value is not None and bool(value[0].item())


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return XHandBunsenManipulatePolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        pregrasp_steps=80,
        grasp_steps=50,
        rotate_steps=90,
        server=server,
    )
