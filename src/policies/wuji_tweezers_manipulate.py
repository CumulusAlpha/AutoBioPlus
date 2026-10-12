from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from assets.mjcf.robot import WUJI_JOINT_ORDER
from src.utils.policy_helpers.wuji_tweezers_manipulate_common import (
    BUILTIN_DEXONOMY_GRASP_QPOS as BUILTIN_DEXONOMY_GRASP_QPOS,
    BUILTIN_DEXONOMY_OBJ_POSE as BUILTIN_DEXONOMY_OBJ_POSE,
    BUILTIN_DEXONOMY_PREGRASP_QPOS as BUILTIN_DEXONOMY_PREGRASP_QPOS,
    DEXONOMY_INTERP_STEPS as DEXONOMY_INTERP_STEPS,
    DEXONOMY_PHASE_KEYS as DEXONOMY_PHASE_KEYS,
    WUJI_SQUEEZE_QPOS as WUJI_SQUEEZE_QPOS,
    WujiTweezersManipulateState as WujiTweezersManipulateState,
    _available_contact_count as _available_contact_count,
    _builtin_dexonomy_payload as _builtin_dexonomy_payload,
    _load_dexonomy_payload as _load_dexonomy_payload,
    _load_tweezers_rel_pose_from_file as _load_tweezers_rel_pose_from_file,
    _load_wuji_joint_qpos_from_file as _load_wuji_joint_qpos_from_file,
    _parse_pose_values as _parse_pose_values,
)
from src.utils.policy_helpers.wuji_tweezers_manipulate_diagnostics import (
    WujiTweezersDiagnostics,
)
from src.utils.policy_helpers.wuji_tweezers_manipulate_geometry import (
    WujiTweezersGeometry,
)
from src.utils.kinematics import batched_interpolate_linear, qpos_linear_interpolate
from src.utils.observations import observation_tensor
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_distance,
)


class WujiTweezersManipulatePolicy(WujiTweezersDiagnostics, WujiTweezersGeometry):
    """UR5e+Wuji tweezer grasp-and-lift policy driven by palm IK targets."""

    def __init__(
        self,
        action_shape,
        device,
        env,
        target_pose_path: str | Path | None = None,
        pregrasp_steps: int = 40,
        grasp_steps: int = 10,
        grasp_hold_steps: int = 20,
        lift_steps: int = 60,
        adjust_steps: int = 20,
        align_xy_steps: int = 60,
        squeeze_steps: int = 20,
        transport_steps: int = 60,
        transport_hold_steps: int = 20,
        release_steps: int = 20,
        pregrasp_z_offset: float = 0.03,
        # lift_height: float = 0.085,
        lift_height: float = 0.04,
        post_align_lift_height: float = 0.12,
        align_xy_x_offset: float = -0.03,
        align_xy_y_offset: float = 0.005,
        pos_tol: float = 0.02,
        rot_tol: float = 0.05,
        pregrasp_pos_tol: float = 0.04,
        pregrasp_rot_tol: float = 0.03,
        dexonomy_max_pos_step: float = 0.004,
        dexonomy_max_rot_step: float = 0.035,
        dexonomy_max_hand_step: float = 0.035,
        dexonomy_min_interp_steps: int = DEXONOMY_INTERP_STEPS,
        loop: bool = False,
        tweezers_site_name: str = "tweezers/tweezers_site",
        tube_site_name: str = "",
        beaker_center_site_name: str = "beaker/beaker_center",
        server=None,
    ) -> None:
        self.device = device
        self.env = env
        self.num_envs, self.act_dim = action_shape
        self.pose_dim = 7
        self.hand_dim = len(WUJI_JOINT_ORDER)
        self.expected_action_dim = self.pose_dim + self.hand_dim
        if self.act_dim != self.expected_action_dim:
            raise ValueError(
                f"Expected action_dim={self.expected_action_dim} (7D palm IK + {self.hand_dim} Wuji qpos), "
                f"got {self.act_dim}. Check ur5e_wuji action config."
            )

        # The floating Wuji log used right_palm_link as the mocap/base body, so the fixed
        # mocap-to-palm offset is identity for the UR5e IK frame right_palm_link.
        self.mocap_to_palm_pose = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )

        self.target_pose_path = Path(target_pose_path) if target_pose_path else None
        self.uses_dexonomy_grasp = (
            self.target_pose_path is None or self.target_pose_path.suffix == ".npy"
        )
        self.tweezers_in_mocap_pose: torch.Tensor | None = None
        self.mocap_in_tweezers_pose: torch.Tensor | None = None
        self.palm_in_tweezers_pose: torch.Tensor | None = None
        self.target_tweezers_in_palm_pose: torch.Tensor | None = None
        self.target_joint_qpos: dict[str, float] = {}
        self.dexonomy_pregrasp_rows: torch.Tensor | None = None
        self.dexonomy_grasp_rows: torch.Tensor | None = None
        self.dexonomy_grasp_row: torch.Tensor | None = None

        if self.uses_dexonomy_grasp:
            if self.target_pose_path is None:
                dexonomy_phases, source_obj_pose = _builtin_dexonomy_payload(
                    self.expected_action_dim
                )
            else:
                dexonomy_phases, source_obj_pose = _load_dexonomy_payload(
                    self.target_pose_path,
                    self.expected_action_dim,
                )
            source_obj_pose_tensor = torch.as_tensor(
                source_obj_pose,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            source_obj_pose_tensor[:, 3:7] = F.normalize(
                source_obj_pose_tensor[:, 3:7], dim=-1
            )
            source_obj_inv = pose_inv_wxyz(source_obj_pose_tensor)

            self.dexonomy_pregrasp_rows = self._dexonomy_rows_to_tweezers_local(
                dexonomy_phases["pregrasp_qpos"],
                source_obj_inv,
            )
            self.dexonomy_grasp_rows = self._dexonomy_rows_to_tweezers_local(
                dexonomy_phases["grasp_qpos"],
                source_obj_inv,
            )
            self.dexonomy_grasp_row = self.dexonomy_grasp_rows[-1]

            self.palm_in_tweezers_pose = self.dexonomy_grasp_row[:7]
            self.target_tweezers_in_palm_pose = pose_inv_wxyz(
                self.palm_in_tweezers_pose.unsqueeze(0)
            )[0]
            self.target_hand_action = self.dexonomy_grasp_row[7:]
        else:
            rel_pos, rel_quat = _load_tweezers_rel_pose_from_file(self.target_pose_path)
            self.tweezers_in_mocap_pose = torch.tensor(
                [*rel_pos, *rel_quat],
                device=self.device,
                dtype=torch.float32,
            )
            self.tweezers_in_mocap_pose[3:7] = F.normalize(
                self.tweezers_in_mocap_pose[3:7], dim=0
            )
            self.mocap_in_tweezers_pose = pose_inv_wxyz(
                self.tweezers_in_mocap_pose.unsqueeze(0)
            )[0]
            self.target_tweezers_in_palm_pose = self.tweezers_in_mocap_pose

            self.target_joint_qpos = _load_wuji_joint_qpos_from_file(
                self.target_pose_path
            )
            self.target_hand_action = self._build_target_hand_action()

        self.squeeze_hand_action = torch.tensor(
            WUJI_SQUEEZE_QPOS,
            device=self.device,
            dtype=torch.float32,
        )
        self.light_squeeze_hand_action = (
            0.9 * self.target_hand_action + 0.1 * self.squeeze_hand_action
        )
        self.release_hand_action = (
            0.5 * self.target_hand_action + 0.5 * self.squeeze_hand_action
        )
        self.pregrasp_steps = max(1, int(pregrasp_steps))
        self.grasp_steps = max(1, int(grasp_steps))
        self.grasp_hold_steps = max(1, int(grasp_hold_steps))
        self.lift_steps = max(1, int(lift_steps))
        self.adjust_steps = max(1, int(adjust_steps))
        self.align_xy_steps = max(1, int(align_xy_steps))
        self.squeeze_steps = max(1, int(squeeze_steps))
        self.transport_steps = max(1, int(transport_steps))
        self.transport_hold_steps = max(1, int(transport_hold_steps))
        self.release_steps = max(1, int(release_steps))
        self.pregrasp_z_offset = float(pregrasp_z_offset)
        self.lift_height = float(lift_height)
        self.post_align_lift_height = float(post_align_lift_height)
        self.align_xy_x_offset = float(align_xy_x_offset)
        self.align_xy_y_offset = float(align_xy_y_offset)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)
        self.pregrasp_pos_tol = float(pregrasp_pos_tol)
        self.pregrasp_rot_tol = float(pregrasp_rot_tol)
        self.dexonomy_max_pos_step = float(dexonomy_max_pos_step)
        self.dexonomy_max_rot_step = float(dexonomy_max_rot_step)
        self.dexonomy_max_hand_step = float(dexonomy_max_hand_step)
        self.dexonomy_min_interp_steps = max(1, int(dexonomy_min_interp_steps))
        if (
            self.dexonomy_max_pos_step <= 0.0
            or self.dexonomy_max_rot_step <= 0.0
            or self.dexonomy_max_hand_step <= 0.0
        ):
            raise ValueError("Dexonomy speed limits must be positive.")
        self.loop = bool(loop)
        self.tweezers_site_name = str(tweezers_site_name)
        self.tube_site_name = str(tube_site_name)
        self.beaker_center_site_name = str(beaker_center_site_name)
        self._tweezers_site_id: int | None = None
        self._tube_site_id: int | None = None
        self._beaker_center_site_id: int | None = None

        self.traj: torch.Tensor | None = None
        self.traj_len: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None
        self.state: torch.Tensor | None = None
        self.complete: torch.Tensor | None = None
        self._latest_pos_err: torch.Tensor | None = None
        self._latest_rot_err: torch.Tensor | None = None
        self._latest_action: torch.Tensor | None = None
        self._debug_step = 0
        self._debug_print_interval = 10
        self.server = server
        self._contact_force_text = None
        self._success_debug_text = None
        self._geom_id_to_body_id: dict[int, int] = {}
        self._geom_id_to_name: dict[int, str] = {}
        self._tweezers_body_ids: set[int] = set()
        self._tweezers_geom_ids: set[int] = set()
        self._wuji_collision_geom_ids: set[int] = set()
        self._tweezers_weight_n = 0.0
        self._init_pose_print_debug()
        self._init_contact_force_debug()
        self._init_success_debug()

        print(
            "Loaded Wuji tweezer state policy: "
            f"pregrasp_steps={self.pregrasp_steps}, grasp_steps={self.grasp_steps}, "
            f"hold_steps={self.grasp_hold_steps}, lift_steps={self.lift_steps}, "
            f"adjust_steps={self.adjust_steps}, "
            f"align_xy_steps={self.align_xy_steps}, "
            f"squeeze_steps={self.squeeze_steps}, "
            f"transport_steps={self.transport_steps}, "
            f"transport_hold_steps={self.transport_hold_steps}, "
            f"release_steps={self.release_steps}, "
            f"lift_height={self.lift_height}, "
            f"post_align_lift_height={self.post_align_lift_height}, "
            f"align_xy_x_offset={self.align_xy_x_offset}, "
            f"align_xy_y_offset={self.align_xy_y_offset}, "
            f"pos_tol={self.pos_tol}, rot_tol={self.rot_tol}, "
            f"pregrasp_pos_tol={self.pregrasp_pos_tol}, pregrasp_rot_tol={self.pregrasp_rot_tol}, "
            f"action_dim={self.act_dim}"
        )
        if self.target_pose_path is None:
            print("Loaded built-in Dexonomy grasp_qpos")
        else:
            source = (
                "Dexonomy grasp_qpos npy"
                if self.uses_dexonomy_grasp
                else "Wuji tweezer rel_pose + qpos"
            )
            print(f"Loaded {source} from: {self.target_pose_path}")
        if self.uses_dexonomy_grasp:
            print(
                "Dexonomy timing enabled: "
                f"min_interp_steps={self.dexonomy_min_interp_steps}, "
                f"max_pos_step={self.dexonomy_max_pos_step}, "
                f"max_rot_step={self.dexonomy_max_rot_step}, "
                f"max_hand_step={self.dexonomy_max_hand_step}, "
                f"pregrasp_rows={self.dexonomy_pregrasp_rows.shape[0]}, "
                f"grasp_rows={self.dexonomy_grasp_rows.shape[0]}, "
                f"squeeze_steps={self.squeeze_steps}"
            )

    def reset(self) -> None:
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.state = None
        self.complete = None
        self._set_env_policy_complete(False)
        self._latest_pos_err = None
        self._latest_rot_err = None
        self._latest_action = None
        self._debug_step = 0
        if self._success_debug_text is not None:
            self._success_debug_text.value = "RESET"

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
            self.state[env_ids] = WujiTweezersManipulateState.PREGRASP.value
        if self.complete is not None:
            self.complete[env_ids] = False
        self._set_env_policy_complete(False, env_ids)

        if 0 in env_ids.detach().cpu().tolist():
            self._debug_step = 0
            if self._success_debug_text is not None:
                self._success_debug_text.value = "RESET"

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

        self._update_contact_force_display()
        self._update_success_debug_text()
        self._latest_action = action.detach().clone()
        self._set_env_policy_complete(self.complete)
        return action

    def init_buffers(self) -> None:
        max_len = (
            max(
                self.pregrasp_steps,
                self.grasp_steps,
                self.grasp_hold_steps,
                self.lift_steps,
                self.adjust_steps,
                self.align_xy_steps,
                self.squeeze_steps,
                self.transport_steps,
                self.transport_hold_steps,
                self.release_steps,
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
            WujiTweezersManipulateState.PREGRASP.value,
            device=self.device,
            dtype=torch.long,
        )
        self.complete = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._set_env_policy_complete(self.complete)

    def plan_state_batch(self, obs, plan_mask: torch.Tensor) -> None:
        state_table = {
            WujiTweezersManipulateState.PREGRASP.value: self._plan_pregrasp,
            WujiTweezersManipulateState.GRASP.value: self._plan_grasp,
            WujiTweezersManipulateState.GRASP_HOLD.value: self._plan_grasp_hold,
            WujiTweezersManipulateState.LIFT.value: self._plan_lift,
            WujiTweezersManipulateState.ADJUST.value: self._plan_adjust,
            WujiTweezersManipulateState.ALIGN_XY.value: self._plan_align_xy,
            WujiTweezersManipulateState.POST_ALIGN_LIFT.value: self._plan_post_align_lift,
            WujiTweezersManipulateState.TRANSPORT.value: self._plan_transport,
            WujiTweezersManipulateState.TRANSPORT_HOLD.value: self._plan_transport_hold,
            WujiTweezersManipulateState.SQUEEZE.value: self._plan_squeeze,
            WujiTweezersManipulateState.RELEASE.value: self._plan_release,
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
        if self.uses_dexonomy_grasp:
            return self._update_dexonomy_traj_progress(obs, target_action)

        last = self.idx == (self.traj_len - 1)
        pose_close = self.target_action_is_reached(obs, target_action, last)
        advance = pose_close & (~last)
        self.idx = torch.minimum(self.idx + advance.long(), self.traj_len - 1)
        done = last & pose_close & (self.traj_len > 0)
        release_complete = (
            self.state == WujiTweezersManipulateState.RELEASE.value
        ) & done
        if release_complete.any():
            self.complete[release_complete] = True
        if not self.loop:
            done = done & (self.state != WujiTweezersManipulateState.RELEASE.value)
        return done

    def _update_dexonomy_traj_progress(
        self, obs, target_action: torch.Tensor
    ) -> torch.Tensor:
        last = self.idx == (self.traj_len - 1)
        self.target_action_is_reached(obs, target_action, last)
        advance = ~last
        self.idx = torch.minimum(self.idx + advance.long(), self.traj_len - 1)
        done = last & (self.traj_len > 0)
        release_complete = (
            self.state == WujiTweezersManipulateState.RELEASE.value
        ) & done
        if release_complete.any():
            self.complete[release_complete] = True
        if not self.loop:
            done = done & (self.state != WujiTweezersManipulateState.RELEASE.value)
        return done

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

        flag = getattr(self.env, "wuji_tweezers_manipulate_policy_complete", None)
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
                    f"Cannot set wuji_tweezers_manipulate_policy_complete for {env_ids.numel()} envs from value shape {tuple(value.shape)}."
                )
        self.env.wuji_tweezers_manipulate_policy_complete = flag

    def target_action_is_reached(
        self,
        obs,
        target_action: torch.Tensor,
        last: torch.Tensor,
    ) -> torch.Tensor:
        palm_pose_w = self._get_palm_pose_w(obs)
        pos_err = torch.linalg.norm(palm_pose_w[:, :3] - target_action[:, :3], dim=-1)
        rot_err = quat_distance(
            F.normalize(palm_pose_w[:, 3:7], dim=-1),
            F.normalize(target_action[:, 3:7], dim=-1),
        )

        self._latest_pos_err = pos_err.detach().clone()
        self._latest_rot_err = rot_err.detach().clone()
        # self._debug_print_pose_error(obs, palm_pose_w, target_action, pos_err, rot_err)

        pos_tol, rot_tol = self._target_reach_tolerance()
        pos_close = pos_err <= pos_tol
        pose_close = pos_close & (rot_err <= rot_tol)
        return torch.where(last, pose_close, pos_close)

    def _target_reach_tolerance(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_tol = torch.full(
            (self.num_envs,),
            self.pos_tol,
            device=self.device,
            dtype=torch.float32,
        )
        rot_tol = torch.full_like(pos_tol, self.rot_tol)

        pregrasp = self.state == WujiTweezersManipulateState.PREGRASP.value
        pos_tol = torch.where(
            pregrasp,
            torch.full_like(pos_tol, self.pregrasp_pos_tol),
            pos_tol,
        )
        rot_tol = torch.where(
            pregrasp,
            torch.full_like(rot_tol, self.pregrasp_rot_tol),
            rot_tol,
        )
        return pos_tol, rot_tol

    def advance_state(self, done_mask: torch.Tensor) -> None:
        new_state = self.state.clone()
        pregrasp_done = done_mask & (
            self.state == WujiTweezersManipulateState.PREGRASP.value
        )
        grasp_done = done_mask & (
            self.state == WujiTweezersManipulateState.GRASP.value
        )
        grasp_hold_done = done_mask & (
            self.state == WujiTweezersManipulateState.GRASP_HOLD.value
        )
        lift_done = done_mask & (
            self.state == WujiTweezersManipulateState.LIFT.value
        )
        adjust_done = done_mask & (
            self.state == WujiTweezersManipulateState.ADJUST.value
        )
        align_xy_done = done_mask & (
            self.state == WujiTweezersManipulateState.ALIGN_XY.value
        )
        post_align_lift_done = done_mask & (
            self.state == WujiTweezersManipulateState.POST_ALIGN_LIFT.value
        )
        transport_done = done_mask & (
            self.state == WujiTweezersManipulateState.TRANSPORT.value
        )
        transport_hold_done = done_mask & (
            self.state == WujiTweezersManipulateState.TRANSPORT_HOLD.value
        )
        squeeze_done = done_mask & (
            self.state == WujiTweezersManipulateState.SQUEEZE.value
        )
        release_done = done_mask & (
            self.state == WujiTweezersManipulateState.RELEASE.value
        )

        new_state[pregrasp_done] = WujiTweezersManipulateState.GRASP.value
        new_state[grasp_done] = WujiTweezersManipulateState.GRASP_HOLD.value
        new_state[grasp_hold_done] = WujiTweezersManipulateState.LIFT.value
        new_state[squeeze_done] = WujiTweezersManipulateState.POST_ALIGN_LIFT.value
        new_state[lift_done] = WujiTweezersManipulateState.ADJUST.value
        new_state[adjust_done] = WujiTweezersManipulateState.ALIGN_XY.value
        new_state[align_xy_done] = WujiTweezersManipulateState.SQUEEZE.value
        new_state[post_align_lift_done] = (
            WujiTweezersManipulateState.TRANSPORT.value
        )
        new_state[transport_done] = WujiTweezersManipulateState.TRANSPORT_HOLD.value
        new_state[transport_hold_done] = WujiTweezersManipulateState.RELEASE.value
        new_state[release_done] = WujiTweezersManipulateState.PREGRASP.value

        self.state = new_state
        self.traj_len[done_mask] = 0
        self.idx[done_mask] = 0

    def _plan_pregrasp(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        if self.uses_dexonomy_grasp:
            if self.dexonomy_pregrasp_rows is None:
                raise RuntimeError("Missing Dexonomy pregrasp qpos rows.")
            return self._plan_dexonomy_phase(obs, idxs, self.dexonomy_pregrasp_rows)

        start_pose_w = self._get_palm_pose_w(obs)[idxs]
        target_pose_w = self._grasp_pose_from_tweezers(
            self._get_tweezers_pose_w(obs)[idxs]
        )
        target_pose_w[:, 2] += self.pregrasp_z_offset
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            start_hand_action,
            self.pregrasp_steps,
        )

    def _plan_grasp(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        if self.uses_dexonomy_grasp:
            if self.dexonomy_grasp_rows is None:
                raise RuntimeError("Missing Dexonomy grasp qpos rows.")
            traj = self._plan_dexonomy_phase(obs, idxs, self.dexonomy_grasp_rows)
            start_hand_action = self._extract_current_hand_action(obs)[idxs]
            target_hand_action = self.light_squeeze_hand_action.unsqueeze(0).repeat(
                idxs.numel(), 1
            )
            traj[:, :, 7:] = qpos_linear_interpolate(
                start_hand_action,
                target_hand_action,
                traj.shape[1] - 1,
            )
            return traj

        tweezers_pose_w = self._get_tweezers_pose_w(obs)[idxs]
        start_pose_w = self._get_palm_pose_w(obs)[idxs]
        target_pose_w = self._grasp_pose_from_tweezers(tweezers_pose_w)
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = self.light_squeeze_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            start_hand_action,
            target_hand_action,
            self.grasp_steps,
        )

    def _plan_grasp_hold(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        hold_pose_w = self._get_palm_pose_w(obs)[idxs]
        hand_action = self.light_squeeze_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            hold_pose_w,
            hold_pose_w,
            hand_action,
            hand_action,
            self.grasp_hold_steps,
        )

    def _plan_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        return self._plan_lift_by_height(obs, idxs, self.lift_height)

    def _plan_adjust(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_palm_pose_w(obs)[idxs]
        tweezers_pose_w = self._get_tweezers_pose_w(obs)[idxs]
        initial_tweezers_pose_w = self._get_initial_tweezers_pose_w()[idxs]
        tweezers_delta_w = compose_pose_wxyz(
            initial_tweezers_pose_w,
            pose_inv_wxyz(tweezers_pose_w),
        )
        target_pose_w = compose_pose_wxyz(tweezers_delta_w, start_pose_w)
        target_pose_w[:, :3] = start_pose_w[:, :3]
        hand_action = self.light_squeeze_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.adjust_steps,
        )

    def _plan_post_align_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        return self._plan_lift_by_height(
            obs,
            idxs,
            self.post_align_lift_height,
            self.squeeze_hand_action,
        )

    def _plan_lift_by_height(
        self,
        obs,
        idxs: torch.Tensor,
        height: float,
        hand_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        start_pose_w = self._get_palm_pose_w(obs)[idxs]
        target_pose_w = start_pose_w.clone()
        target_pose_w[:, 2] = target_pose_w[:, 2] + height
        hand_action = (
            hand_action if hand_action is not None else self._hold_hand_action()
        )
        hand_action = hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.lift_steps,
        )

    def _plan_align_xy(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_palm_pose_w(obs)[idxs]
        tweezers_site_pos_w = self._get_tweezers_site_pos_w()[idxs]
        tube_site_pos_w = self._get_tube_site_pos_w(obs)[idxs]

        target_pose_w = start_pose_w.clone()
        target_pose_w[:, :2] = target_pose_w[:, :2] + (
            tube_site_pos_w[:, :2] - tweezers_site_pos_w[:, :2]
        )
        target_pose_w[:, 0] += self.align_xy_x_offset
        target_pose_w[:, 1] += self.align_xy_y_offset
        hand_action = self._hold_hand_action().unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.align_xy_steps,
        )

    def _plan_squeeze(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        hold_pose_w = self._get_palm_pose_w(obs)[idxs]
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = self.squeeze_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            hold_pose_w,
            hold_pose_w,
            start_hand_action,
            target_hand_action,
            self.squeeze_steps,
        )

    def _plan_transport(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose_w = self._get_palm_pose_w(obs)[idxs]
        tube_pos_w = self._get_tube_root_pos_w(obs)[idxs]
        beaker_pos_w = self._get_beaker_center_pos_w()[idxs]

        target_pose_w = start_pose_w.clone()
        target_pose_w[:, :2] = start_pose_w[:, :2] + (
            beaker_pos_w[:, :2] - tube_pos_w[:, :2]
        )
        hand_action = self.squeeze_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            start_pose_w,
            target_pose_w,
            hand_action,
            hand_action,
            self.transport_steps,
        )

    def _plan_transport_hold(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        hold_pose_w = self._get_palm_pose_w(obs)[idxs]
        hand_action = self.squeeze_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        return self.plan_action_traj(
            hold_pose_w,
            hold_pose_w,
            hand_action,
            hand_action,
            self.transport_hold_steps,
        )

    def _plan_release(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        hold_pose_w = self._get_palm_pose_w(obs)[idxs]
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        target_hand_action = self.release_hand_action.unsqueeze(0).repeat(
            idxs.numel(), 1
        )
        return self.plan_action_traj(
            hold_pose_w,
            hold_pose_w,
            start_hand_action,
            target_hand_action,
            self.release_steps,
        )

    def plan_action_traj(
        self,
        start_pose_w: torch.Tensor,
        target_pose_w: torch.Tensor,
        start_hand_action: torch.Tensor,
        target_hand_action: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        pose_traj = self._build_pose_traj(start_pose_w, target_pose_w, num_steps)
        hand_traj = qpos_linear_interpolate(
            start_hand_action,
            target_hand_action,
            num_steps,
        )
        return torch.cat((pose_traj, hand_traj), dim=-1)

    def _plan_dexonomy_phase(
        self,
        obs,
        idxs: torch.Tensor,
        local_rows: torch.Tensor,
    ) -> torch.Tensor:
        tweezers_pose_w = self._get_tweezers_pose_w(obs)[idxs]
        dexonomy_world_rows = self._dexonomy_local_rows_to_world_actions(
            local_rows,
            tweezers_pose_w,
        )

        start_pose_w = self._get_palm_pose_w(obs)[idxs]
        start_hand_action = self._extract_current_hand_action(obs)[idxs]
        current_action = torch.cat(
            (start_pose_w, start_hand_action),
            dim=-1,
        )
        segments = []

        for row_idx in range(dexonomy_world_rows.shape[1]):
            target_action = dexonomy_world_rows[:, row_idx]
            segment = self._plan_speed_limited_action_traj(
                current_action, target_action
            )
            segments.append(segment[:, 1:])
            current_action = target_action

        return torch.cat(segments, dim=1)

    def _dexonomy_rows_to_tweezers_local(
        self,
        rows_np: np.ndarray,
        source_obj_inv: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.as_tensor(rows_np, device=self.device, dtype=torch.float32).clone()
        rows[:, 3:7] = F.normalize(rows[:, 3:7], dim=-1)
        source_obj_inv_rows = source_obj_inv.repeat(rows.shape[0], 1)
        rows[:, :7] = compose_pose_wxyz(source_obj_inv_rows, rows[:, :7])
        return rows

    def _dexonomy_local_rows_to_world_actions(
        self,
        rows: torch.Tensor,
        tweezers_pose_w: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tweezers_pose_w.shape[0]
        num_rows = rows.shape[0]
        tweezers_rows = (
            tweezers_pose_w[:, None, :].repeat(1, num_rows, 1).reshape(-1, 7)
        )
        local_poses = rows[None, :, :7].repeat(batch_size, 1, 1).reshape(-1, 7)
        world_poses = compose_pose_wxyz(tweezers_rows, local_poses).reshape(
            batch_size,
            num_rows,
            7,
        )
        hand_actions = rows[None, :, 7:].repeat(batch_size, 1, 1)
        return torch.cat((world_poses, hand_actions), dim=-1)

    def _plan_speed_limited_action_traj(
        self,
        current_action: torch.Tensor,
        target_action: torch.Tensor,
    ) -> torch.Tensor:
        steps = self._dexonomy_speed_limited_steps(current_action, target_action)
        return self.plan_action_traj(
            current_action[:, :7],
            target_action[:, :7],
            current_action[:, 7:],
            target_action[:, 7:],
            steps,
        )

    def _dexonomy_speed_limited_steps(
        self,
        current_action: torch.Tensor,
        target_action: torch.Tensor,
    ) -> int:
        pos_dist = torch.linalg.norm(
            target_action[:, :3] - current_action[:, :3], dim=-1
        )
        rot_dist = quat_distance(
            F.normalize(current_action[:, 3:7], dim=-1),
            F.normalize(target_action[:, 3:7], dim=-1),
        )
        hand_dist = torch.amax(
            torch.abs(target_action[:, 7:] - current_action[:, 7:]),
            dim=-1,
        )

        pos_steps = torch.ceil(pos_dist / self.dexonomy_max_pos_step)
        rot_steps = torch.ceil(rot_dist / self.dexonomy_max_rot_step)
        hand_steps = torch.ceil(hand_dist / self.dexonomy_max_hand_step)
        required_steps = torch.maximum(torch.maximum(pos_steps, rot_steps), hand_steps)
        steps = int(torch.amax(required_steps).detach().cpu().item())
        return max(self.dexonomy_min_interp_steps, steps)

    def _hold_hand_action(self) -> torch.Tensor:
        return self.light_squeeze_hand_action

    def _grasp_pose_from_tweezers(self, tweezers_pose_w: torch.Tensor) -> torch.Tensor:
        if self.uses_dexonomy_grasp:
            if self.palm_in_tweezers_pose is None:
                raise RuntimeError("Missing Dexonomy palm-in-tweezers pose.")
            return compose_pose_wxyz(
                tweezers_pose_w,
                self.palm_in_tweezers_pose.unsqueeze(0).repeat(
                    tweezers_pose_w.shape[0], 1
                ),
            )

        if self.mocap_in_tweezers_pose is None:
            raise RuntimeError("Missing mocap-in-tweezers pose.")
        mocap_pose_w = compose_pose_wxyz(
            tweezers_pose_w,
            self.mocap_in_tweezers_pose.unsqueeze(0).repeat(
                tweezers_pose_w.shape[0], 1
            ),
        )
        return compose_pose_wxyz(
            mocap_pose_w,
            self.mocap_to_palm_pose.unsqueeze(0).repeat(tweezers_pose_w.shape[0], 1),
        )

    def _build_pose_traj(
        self,
        start_pose: torch.Tensor,
        end_pose: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos=start_pose[:, :3],
            end_pos=end_pose[:, :3],
            start_quat=F.normalize(start_pose[:, 3:7], dim=-1),
            end_quat=F.normalize(end_pose[:, 3:7], dim=-1),
            num_steps=steps,
        )
        return torch.cat((pos_traj, quat_traj), dim=-1)

    def _build_target_hand_action(self) -> torch.Tensor:
        return torch.tensor(
            [self._target_for_joint_expr(name) for name in WUJI_JOINT_ORDER],
            device=self.device,
            dtype=torch.float32,
        )

    def _target_for_joint_expr(self, joint_expr: str) -> float:
        suffix = joint_expr.replace(".*", "")
        for joint_name, value in self.target_joint_qpos.items():
            if joint_name == suffix or joint_name.endswith(suffix):
                return value
        raise KeyError(
            f"No Wuji target qpos found for joint expression {joint_expr!r} in {self.target_pose_path}."
        )

    def _extract_current_hand_action(self, obs) -> torch.Tensor:
        qpos = observation_tensor(
            obs, "current_qpos", self.device, self.num_envs, select_first=True
        )
        return qpos[:, -self.hand_dim :]

    def _get_palm_pose_w(self, obs) -> torch.Tensor:
        pose = observation_tensor(
            obs, "palm_body_pose_w", self.device, self.num_envs, select_first=True
        )[:, :7]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    return WujiTweezersManipulatePolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        target_pose_path=replay_path or None,
        loop=False,
        server=server,
    )
