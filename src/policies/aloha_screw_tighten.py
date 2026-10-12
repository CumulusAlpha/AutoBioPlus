import math

import torch
import viser

from src.utils.policy_helpers.aloha_screw_tighten_common import (
    CAP_COLLISION_GEOM_NAMES as CAP_COLLISION_GEOM_NAMES,
    CAP_ROTATION_AXIS_LOCAL as CAP_ROTATION_AXIS_LOCAL,
    FINGERTIP_SPHERE_GEOM_NAMES as FINGERTIP_SPHERE_GEOM_NAMES,
    GRIPPER_COLLISION_GEOM_NAMES as GRIPPER_COLLISION_GEOM_NAMES,
    AlohaScrewTightenStage as AlohaScrewTightenStage,
    hold_with_gripper as hold_with_gripper,
    interpolate_dual as interpolate_dual,
    pack_action as pack_action,
    random_sphere_torch as random_sphere_torch,
)
from src.utils.policy_helpers.aloha_screw_tighten_contacts import (
    AlohaContactDiagnostics,
)
from src.utils.policy_helpers.aloha_screw_tighten_plans import AlohaMotionPlans
from src.utils.quat_util import (
    quat_distance,
    quat_rotate_vec,
)


class AlohaScrewTightenPolicy(AlohaContactDiagnostics, AlohaMotionPlans):
    def __init__(
        self,
        action_shape,
        device,
        traj_steps=20,
        pos_threshold=0.02,
        rot_threshold=0.1,
        server: viser.ViserServer = None,
        env=None,
    ):
        self.device = device
        self.traj_steps = traj_steps
        self.default_pos_threshold = pos_threshold
        self.default_rot_threshold = rot_threshold
        self.waypoint_max_wait_steps = 12
        self.server = server

        self.B = action_shape[0]
        self.action_dim = action_shape[-1]
        self.batch_idx = torch.arange(self.B, device=device)

        self.traj = None
        self.traj_len = None
        self.idx = None
        self.stage = None
        self.target_pos = None
        self.waypoint_wait_count = None
        self.target_quat1 = torch.tensor(
            [0.270593, 0.653276, 0.270593, -0.653276], device=self.device
        )

        self.target_quat2 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.target_quat3 = torch.tensor(
            [0.701057, 0.092295, 0.701057, -0.092295], device=self.device
        )
        self.phase_steps = 30
        self.grasp_settle_steps = 10
        self.tube_pos_start = torch.zeros((self.B, 3), device=self.device)
        self.cap_pos_start = torch.zeros((self.B, 3), device=self.device)
        self.cap_pos_mid = torch.zeros((self.B, 3), device=self.device)
        self.tube_pos_mid = torch.zeros((self.B, 3), device=self.device)
        self.arm1_ref_pos = torch.zeros((self.B, 3), device=self.device)

        self.tighten_total_loops = 5
        self.tighten_first_spin_loops = 3
        self.tighten_spin_loops = 7
        self.tighten_angle = math.pi / 10.0
        self.tighten_dz = -0.003 / 1000.0
        self.tighten_interp_steps = 20
        self.tighten_cap_grip_close = 0.016
        self.tighten_cap_grip_open = 0.037
        self.tighten_cap_grip_seat = 0.026
        self.tighten_tube_grip_hold = 0.001
        self.cap_grasp_local_offset = torch.tensor(
            [-0.005, 0.0, 0.006],
            device=self.device,
        )
        self.cap_seat_loose_steps = 12
        self.cap_seat_close_steps = 8
        self.spin_align_steps = 10
        self.tube_grasp_z_offset = 0.08
        self.pre_insert_tilt_steps = 12
        self.pre_insert_max_tilt = 0.14
        self.cap_site_to_bottom = 0.011
        self.insert_align_x_bias = -0.003

        self._tighten_substage_spin = 0
        self._tighten_substage_release = 1
        self._tighten_substage_return_mid1 = 2
        self._tighten_substage_return_mid2 = 3

        self.tighten_loop_idx = torch.zeros(
            (self.B,), device=self.device, dtype=torch.long
        )
        self.tighten_substage = torch.zeros(
            (self.B,), device=self.device, dtype=torch.long
        )
        self.tighten_mid1_pos = torch.zeros((self.B, 3), device=self.device)
        self.tighten_mid1_quat = torch.zeros((self.B, 4), device=self.device)
        self.tighten_mid2_pos = torch.zeros((self.B, 3), device=self.device)
        self.tighten_spin_grasp_z_offset = torch.zeros((self.B,), device=self.device)
        self.debug_log = True
        self.debug_step = 0
        self.env = env
        self._cap_geom_ids: set[int] = set()
        self._cap_body_ids: set[int] = set()
        self._gripper_collision_geom_ids: set[int] = set()
        self._fingertip_sphere_geom_ids: set[int] = set()
        self._geom_id_to_name: dict[int, str] = {}
        self._geom_id_to_body_id: dict[int, int] = {}

        # Per-stage thresholds. Stages not listed here fall back to defaults.
        self.stage_pos_threshold = {
            AlohaScrewTightenStage.STAGE_ALIGN_INSERT: 0.001,
            AlohaScrewTightenStage.STAGE_PRE_INSERT_TILT: 0.002,
            AlohaScrewTightenStage.STAGE_INSERT: 0.002,
            AlohaScrewTightenStage.STAGE_SPIN_ALIGN: 0.002,
        }
        self.stage_rot_threshold = {
            AlohaScrewTightenStage.STAGE_APPROACH_MID: 0.04,
            AlohaScrewTightenStage.STAGE_DUAL_GRASP_CLOSE: 0.04,
            AlohaScrewTightenStage.STAGE_DUAL_GRASP_SETTLE: 0.04,
            AlohaScrewTightenStage.STAGE_DUAL_LIFT_MOVE: 0.04,
            AlohaScrewTightenStage.STAGE_ALIGN_INSERT: 0.04,
            AlohaScrewTightenStage.STAGE_PRE_INSERT_TILT: 0.04,
            AlohaScrewTightenStage.STAGE_INSERT: 0.04,
            AlohaScrewTightenStage.STAGE_SPIN_ALIGN: 0.04,
        }

        self.align_waypoint_pos_th = 0.01
        self.align_final_pos_th = 0.0005
        self.align_final_rot_th = 0.003

        self.insert_waypoint_pos_th = 0.01
        self.insert_final_pos_th = 0.002
        self.insert_final_rot_th = 0.04

        self.align_stable_steps = 3
        self.insert_stable_steps = 3
        self.align_stable_count = torch.zeros(
            (self.B,), device=self.device, dtype=torch.long
        )
        self.insert_stable_count = torch.zeros(
            (self.B,), device=self.device, dtype=torch.long
        )

        self.target_frame = None
        if self.server is not None:
            self.target_frame = self.server.scene.add_frame(
                name="target_frame",
                axes_radius=0.001,
                origin_radius=0.005,
            )

    def _need_init(self, obs):
        expected_device = torch.device(self.device)
        return (
            self.traj is None
            or self.traj.shape[0] != self.B
            or self.traj.device != expected_device
        )

    def init_buffers(self, obs):
        T = self.traj_steps + 1
        self.traj = torch.zeros((self.B, T, self.action_dim), device=self.device)
        self.traj_len = torch.zeros((self.B,), device=self.device, dtype=torch.long)
        self.idx = torch.zeros((self.B,), device=self.device, dtype=torch.long)
        self.stage = torch.zeros((self.B,), device=self.device, dtype=torch.long)
        self.target_pos = torch.zeros((self.B, 3), device=self.device)
        self.waypoint_wait_count = torch.zeros(
            (self.B,), device=self.device, dtype=torch.long
        )
        self.batch_idx = torch.arange(self.B, device=self.device)
        self._plan_stage_batch(
            obs, torch.ones((self.B,), device=self.device, dtype=torch.bool)
        )

    def _advance_waypoint(self, mask):
        not_last = self.idx < (self.traj_len - 1)
        self.waypoint_wait_count = torch.where(
            not_last,
            self.waypoint_wait_count + 1,
            torch.zeros_like(self.waypoint_wait_count),
        )
        timeout = not_last & (self.waypoint_wait_count >= self.waypoint_max_wait_steps)
        advance = not_last & (mask | timeout)
        if advance.any():
            self.idx[advance] += 1
            self.waypoint_wait_count[advance] = 0

    def _plan_stage_batch(self, obs, mask):
        if not mask.any():
            return
        table = self._get_stage_table()
        for stage_val, fn in table.items():
            stage_mask = mask & (self.stage == int(stage_val))
            if not stage_mask.any():
                continue
            idxs = torch.where(stage_mask)[0]
            self.apply_plan(idxs, fn(obs, idxs))

    def apply_plan(self, idxs, traj):
        length = traj.shape[1]
        current_length = self.traj.shape[1]
        if length > current_length:
            new_traj = torch.zeros(
                (self.B, length, self.action_dim),
                device=self.traj.device,
                dtype=self.traj.dtype,
            )
            new_traj[:, :current_length] = self.traj
            self.traj = new_traj
        self.traj[idxs] = 0
        self.traj[idxs, :length] = traj
        self.traj_len[idxs] = length
        self.idx[idxs] = 0
        self.waypoint_wait_count[idxs] = 0

    def reset(self):
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.stage = None
        self.waypoint_wait_count = None
        if self.target_pos is not None:
            self.target_pos.zero_()
        self.tighten_loop_idx.zero_()
        self.tighten_substage.zero_()
        self.tighten_mid1_pos.zero_()
        self.tighten_mid1_quat.zero_()
        self.tighten_mid2_pos.zero_()
        self.tighten_spin_grasp_z_offset.zero_()
        self.arm1_ref_pos.zero_()
        self.align_stable_count.zero_()
        self.insert_stable_count.zero_()
        self.debug_step = 0
        self._set_env_tighten_complete(
            torch.zeros((self.B,), device=self.device, dtype=torch.bool)
        )

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if self.traj is None:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        self.traj[env_ids] = 0.0
        self.traj_len[env_ids] = 0
        self.idx[env_ids] = 0
        self.stage[env_ids] = int(AlohaScrewTightenStage.STAGE_DUAL_REACH)
        if self.waypoint_wait_count is not None:
            self.waypoint_wait_count[env_ids] = 0
        self.tighten_loop_idx[env_ids] = 0
        self.tighten_substage[env_ids] = self._tighten_substage_spin
        self.tighten_mid1_pos[env_ids] = 0.0
        self.tighten_mid1_quat[env_ids] = 0.0
        self.tighten_mid2_pos[env_ids] = 0.0
        self.tighten_spin_grasp_z_offset[env_ids] = 0.0
        self.arm1_ref_pos[env_ids] = 0.0
        self.cap_pos_start[env_ids] = 0.0
        self.tube_pos_start[env_ids] = 0.0
        self.cap_pos_mid[env_ids] = 0.0
        self.tube_pos_mid[env_ids] = 0.0
        self.align_stable_count[env_ids] = 0
        self.insert_stable_count[env_ids] = 0
        self._set_env_tighten_complete(
            torch.zeros((env_ids.numel(),), device=self.device, dtype=torch.bool),
            env_ids,
        )

    def __call__(self, obs):
        if self._need_init(obs):
            self.init_buffers(obs)
        need_plan = self.traj_len == 0
        if need_plan.any():
            self._plan_stage_batch(obs, need_plan)

        error = self._compute_error(obs)
        reached = self._check_waypoint_reached(obs, error)
        self._advance_waypoint(reached)
        stage_done = self._check_stage_done(obs, error)

        if stage_done.any():
            self._advance_stage(stage_done)
            self._plan_stage_batch(obs, stage_done)

        action = self.traj[self.batch_idx, self.idx]
        self._set_env_tighten_complete(
            self.stage >= int(AlohaScrewTightenStage.STAGE_DONE)
        )
        self.debug_step += 1
        return action

    def _set_env_tighten_complete(
        self, value: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        if self.env is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.B, device=self.device, dtype=torch.long)
        flag = getattr(self.env, "aloha_screw_tighten_policy_complete", None)
        if flag is None or flag.shape[0] != self.B:
            flag = torch.zeros((self.B,), device=self.device, dtype=torch.bool)
        flag[env_ids] = value.to(device=self.device, dtype=torch.bool)
        self.env.aloha_screw_tighten_policy_complete = flag

    # =====================================================
    # error
    # =====================================================

    def _compute_error(self, obs):
        # current eef pose
        cur_pos1, cur_quat1 = self._get_eef_pose(obs, arm_id=1, idxs=self.batch_idx)
        cur_pos2, cur_quat2 = self._get_eef_pose(obs, arm_id=2, idxs=self.batch_idx)

        tgt = self.traj[self.batch_idx, self.idx]

        tgt_pos1 = tgt[:, :3]
        tgt_quat1 = tgt[:, 3:7]
        tgt_pos2 = tgt[:, 8:11]
        tgt_quat2 = tgt[:, 11:15]

        pos_err1 = torch.norm(tgt_pos1 - cur_pos1, dim=-1)
        pos_err2 = torch.norm(tgt_pos2 - cur_pos2, dim=-1)

        rot_err1 = quat_distance(cur_quat1, tgt_quat1)
        rot_err2 = quat_distance(cur_quat2, tgt_quat2)

        return {
            "pos": torch.maximum(pos_err1, pos_err2),
            "rot": torch.maximum(rot_err1, rot_err2),
        }

    def _expand_quat(self, quat, batch_size):
        return quat.unsqueeze(0).expand(batch_size, -1)

    def _env_origins(self, idxs, like):
        zeros = torch.zeros((idxs.numel(), 3), device=like.device, dtype=like.dtype)
        if self.env is None or not hasattr(self.env, "scene"):
            return zeros
        env_origins = getattr(self.env.scene, "env_origins", None)
        if env_origins is None:
            return zeros
        return env_origins[idxs].to(device=like.device, dtype=like.dtype)

    def _normalize_vec(self, vec: torch.Tensor) -> torch.Tensor:
        return vec / torch.clamp(torch.linalg.norm(vec, dim=-1, keepdim=True), min=1e-6)

    def _cap_axis_w(self, cap_quat: torch.Tensor) -> torch.Tensor:
        axis_local = (
            torch.tensor(
                CAP_ROTATION_AXIS_LOCAL,
                device=self.device,
                dtype=cap_quat.dtype,
            )
            .unsqueeze(0)
            .expand(cap_quat.shape[0], -1)
        )
        axis = self._normalize_vec(quat_rotate_vec(cap_quat, axis_local))
        return torch.where(axis[:, 2:3] < 0.0, -axis, axis)

    def _get_dual_eef_pose(self, obs, idxs):
        pos1, quat1 = self._get_eef_pose(obs, 1, idxs)
        pos2, quat2 = self._get_eef_pose(obs, 2, idxs)
        return pos1, quat1, pos2, quat2

    def _update_stable_counter(self, mask, final_ok, counter):
        inc_mask = mask & final_ok
        reset_mask = mask & (~final_ok)
        counter = torch.where(inc_mask, counter + 1, counter)
        counter = torch.where(reset_mask, torch.zeros_like(counter), counter)
        return counter

    def _append_segment(self, traj_chunks, seg):
        if traj_chunks:
            seg = seg[:, 1:, :]
        traj_chunks.append(seg)

    def _thresholds_by_stage(self, dtype):
        pos_th = torch.full(
            (self.B,),
            float(self.default_pos_threshold),
            device=self.device,
            dtype=dtype,
        )
        rot_th = torch.full(
            (self.B,),
            float(self.default_rot_threshold),
            device=self.device,
            dtype=dtype,
        )

        for stage_val, th in self.stage_pos_threshold.items():
            pos_th = torch.where(
                self.stage == int(stage_val),
                torch.full_like(pos_th, float(th)),
                pos_th,
            )
        for stage_val, th in self.stage_rot_threshold.items():
            rot_th = torch.where(
                self.stage == int(stage_val),
                torch.full_like(rot_th, float(th)),
                rot_th,
            )
        return pos_th, rot_th

    def _check_waypoint_reached(self, obs, error):
        pos_th, _ = self._thresholds_by_stage(error["pos"].dtype)
        align_mask = self.stage == int(AlohaScrewTightenStage.STAGE_ALIGN_INSERT)
        insert_mask = self.stage == int(AlohaScrewTightenStage.STAGE_INSERT)
        pos_th = torch.where(
            align_mask,
            torch.full_like(pos_th, float(self.align_waypoint_pos_th)),
            pos_th,
        )
        pos_th = torch.where(
            insert_mask,
            torch.full_like(pos_th, float(self.insert_waypoint_pos_th)),
            pos_th,
        )
        return error["pos"] < pos_th

    def _check_stage_done(self, obs, error):
        last = self.idx >= (self.traj_len - 1)
        done_mask = self.stage >= int(AlohaScrewTightenStage.STAGE_DONE)
        align_mask = self.stage == int(AlohaScrewTightenStage.STAGE_ALIGN_INSERT)
        insert_mask = self.stage == int(AlohaScrewTightenStage.STAGE_INSERT)

        align_final_ok = (
            (error["pos"] < float(self.align_final_pos_th))
            & (error["rot"] < float(self.align_final_rot_th))
            & last
        )
        insert_final_ok = (
            (error["pos"] < float(self.insert_final_pos_th))
            & (error["rot"] < float(self.insert_final_rot_th))
            & last
        )

        self.align_stable_count = self._update_stable_counter(
            align_mask, align_final_ok, self.align_stable_count
        )
        self.insert_stable_count = self._update_stable_counter(
            insert_mask, insert_final_ok, self.insert_stable_count
        )

        align_done = align_mask & (
            self.align_stable_count >= int(self.align_stable_steps)
        )
        insert_done = insert_mask & (
            self.insert_stable_count >= int(self.insert_stable_steps)
        )

        pos_th, rot_th = self._thresholds_by_stage(error["pos"].dtype)
        strict_ok = (error["pos"] < pos_th) & (error["rot"] < rot_th)
        normal_done = (~done_mask) & (~align_mask) & (~insert_mask) & last & strict_ok
        return normal_done | align_done | insert_done

    def _get_stage_table(self):
        return {
            AlohaScrewTightenStage.STAGE_DUAL_REACH: self._plan_dual_reach,
            AlohaScrewTightenStage.STAGE_APPROACH_MID: self._plan_mid_approach,
            AlohaScrewTightenStage.STAGE_DUAL_GRASP_CLOSE: self._plan_grasp_close,
            AlohaScrewTightenStage.STAGE_DUAL_GRASP_SETTLE: self._plan_grasp_settle,
            AlohaScrewTightenStage.STAGE_DUAL_LIFT_MOVE: self._plan_lift_move,
            AlohaScrewTightenStage.STAGE_ALIGN_INSERT: self._plan_align,
            AlohaScrewTightenStage.STAGE_PRE_INSERT_TILT: self._plan_pre_insert_tilt,
            AlohaScrewTightenStage.STAGE_INSERT: self._plan_insert,
            AlohaScrewTightenStage.STAGE_SEAT_CAP: self._plan_seat_cap,
            AlohaScrewTightenStage.STAGE_SPIN_ALIGN: self._plan_spin_align,
            AlohaScrewTightenStage.STAGE_TIGHTEN: self._plan_tighten,
            AlohaScrewTightenStage.STAGE_DONE: self._plan_done,
        }

    def _advance_stage(self, mask):
        if not mask.any():
            return

        idxs = torch.where(mask)[0]
        tighten_mask = self.stage[idxs] == AlohaScrewTightenStage.STAGE_TIGHTEN

        normal_idxs = idxs[~tighten_mask]
        if normal_idxs.numel() > 0:
            self.stage[normal_idxs] += 1

        tighten_idxs = idxs[tighten_mask]
        if tighten_idxs.numel() == 0:
            return

        self._advance_tighten_substage(tighten_idxs)

    def _advance_tighten_substage(self, tighten_idxs):
        sub = self.tighten_substage[tighten_idxs]

        spin_mask = sub == self._tighten_substage_spin
        self.tighten_substage[tighten_idxs[spin_mask]] = self._tighten_substage_release

        release_mask = sub == self._tighten_substage_release
        release_idxs = tighten_idxs[release_mask]
        if release_idxs.numel() > 0:
            last_loop = self.tighten_loop_idx[release_idxs] >= (
                self.tighten_total_loops - 1
            )
            done_idxs = release_idxs[last_loop]
            cont_idxs = release_idxs[~last_loop]

            if done_idxs.numel() > 0:
                self.stage[done_idxs] += 1
                self.tighten_substage[done_idxs] = self._tighten_substage_spin

            if cont_idxs.numel() > 0:
                self.tighten_substage[cont_idxs] = self._tighten_substage_return_mid1

        mid1_mask = sub == self._tighten_substage_return_mid1
        self.tighten_substage[tighten_idxs[mid1_mask]] = (
            self._tighten_substage_return_mid2
        )

        mid2_mask = sub == self._tighten_substage_return_mid2
        mid2_idxs = tighten_idxs[mid2_mask]
        if mid2_idxs.numel() > 0:
            self.tighten_substage[mid2_idxs] = self._tighten_substage_spin
            self.tighten_loop_idx[mid2_idxs] += 1

    # =====================================================
    # planners
    # =====================================================

    def _get_eef_pose(self, obs, arm_id: int, idxs=None):
        """
        Args:
            obs: dict of tensors
            arm_id: 1 or 2
            idxs: tensor of indices (subset of batch)

        Returns:
            pos: (N, 3)
            quat: (N, 4)
        """

        if idxs is None:
            idxs = self.batch_idx

        key = f"aloha_{arm_id}_eef_pose"

        # (B, 7)
        eef = obs[key]

        # select batch subset
        eef = eef[idxs]

        # split
        pos = eef[..., :3]
        quat = eef[..., 3:]

        return pos, quat

    def _get_cap_pos(self, obs, idxs):
        """
        Returns:
            pos: (N, 3)
            quat: (N, 4)
        """
        pose = obs["screw_cap_pose_w"][idxs]

        pos = pose[..., :3]
        quat = pose[..., 3:]

        return pos, quat

    def _get_tube_pos(self, obs, idxs):
        """
        Returns:
            pos: (N, 3)
            quat: (N, 4)
        """
        pose = obs["screw_body_pose_w"][idxs]

        pos = pose[..., :3]
        quat = pose[..., 3:]

        return pos, quat


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return AlohaScrewTightenPolicy(
        action_shape=action_shape,
        device=device,
        server=server,
        env=env,
    )
