from __future__ import annotations

import torch
import torch.nn.functional as F
from mjlab.actuator.actuator import TransmissionType

from src.utils.policy_helpers.shadow_centrifuge_insert_common import (
    ShadowCentrifugeInsertStage as ShadowCentrifugeInsertStage,
)
from src.utils.policy_helpers.shadow_centrifuge_insert_geometry import (
    ShadowCentrifugeGeometry,
)
from src.utils.policy_helpers.shadow_centrifuge_insert_replay import (
    ShadowCentrifugeReplay,
)
from src.utils.kinematics import batched_interpolate2, batched_interpolate_linear


class ShadowCentrifugeInsertPolicy(
    ShadowCentrifugeReplay, ShadowCentrifugeGeometry
):
    """Replay shadow-hand action targets with an explicit stage planner."""

    GRASP_YAW_OFFSET_POSE = (
        0.0,
        0.0,
        0.0,
        0.923879533,
        0.0,
        0.0,
        -0.382683432,
    )
    CENTRIFUGE_50ML_SLOT_SITE_NAMES = tuple(
        f"adapter-7x50-{adapter_id}-slot-{slot_row}{slot_id}"
        for adapter_id in range(2)
        for slot_row, count in (("bottom", 2), ("middle", 3), ("top", 2))
        for slot_id in range(count)
    )
    INSERT_REL_POSE = (
        0.0,
        0.0,
        -0.03,
        0.7071067811865475,
        0.0,
        0.0,
        -0.7071067811865475,
    )

    PREGRASP_ACT_TARGET = (
        (
            -0.160973325,
            0.007877131,
            0.116037630,
            0.567829132,
            0.699396610,
            0.276795864,
            0.334362835,
            -0.640046716,
            1.022114400,
            0.009781620,
            -0.000206874,
            0.293635547,
            0.027435452,
            0.122825190,
            -0.050057866,
            0.898483336,
            -0.080864698,
            0.737147748,
            0.208147347,
            -0.070643991,
            1.307410600,
            1.309945580,
            1.500636220,
            1.570649620,
            1.928382640,
        ),
        (
            -0.155723140,
            0.010191832,
            0.114893213,
            0.569227874,
            0.698812783,
            0.277962446,
            0.332230628,
            -0.575331211,
            1.022227410,
            0.009944737,
            -0.000090826,
            0.304231912,
            -0.010854420,
            0.163440943,
            -0.057273984,
            0.904390395,
            -0.083723232,
            0.739568830,
            0.207349405,
            -0.073167607,
            1.311078070,
            1.330173020,
            1.502047060,
            1.584740040,
            1.950789330,
        ),
        (
            -0.145625114,
            0.015846122,
            0.111507103,
            0.570876300,
            0.697612584,
            0.278939873,
            0.331103325,
            -0.432295352,
            1.022318010,
            0.010044822,
            -0.000260588,
            0.327832967,
            -0.060045768,
            0.238407046,
            -0.070390500,
            0.916762829,
            -0.089013375,
            0.741889119,
            0.203857854,
            -0.078295030,
            1.318098310,
            1.368330960,
            1.506999490,
            1.605888840,
            1.975972650,
        ),
        (
            -0.136129498,
            0.022060346,
            0.109142952,
            0.571996868,
            0.696980298,
            0.277882516,
            0.331390053,
            -0.254908115,
            1.024043920,
            0.009959553,
            0.034915112,
            0.355636746,
            -0.112783894,
            0.336017787,
            -0.078907572,
            0.935049474,
            -0.093731128,
            0.743660212,
            0.200423494,
            -0.080562927,
            1.322923300,
            1.421752930,
            1.506747010,
            1.628916860,
            2.003978250,
        ),
        (
            -0.132233962,
            0.027911829,
            0.110840015,
            0.569730699,
            0.696038902,
            0.274001062,
            0.340382427,
            -0.140340373,
            0.957272887,
            0.009775547,
            0.109382853,
            0.377815008,
            -0.135564476,
            0.402240187,
            -0.079726309,
            0.943101525,
            -0.097263753,
            0.742649972,
            0.199017450,
            -0.083425328,
            1.321230050,
            1.458621860,
            1.501659870,
            1.653805260,
            2.034990310,
        ),
        (
            -0.130215570,
            0.032370113,
            0.112142138,
            0.563722908,
            0.694823802,
            0.270027608,
            0.355698586,
            -0.007662657,
            0.935676396,
            0.007986223,
            0.206536487,
            0.418852895,
            -0.146504328,
            0.468130648,
            -0.081052415,
            0.949351311,
            -0.099624366,
            0.740405500,
            0.199215144,
            -0.088213027,
            1.319523570,
            1.497105480,
            1.497129920,
            1.681157350,
            2.069651600,
        ),
    )
    GRASP_ACT_TARGET = (
        (
            -0.129778390,
            0.033539850,
            0.112443940,
            0.559938130,
            0.694846600,
            0.269048270,
            0.362312760,
            0.041549493,
            0.937220450,
            0.009497468,
            0.243935670,
            0.435892100,
            -0.149102480,
            0.496532140,
            -0.080862840,
            0.948238600,
            -0.100719560,
            0.739101700,
            0.199792370,
            -0.090794380,
            1.319356400,
            1.513577700,
            1.494391700,
            1.695898900,
            2.088478600,
        ),
    )
    SQUEEZE_ACT_TARGET = (
        (
            -0.129778390,
            0.033539850,
            0.112443940,
            0.559938130,
            0.694846600,
            0.269048270,
            0.362312760,
            0.290883930,
            0.905268130,
            -0.008578152,
            0.426117120,
            0.516412260,
            -0.153551520,
            0.632336140,
            -0.086985830,
            0.937324640,
            -0.100719560,
            0.739101700,
            0.199792370,
            -0.090794380,
            1.319356400,
            1.577433100,
            1.474223900,
            1.695898900,
            2.088478600,
        ),
    )

    def __init__(
        self,
        action_shape,
        device,
        env,
        npy_path: str | None = None,
        interp_steps: int = 5,
        qpos_interp_steps: int = 1,
        tendon_interp_steps: int = 1,
        loop: bool = True,
        server=None,
    ) -> None:
        del npy_path, server
        self.device = device
        self.env = env
        self.num_envs, self.act_dim = action_shape
        self.pose_dim = 7
        self.expected_row_dim = self.act_dim

        self.z_offset = 0.1198935956 / 2.0
        self.mocap_to_palm_offset = torch.tensor(
            [
                0.00000000,
                0.00000000,
                0.03400001,
                1.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.interp_steps = int(interp_steps)
        self.pose_interp_steps = 20
        self.qpos_interp_steps = max(1, int(qpos_interp_steps))
        self.tendon_interp_steps = max(1, int(tendon_interp_steps))
        self.loop = bool(loop)

        self.pregrasp_rot_step_limit = 0.03
        self.pregrasp_pos_tol = 0.01
        self.pregrasp_rot_tol = 0.08
        self.pregrasp_hold_steps = 15

        self.replay_end_hold_steps = 20
        self.post_rot_step_limit = 0.04
        self.post_pos_tol = 0.005
        self.post_rot_tol = 0.08
        self.pre_insert_final_pos_tol = 0.001
        self.pre_insert_final_rot_tol = 0.04
        self.insert_path_intermediate_pos_tol = 0.001
        self.insert_final_pos_tol = 0.001
        self.insert_final_rot_tol = 0.04
        self.move_above_arc_height = 0.1
        self.post_hold_steps = 10
        self.lift_height = 0.10
        self.pre_insert_clearance = 0.06
        self.insert_slot_x_offset = -0.003
        self.insert_slot_y_offset = -0.001
        self.release_steps = 30
        self.back_x_offset = 0.05
        self.back_z_offset = 0.05
        self.safe_slot_indices = torch.tensor(
            [
                i
                for i, name in enumerate(self.CENTRIFUGE_50ML_SLOT_SITE_NAMES)
                if "-top" not in name
            ],
            device=self.device,
            dtype=torch.long,
        )
        self.insert_rel_pose = torch.tensor(
            self.INSERT_REL_POSE,
            device=self.device,
            dtype=torch.float32,
        )
        self.grasp_yaw_offset_pose = torch.tensor(
            self.GRASP_YAW_OFFSET_POSE,
            device=self.device,
            dtype=torch.float32,
        )
        self._joint_mask, self._tendon_mask = self._build_action_masks()
        self._prev_action: torch.Tensor | None = None
        self._replay_traj_local = self._build_trajectory()
        self._grasp_action_tail = self._replay_traj_local[-1, 7:].clone()

        self.traj: torch.Tensor | None = None
        self.traj_len: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None
        self.stage: torch.Tensor | None = None
        self._tube_pose_ref_valid: torch.Tensor | None = None
        self._insert_targets_valid: torch.Tensor | None = None

        self._tube_pose_ref_w: torch.Tensor | None = None
        self._tube_to_palm_grasp: torch.Tensor | None = None
        self._target_tube_pose_w: torch.Tensor | None = None
        self._target_tube_pre_insert_w: torch.Tensor | None = None
        self._target_palm_above_w: torch.Tensor | None = None
        self._target_palm_pre_insert_w: torch.Tensor | None = None
        self._target_palm_insert_w: torch.Tensor | None = None
        self._palm_body_id = self._resolve_palm_body_id()

        print(
            f"Loaded built-in replay trajectory: {self._replay_traj_local.shape[0]} steps, "
            f"row_dim={self._replay_traj_local.shape[1]}, "
            f"qpos_interp_steps={self.qpos_interp_steps}, "
            f"tendon_interp_steps={self.tendon_interp_steps}"
        )

    def reset(self) -> None:
        self._prev_action = None
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.stage = None
        self._tube_pose_ref_valid = None
        self._insert_targets_valid = None
        self._tube_pose_ref_w = None
        self._tube_to_palm_grasp = None
        self._target_tube_pose_w = None
        self._target_tube_pre_insert_w = None
        self._target_palm_above_w = None
        self._target_palm_pre_insert_w = None
        self._target_palm_insert_w = None

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
            self.stage[env_ids] = ShadowCentrifugeInsertStage.PREGRASP_ALIGN.value
        if self._tube_pose_ref_valid is not None:
            self._tube_pose_ref_valid[env_ids] = False
        if self._insert_targets_valid is not None:
            self._insert_targets_valid[env_ids] = False

    def __call__(self, obs):
        tube_pose_w = self._extract_tube_pose_w(obs)

        if (
            self._tube_pose_ref_w is None
            or self._tube_pose_ref_w.shape[0] != self.num_envs
        ):
            self._tube_pose_ref_w = tube_pose_w.clone()
            self._tube_pose_ref_valid = torch.ones(
                (self.num_envs,),
                device=self.device,
                dtype=torch.bool,
            )
        else:
            if self._tube_pose_ref_valid is None:
                self._tube_pose_ref_valid = torch.ones(
                    (self.num_envs,),
                    device=self.device,
                    dtype=torch.bool,
                )
            invalid_ref = ~self._tube_pose_ref_valid
            if invalid_ref.any():
                self._tube_pose_ref_w[invalid_ref] = tube_pose_w[invalid_ref]
                self._tube_pose_ref_valid[invalid_ref] = True

        if self.traj is None:
            self.init_buffers(obs)

        need_plan = self.traj_len == 0
        if need_plan.any():
            self._plan_stage_batch(obs, need_plan)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        target_action = self.traj[batch_idx, self.idx]
        action = self._smooth_action(target_action)

        done_mask = self.update_traj_progress(obs, target_action)
        if done_mask.any():
            self._advance_stage(done_mask)
            self._plan_stage_batch(obs, done_mask)

        return action

    def init_buffers(self, obs) -> None:
        del obs
        max_len = max(
            self._replay_traj_local.shape[0] + self.replay_end_hold_steps,
            512,
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
        self.stage = torch.full(
            (self.num_envs,),
            ShadowCentrifugeInsertStage.PREGRASP_ALIGN.value,
            device=self.device,
            dtype=torch.long,
        )

    def _plan_stage_batch(self, obs, plan_mask: torch.Tensor) -> None:
        if not plan_mask.any():
            return

        stage_table = {
            ShadowCentrifugeInsertStage.PREGRASP_ALIGN.value: self._plan_pregrasp_align,
            ShadowCentrifugeInsertStage.REPLAY.value: self._plan_replay,
            ShadowCentrifugeInsertStage.LIFT.value: self._plan_lift,
            ShadowCentrifugeInsertStage.MOVE_ABOVE.value: self._plan_move_above,
            ShadowCentrifugeInsertStage.PRE_INSERT.value: self._plan_pre_insert,
            ShadowCentrifugeInsertStage.INSERT.value: self._plan_insert,
            ShadowCentrifugeInsertStage.RELEASE.value: self._plan_release,
            ShadowCentrifugeInsertStage.BACK.value: self._plan_back,
            ShadowCentrifugeInsertStage.DONE.value: self._plan_done,
            ShadowCentrifugeInsertStage.HOLD.value: self._plan_hold,
        }

        for stage_value, plan_fn in stage_table.items():
            stage_mask = plan_mask & (self.stage == stage_value)
            if not stage_mask.any():
                continue
            idxs = torch.where(stage_mask)[0]
            traj = plan_fn(obs, idxs)
            self.apply_plan(idxs, traj)

    def apply_plan(self, idxs: torch.Tensor, traj: torch.Tensor) -> None:
        if idxs.numel() == 0:
            return

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
            (self.num_envs, new_len, self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )
        if old_traj is not None:
            self.traj[:, : old_traj.shape[1]] = old_traj

    def update_traj_progress(self, obs, target_action: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(self.num_envs, device=self.device)
        last = self.idx == (self.traj_len - 1)
        hold_stage = self.stage == ShadowCentrifugeInsertStage.HOLD.value

        pose_close = self.target_action_is_reached(obs, target_action, last=last)
        advance = pose_close & (~last) & (~hold_stage)
        self.idx = torch.minimum(self.idx + advance.long(), self.traj_len - 1)

        done = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        done = torch.where(~hold_stage, last & pose_close, done)
        done = done & (self.traj_len[batch_idx] > 0)
        return done

    def target_action_is_reached(
        self,
        obs,
        target_action: torch.Tensor,
        last: torch.Tensor,
    ) -> torch.Tensor:
        palm_pose_w = self._get_current_palm_pose_w(obs)
        if palm_pose_w is None:
            palm_pose_w = target_action[:, :7]

        pos_err = torch.linalg.norm(palm_pose_w[:, :3] - target_action[:, :3], dim=-1)
        qa = F.normalize(palm_pose_w[:, 3:7], dim=-1)
        qb = F.normalize(target_action[:, 3:7], dim=-1)
        dot = torch.sum(qa * qb, dim=-1).abs().clamp(-1.0, 1.0)
        rot_err = 2.0 * torch.acos(dot)

        stage_pos_tol = torch.full(
            (self.num_envs,),
            float(self.post_pos_tol),
            device=self.device,
            dtype=torch.float32,
        )
        stage_rot_tol = torch.full(
            (self.num_envs,),
            float(self.post_rot_tol),
            device=self.device,
            dtype=torch.float32,
        )
        pregrasp = self.stage == ShadowCentrifugeInsertStage.PREGRASP_ALIGN.value
        stage_pos_tol = torch.where(
            pregrasp,
            torch.full_like(stage_pos_tol, float(self.pregrasp_pos_tol)),
            stage_pos_tol,
        )
        stage_rot_tol = torch.where(
            pregrasp,
            torch.full_like(stage_rot_tol, float(self.pregrasp_rot_tol)),
            stage_rot_tol,
        )
        pre_insert_final = (
            self.stage == ShadowCentrifugeInsertStage.PRE_INSERT.value
        ) & last
        stage_pos_tol = torch.where(
            pre_insert_final,
            torch.full_like(stage_pos_tol, float(self.pre_insert_final_pos_tol)),
            stage_pos_tol,
        )
        stage_rot_tol = torch.where(
            pre_insert_final,
            torch.full_like(stage_rot_tol, float(self.pre_insert_final_rot_tol)),
            stage_rot_tol,
        )
        insert_final = (
            self.stage == ShadowCentrifugeInsertStage.INSERT.value
        ) & last
        stage_pos_tol = torch.where(
            insert_final,
            torch.full_like(stage_pos_tol, float(self.insert_final_pos_tol)),
            stage_pos_tol,
        )
        stage_rot_tol = torch.where(
            insert_final,
            torch.full_like(stage_rot_tol, float(self.insert_final_rot_tol)),
            stage_rot_tol,
        )
        insert_path_intermediate = (
            self.stage == ShadowCentrifugeInsertStage.INSERT.value
        ) & (~last)
        stage_pos_tol = torch.where(
            insert_path_intermediate,
            torch.full_like(
                stage_pos_tol, float(self.insert_path_intermediate_pos_tol)
            ),
            stage_pos_tol,
        )
        pose_close = (pos_err <= stage_pos_tol) & (rot_err <= stage_rot_tol)
        pos_only_close = pos_err <= stage_pos_tol
        return torch.where(~last, pos_only_close, pose_close)

    def _advance_stage(self, done_mask: torch.Tensor) -> None:
        if not done_mask.any():
            return

        new_stage = self.stage.clone()

        transition_table = {
            ShadowCentrifugeInsertStage.PREGRASP_ALIGN.value: ShadowCentrifugeInsertStage.REPLAY.value,
            ShadowCentrifugeInsertStage.REPLAY.value: ShadowCentrifugeInsertStage.LIFT.value,
            ShadowCentrifugeInsertStage.LIFT.value: ShadowCentrifugeInsertStage.MOVE_ABOVE.value,
            ShadowCentrifugeInsertStage.MOVE_ABOVE.value: ShadowCentrifugeInsertStage.PRE_INSERT.value,
            ShadowCentrifugeInsertStage.PRE_INSERT.value: ShadowCentrifugeInsertStage.INSERT.value,
            ShadowCentrifugeInsertStage.INSERT.value: ShadowCentrifugeInsertStage.RELEASE.value,
            ShadowCentrifugeInsertStage.RELEASE.value: ShadowCentrifugeInsertStage.BACK.value,
            ShadowCentrifugeInsertStage.BACK.value: ShadowCentrifugeInsertStage.HOLD.value,
            ShadowCentrifugeInsertStage.DONE.value: ShadowCentrifugeInsertStage.HOLD.value,
            ShadowCentrifugeInsertStage.HOLD.value: ShadowCentrifugeInsertStage.HOLD.value,
        }

        for src, dst in transition_table.items():
            mask = done_mask & (self.stage == src)
            new_stage[mask] = dst

        self.stage = new_stage
        self.traj_len[done_mask] = 0
        self.idx[done_mask] = 0

    def _plan_pregrasp_align(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        target_pose_w = self._world_replay_traj(idxs)[:, 0, :7]

        current_pose_w = self._get_current_palm_pose_w(obs)
        if current_pose_w is None:
            current_pose_w = target_pose_w.clone()
        current_pose_w = current_pose_w[idxs]

        pose_traj = self._make_step_pose_traj(
            current_pose_w,
            target_pose_w,
            max_rot_step=self.pregrasp_rot_step_limit,
            hold_steps=self.pregrasp_hold_steps,
        )
        action_tail = torch.zeros((idxs.numel(), self.act_dim - 7), device=self.device)
        return self._pack_action_traj(pose_traj, action_tail)

    def _plan_replay(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        del obs
        replay = self._world_replay_traj(idxs)
        start_tail = torch.zeros_like(replay[:, 0, 7:])
        if self._prev_action is not None:
            start_tail = self._prev_action[idxs, 7:]
        bridge_tail = self._interpolate_action_tail(
            start_tail,
            replay[:, 0, 7:],
            self.interp_steps + 1,
        )
        bridge_pose = replay[:, :1, :7].repeat(1, bridge_tail.shape[1], 1)
        replay = torch.cat(
            (torch.cat((bridge_pose, bridge_tail), dim=-1), replay[:, 1:]),
            dim=1,
        )
        if self.loop:
            return replay

        hold = replay[:, -1:, :].repeat(1, self.replay_end_hold_steps, 1)
        return torch.cat((replay, hold), dim=1)

    def _plan_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_pose_w = self._current_or_last_palm_pose(obs, idxs)
        target_pose_w = current_pose_w.clone()
        target_pose_w[:, 2] += self.lift_height
        return self._plan_post_pose_traj(current_pose_w, target_pose_w)

    def _plan_move_above(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        self._ensure_insert_targets(obs)
        current_pose_w = self._current_or_last_palm_pose(obs, idxs)
        tube_pose_w = self._extract_tube_pose_w(obs)[idxs]
        target_slot_pose_w = self._select_centrifuge_insert_slot_pose_w(obs)[idxs]
        target_tube_pos_w = target_slot_pose_w[:, :3].clone()
        target_tube_pos_w[:, 2] += self.pre_insert_clearance

        palm_to_tube = self._compose_pose_wxyz(
            self._invert_pose_wxyz(current_pose_w),
            tube_pose_w,
        )
        target_palm_quat_w = self._palm_down_quat_towards(
            self._get_robot_base_pos_w()[idxs],
            target_slot_pose_w[:, :3],
        )
        target_tube_offset_w = self._quat_rotate_vec_wxyz(
            target_palm_quat_w,
            palm_to_tube[:, :3],
        )

        target_pose_w = current_pose_w.clone()
        target_pose_w[:, :3] = target_tube_pos_w - target_tube_offset_w
        target_pose_w[:, 3:7] = target_palm_quat_w
        self._target_palm_above_w[idxs] = target_pose_w
        arc_height = (
            torch.linalg.norm(
                target_pose_w[:, :2] - current_pose_w[:, :2],
                dim=-1,
                keepdim=True,
            )
            / 5.0
        )
        return self._plan_parabolic_post_pose_traj(
            current_pose_w,
            target_pose_w,
            height=arc_height,
        )

    def _plan_pre_insert(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        self._ensure_insert_targets(obs)
        current_palm_pose_w = self._current_or_last_palm_pose(obs, idxs)
        current_tube_pose_w = self._extract_tube_pose_w(obs)[idxs]
        target_slot_pose_w = self._select_centrifuge_insert_slot_pose_w(obs)[idxs]
        target_tube_quat_w = self._align_local_z_to_target_z_wxyz(
            current_tube_pose_w[:, 3:7],
            target_slot_pose_w[:, 3:7],
        )
        target_tube_pre_insert_w = self._target_tube_pre_insert_w[idxs].clone()
        target_tube_pre_insert_w[:, 3:7] = target_tube_quat_w

        tube_delta_w = self._compose_pose_wxyz(
            target_tube_pre_insert_w,
            self._invert_pose_wxyz(current_tube_pose_w),
        )
        target_palm_pre_insert_w = self._compose_pose_wxyz(
            tube_delta_w, current_palm_pose_w
        )

        target_tube_insert_w = self._target_tube_pose_w[idxs].clone()
        target_tube_insert_w[:, 3:7] = target_tube_quat_w
        tube_insert_delta_w = self._compose_pose_wxyz(
            target_tube_insert_w,
            self._invert_pose_wxyz(target_tube_pre_insert_w),
        )
        target_palm_insert_w = self._compose_pose_wxyz(
            tube_insert_delta_w,
            target_palm_pre_insert_w,
        )

        self._target_tube_pre_insert_w[idxs] = target_tube_pre_insert_w
        self._target_tube_pose_w[idxs] = target_tube_insert_w
        self._target_palm_pre_insert_w[idxs] = target_palm_pre_insert_w
        self._target_palm_insert_w[idxs] = target_palm_insert_w
        return self._plan_post_pose_traj(current_palm_pose_w, target_palm_pre_insert_w)

    def _plan_insert(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        del obs
        if self._target_palm_pre_insert_w is None or self._target_palm_insert_w is None:
            raise RuntimeError(
                "Insert targets must be cached during pre-insert planning."
            )
        return self._plan_post_pose_traj(
            self._target_palm_pre_insert_w[idxs],
            self._target_palm_insert_w[idxs],
        )

    def _plan_release(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_pose_w = self._current_or_last_palm_pose(obs, idxs)
        pose_traj = current_pose_w.unsqueeze(1).repeat(1, self.release_steps, 1)
        start_tail = self._grasp_action_tail.unsqueeze(0).repeat(idxs.numel(), 1)
        target_tail = self._open_action_tail_for_batch(idxs.numel())
        tail_traj = self._interpolate_action_tail(
            start_tail, target_tail, self.release_steps
        )
        return torch.cat((pose_traj, tail_traj), dim=-1)

    def _plan_back(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_pose_w = self._current_or_last_palm_pose(obs, idxs)
        target_pose_w = current_pose_w.clone()
        target_pose_w[:, 0] += self.back_x_offset
        target_pose_w[:, 2] += self.back_z_offset
        return self._plan_open_pose_traj(current_pose_w, target_pose_w)

    def _plan_done(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_pose_w = self._current_or_last_palm_pose(obs, idxs)
        action_tail = self._open_action_tail_for_batch(idxs.numel())
        pose_traj = current_pose_w.unsqueeze(1).repeat(1, self.post_hold_steps, 1)
        return self._pack_action_traj(pose_traj, action_tail)

    def _plan_hold(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_pose_w = self._current_or_last_palm_pose(obs, idxs)
        action_tail = self._open_action_tail_for_batch(idxs.numel())
        return self._pack_action_traj(current_pose_w.unsqueeze(1), action_tail)

    def _world_replay_traj(self, idxs: torch.Tensor) -> torch.Tensor:
        tube_pose_ref_w = self._tube_pose_ref_w
        if tube_pose_ref_w is None:
            raise RuntimeError(
                "tube_pose_ref_w must be initialized before replay planning."
            )

        batch_size = idxs.numel()
        num_steps = self._replay_traj_local.shape[0]
        rows = self._replay_traj_local.unsqueeze(0).repeat(batch_size, 1, 1)

        local_pose = rows[:, :, :7].reshape(batch_size * num_steps, 7)
        tube_pose = (
            tube_pose_ref_w[idxs]
            .unsqueeze(1)
            .repeat(1, num_steps, 1)
            .reshape(batch_size * num_steps, 7)
        )
        world_pose = self._compose_pose_wxyz(tube_pose, local_pose)
        offset_pose = self.mocap_to_palm_offset.unsqueeze(0).repeat(
            batch_size * num_steps, 1
        )
        world_pose = self._compose_pose_wxyz(world_pose, offset_pose)
        world_pose[:, 2] += self.z_offset
        rows[:, :, :7] = world_pose.reshape(batch_size, num_steps, 7)
        return rows

    def _plan_post_pose_traj(
        self,
        start_pose_w: torch.Tensor,
        target_pose_w: torch.Tensor,
    ) -> torch.Tensor:
        pose_traj = self._make_step_pose_traj(
            start_pose_w,
            target_pose_w,
            max_rot_step=self.post_rot_step_limit,
            hold_steps=self.post_hold_steps,
        )
        return self._pack_action_traj(
            pose_traj, self._post_action_tail_for_batch(start_pose_w.shape[0])
        )

    def _plan_open_pose_traj(
        self,
        start_pose_w: torch.Tensor,
        target_pose_w: torch.Tensor,
    ) -> torch.Tensor:
        pose_traj = self._make_step_pose_traj(
            start_pose_w,
            target_pose_w,
            max_rot_step=self.post_rot_step_limit,
            hold_steps=self.post_hold_steps,
        )
        return self._pack_action_traj(
            pose_traj,
            self._open_action_tail_for_batch(start_pose_w.shape[0]),
        )

    def _plan_parabolic_post_pose_traj(
        self,
        start_pose_w: torch.Tensor,
        target_pose_w: torch.Tensor,
        height: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_steps = max(1, self.pose_interp_steps)
        pos_traj, quat_traj = batched_interpolate2(
            start_pos=start_pose_w[:, :3],
            end_pos=target_pose_w[:, :3],
            start_quat=start_pose_w[:, 3:7],
            end_quat=target_pose_w[:, 3:7],
            num_steps=num_steps,
            height=height,
        )
        pose_traj = torch.cat((pos_traj, quat_traj), dim=-1)
        hold = target_pose_w.unsqueeze(1).repeat(1, max(1, self.post_hold_steps), 1)
        pose_traj = torch.cat((pose_traj, hold), dim=1)
        return self._pack_action_traj(
            pose_traj,
            self._post_action_tail_for_batch(start_pose_w.shape[0]),
        )

    def _make_step_pose_traj(
        self,
        start_pose_w: torch.Tensor,
        target_pose_w: torch.Tensor,
        max_rot_step: float,
        hold_steps: int,
    ) -> torch.Tensor:
        qa = F.normalize(start_pose_w[:, 3:7], dim=-1)
        qb = F.normalize(target_pose_w[:, 3:7], dim=-1)
        dot = torch.sum(qa * qb, dim=-1).abs().clamp(-1.0, 1.0)
        max_rot_err = torch.max(2.0 * torch.acos(dot))
        rot_steps = int(torch.ceil(max_rot_err / max_rot_step).item())
        num_steps = max(1, self.pose_interp_steps, rot_steps)
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

    def _post_action_tail(self, idxs: torch.Tensor) -> torch.Tensor:
        return self._grasp_action_tail.unsqueeze(0).repeat(idxs.numel(), 1)

    def _post_action_tail_for_batch(self, batch_size: int) -> torch.Tensor:
        return self._grasp_action_tail.unsqueeze(0).repeat(batch_size, 1)

    def _open_action_tail_for_batch(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.act_dim - 7),
            device=self.device,
            dtype=torch.float32,
        )

    def _current_or_last_palm_pose(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        current_pose_w = self._get_current_palm_pose_w(obs)
        if current_pose_w is not None:
            return current_pose_w[idxs]

        last_pose_w = self.traj[idxs, torch.clamp(self.idx[idxs], min=0), :7]
        return last_pose_w.clone()

    def _build_action_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_mask = torch.zeros(self.act_dim, dtype=torch.bool, device=self.device)
        tendon_mask = torch.zeros(self.act_dim, dtype=torch.bool, device=self.device)
        idx = 0
        action_manager = self.env.action_manager
        for term in action_manager._terms.values():
            dim = term.action_dim
            transmission_type = getattr(term.cfg, "transmission_type", None)
            is_joint = transmission_type == TransmissionType.JOINT
            is_tendon = transmission_type == TransmissionType.TENDON
            if is_joint:
                joint_mask[idx : idx + dim] = True
            elif is_tendon:
                tendon_mask[idx : idx + dim] = True
            idx += dim
        return joint_mask, tendon_mask


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    return ShadowCentrifugeInsertPolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        npy_path=replay_path,
        interp_steps=5,
        qpos_interp_steps=5,
        tendon_interp_steps=5,
        loop=False,
        server=server,
    )
