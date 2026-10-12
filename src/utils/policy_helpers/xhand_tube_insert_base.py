"""Shared staged insertion planner used by the XHand policy."""

from __future__ import annotations

import torch
import viser

from src.utils.policy_helpers.xhand_tube_insert_common import (
    XHandTubeInsertStage,
    sample_target_pos_from_site,
)
from src.utils.kinematics import batched_interpolate_linear
from src.utils.quat_util import (
    axis_angle_to_quat,
    quat_distance,
    quat_mul,
)


class _XHandTubeInsertBase:
    insert_success_xy_threshold = 0.025
    insert_success_z_threshold = 0.002

    def __init__(
        self,
        traj_steps=20,
        height=None,
        device=None,
        pos_threshold=0.01,
        rot_threshold=0.1,
        server: viser.ViserServer = None,
    ):
        self.device = device
        self.traj_steps = traj_steps
        self.height = height

        self.pos_threshold = pos_threshold
        self.rot_threshold = rot_threshold

        # Per-env buffers (initialized on first call)
        self.traj = None  # (B, T_max, 7)
        self.traj_len = None  # (B,)
        self.idx = None  # (B,)
        self.stage = None  # (B,) long, stores XHandTubeInsertStage.value
        self.gripper = None  # (B,) float

        # Cached per-env target (sampled only in MOVE)
        self.target_pos = None  # (B,3)

        self.server = server
        if server is not None:
            self.frame_handle = server.scene.add_frame(
                name="target_pose",
                axes_length=0.05,
                axes_radius=0.01,
                origin_radius=0.02,
            )

    def __call__(self, obs):
        pinch_pose = obs["pinch_site_pose_w"]
        B = pinch_pose.shape[0]
        device = pinch_pose.device
        dtype = pinch_pose.dtype

        # Lazy init / handle num_envs change
        if self.traj is None or self.traj.shape[0] != B or self.traj.device != device:
            self.init_buffers(obs)

        # If any env has no valid plan yet, plan it
        need_plan = self.traj_len == 0
        if need_plan.any():
            self._plan_stage_batch(obs, need_plan)

        batch_idx = torch.arange(B, device=device)

        # (B,7)
        target_pose = self.traj[batch_idx, self.idx]

        # Visualize env0 target
        if self.server is not None:
            self.frame_handle.position = target_pose[0, :3].detach().cpu().numpy()
            self.frame_handle.wxyz = target_pose[0, 3:].detach().cpu().numpy()

        pos_err = torch.norm(target_pose[:, :3] - pinch_pose[:, :3], dim=-1)
        rot_err = quat_distance(target_pose[:, 3:], pinch_pose[:, 3:])

        last = self.idx == (self.traj_len - 1)

        # -----------------------
        # idx advance (per-env)
        # -----------------------
        pos_tol_nonlast = torch.full(
            (B,), float(self.pos_threshold), device=device, dtype=dtype
        )
        pos_tol_nonlast = torch.where(
            self.stage == XHandTubeInsertStage.INSERT.value,
            torch.full((B,), 0.01, device=device, dtype=dtype),
            pos_tol_nonlast,
        )
        pos_ok_nonlast = pos_err < pos_tol_nonlast

        advance = pos_ok_nonlast & (~last)
        self.idx = self.idx + advance.long()

        # -----------------------
        # stage complete (per-env)
        # -----------------------
        pos_ok_last = pos_err < float(self.pos_threshold)
        rot_ok_last = rot_err < float(self.rot_threshold)

        done_ok = pos_ok_last & rot_ok_last
        done_ok = torch.where(
            self.stage == XHandTubeInsertStage.LIFT.value, pos_ok_last, done_ok
        )

        done_traj = self._stage_done_mask(obs, last & done_ok)

        if done_traj.any():
            # Advance stage per env
            new_stage = self.stage.clone()

            m = done_traj & (self.stage == XHandTubeInsertStage.APPROACH.value)
            new_stage[m] = XHandTubeInsertStage.ALIGN.value

            m = done_traj & (self.stage == XHandTubeInsertStage.ALIGN.value)
            new_stage[m] = XHandTubeInsertStage.GRASP.value

            m = done_traj & (self.stage == XHandTubeInsertStage.GRASP.value)
            new_stage[m] = XHandTubeInsertStage.LIFT.value

            m = done_traj & (self.stage == XHandTubeInsertStage.LIFT.value)
            new_stage[m] = XHandTubeInsertStage.MOVE.value

            m = done_traj & (self.stage == XHandTubeInsertStage.MOVE.value)
            new_stage[m] = XHandTubeInsertStage.PRE_INSERT.value

            m = done_traj & (self.stage == XHandTubeInsertStage.PRE_INSERT.value)
            new_stage[m] = XHandTubeInsertStage.INSERT.value

            m = done_traj & (self.stage == XHandTubeInsertStage.INSERT.value)
            new_stage[m] = XHandTubeInsertStage.INSERT_HOLD.value

            m = done_traj & (self.stage == XHandTubeInsertStage.INSERT_HOLD.value)
            new_stage[m] = XHandTubeInsertStage.RELEASE.value

            m = done_traj & (self.stage == XHandTubeInsertStage.RELEASE.value)
            new_stage[m] = XHandTubeInsertStage.DONE.value

            m = done_traj & (self.stage == XHandTubeInsertStage.DONE.value)
            new_stage[m] = XHandTubeInsertStage.HOLD.value

            self.stage = new_stage

            # Re-plan only finished envs
            self._plan_stage_batch(obs, done_traj)

        action = torch.cat([target_pose, self.gripper[:, None]], dim=-1)
        return action

    def reset(self):
        # Clear buffers so next call re-initializes
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.stage = None
        self.gripper = None
        if self.target_pos is not None:
            self.target_pos.zero_()
        print("Policy reset to initial state.")

    def _tube_inserted_mask(self, obs) -> torch.Tensor | None:
        if "tube_body_pose_w" not in obs or "target_tube_pose_w" not in obs:
            return None

        tube_pos = obs["tube_body_pose_w"][:, :3]
        target_pos = obs["target_tube_pose_w"][:, :3]
        tube_xy_dist = torch.linalg.norm(tube_pos[:, :2] - target_pos[:, :2], dim=-1)
        tube_z_dist = torch.abs(tube_pos[:, 2] - target_pos[:, 2])
        return (tube_xy_dist <= self.insert_success_xy_threshold) & (
            tube_z_dist <= self.insert_success_z_threshold
        )

    def _stage_done_mask(self, obs, done_traj: torch.Tensor) -> torch.Tensor:
        insert_stage = self.stage == XHandTubeInsertStage.INSERT.value
        if not insert_stage.any():
            return done_traj

        tube_inserted = self._tube_inserted_mask(obs)
        if tube_inserted is None:
            return done_traj
        return torch.where(insert_stage, tube_inserted, done_traj)

    # =========================
    # batched planning
    # =========================

    def init_buffers(self, obs):
        pinch_pose = obs["pinch_site_pose_w"]
        B = pinch_pose.shape[0]
        device = pinch_pose.device
        dtype = pinch_pose.dtype

        T_max = self.traj_steps + 1

        self.traj = torch.zeros((B, T_max, 7), device=device, dtype=dtype)
        self.traj_len = torch.zeros((B,), device=device, dtype=torch.long)
        self.idx = torch.zeros((B,), device=device, dtype=torch.long)
        self.stage = torch.full(
            (B,),
            XHandTubeInsertStage.APPROACH.value,
            device=device,
            dtype=torch.long,
        )
        self.gripper = torch.full((B,), -1.0, device=device, dtype=dtype)

        self.target_pos = torch.zeros((B, 3), device=device, dtype=dtype)

        self._plan_stage_batch(obs, torch.ones((B,), device=device, dtype=torch.bool))

    def apply_plan(self, idxs: torch.Tensor, traj: torch.Tensor, gripper_value: float):
        if idxs.numel() == 0:
            return
        L = traj.shape[1]
        self.traj[idxs] = 0.0
        self.traj[idxs, :L] = traj
        self.traj_len[idxs] = L
        self.idx[idxs] = 0
        self.gripper[idxs] = gripper_value

    def _plan_stage_batch(self, obs, plan_mask: torch.Tensor):
        if not plan_mask.any():
            return

        # -----------------------------
        # XHandTubeInsertStage → (planner_fn, gripper)
        # -----------------------------
        stage_table = {
            XHandTubeInsertStage.APPROACH.value: (self._plan_approach, -1.0),
            XHandTubeInsertStage.ALIGN.value: (self._plan_align, -1.0),
            XHandTubeInsertStage.GRASP.value: (self._plan_grasp, 1.0),
            XHandTubeInsertStage.LIFT.value: (self._plan_lift, 1.0),
            XHandTubeInsertStage.MOVE.value: (self._plan_move, 1.0),
            XHandTubeInsertStage.PRE_INSERT.value: (self._plan_pre_insert, 1.0),
            XHandTubeInsertStage.INSERT.value: (self._plan_insert, 1.0),
            XHandTubeInsertStage.INSERT_HOLD.value: (self._plan_insert_hold, 1.0),
            XHandTubeInsertStage.RELEASE.value: (self._plan_release, -1.0),
            XHandTubeInsertStage.DONE.value: (self._plan_done, -1.0),
            XHandTubeInsertStage.HOLD.value: (self._plan_hold, -1.0),
        }

        # -----------------------------
        # Loop over all stages
        # -----------------------------
        for stage_val, (plan_fn, grip) in stage_table.items():
            m = plan_mask & (self.stage == stage_val)
            if not m.any():
                continue

            idxs = torch.where(m)[0]
            traj = plan_fn(obs, idxs)
            self.apply_plan(idxs, traj, grip)

    # =========================
    # stage implementations (subset)
    # =========================

    def _compute_grasp_pose(self, cap_pose: torch.Tensor, body_pose: torch.Tensor):
        cap_pos = cap_pose[:, :3]
        body_quat = body_pose[:, 3:]

        B = cap_pos.shape[0]
        device = cap_pos.device
        dtype = cap_pos.dtype

        bias = torch.tensor([0.0, 0.0, -0.005], device=device, dtype=dtype).expand(B, 3)
        pos = cap_pos + bias

        # Rx(pi)
        qx = axis_angle_to_quat(
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            torch.tensor(torch.pi, device=device, dtype=dtype),
            B,
        )

        # Rz(pi)
        qz = axis_angle_to_quat(
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
            torch.tensor(torch.pi, device=device, dtype=dtype),
            B,
        )

        quat = quat_mul(body_quat, qx)
        quat = quat_mul(quat, qz)

        return pos, quat

    def _plan_approach(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]
        cap_pose = obs["tube_cap_pose_w"][idxs]
        body_pose = obs["tube_body_pose_w"][idxs]

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        grasp_pos, grasp_quat = self._compute_grasp_pose(cap_pose, body_pose)

        pre_pos = grasp_pos + torch.tensor(
            [0.0, 0.0, 0.04], device=start_pos.device, dtype=start_pos.dtype
        )

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos, pre_pos, start_quat, grasp_quat, self.traj_steps
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_align(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]
        cap_pose = obs["tube_cap_pose_w"][idxs]
        body_pose = obs["tube_body_pose_w"][idxs]

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        grasp_pos, grasp_quat = self._compute_grasp_pose(cap_pose, body_pose)

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos,
            grasp_pos,
            start_quat,
            grasp_quat,
            num_steps=5,
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_grasp(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]
        cap_pose = obs["tube_cap_pose_w"][idxs]
        body_pose = obs["tube_body_pose_w"][idxs]

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        grasp_pos, grasp_quat = self._compute_grasp_pose(cap_pose, body_pose)

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos, grasp_pos, start_quat, grasp_quat, self.traj_steps
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_lift(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        end_pos = start_pos + torch.tensor(
            [0.0, 0.0, 0.1], device=start_pos.device, dtype=start_pos.dtype
        )

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos, end_pos, start_quat, start_quat, self.traj_steps
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_move(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]

        if "cryovial_pose_w" in obs:
            target_pos = obs["cryovial_pose_w"][idxs, :3].clone()
        elif "target_tube_pose_w" in obs:
            target_pos = obs["target_tube_pose_w"][idxs, :3].clone()
        else:
            slot_pose = obs["slot1_site_pose_w"][idxs]
            target_pos = sample_target_pos_from_site(slot_pose)

        # cache target for this env (reused by PRE_INSERT and INSERT)
        if self.target_pos is None:
            raise RuntimeError("target_pos buffer is not initialized")
        self.target_pos[idxs] = target_pos

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        end_pos = target_pos.clone()
        end_pos[:, 2] = start_pos[:, 2]

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos, end_pos, start_quat, start_quat, self.traj_steps
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_pre_insert(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]

        if self.target_pos is None:
            raise RuntimeError(
                "target_pos is not initialized; MOVE must run before PRE_INSERT"
            )
        target_pos = self.target_pos[idxs]

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        # 👇 Align only XY, keeping the height unchanged
        align_pos = start_pos.clone()
        align_pos[:, 0:2] = target_pos[:, 0:2]

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos,
            align_pos,
            start_quat,
            start_quat,
            num_steps=5,
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_insert(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]
        tube_pose = obs["tube_body_pose_w"][idxs]

        if self.target_pos is None:
            raise RuntimeError(
                "target_pos is not initialized; MOVE must run before INSERT"
            )
        target_pos = self.target_pos[idxs]

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        tube_delta = target_pos - tube_pose[:, :3]
        end_pos = start_pos + tube_delta

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos, end_pos, start_quat, start_quat, self.traj_steps
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_release(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]

        pos = pinch_pose[:, :3]
        quat = pinch_pose[:, 3:]

        return torch.cat(
            [
                pos.unsqueeze(1).repeat(1, self.traj_steps, 1),
                quat.unsqueeze(1).repeat(1, self.traj_steps, 1),
            ],
            dim=-1,
        )

    def _plan_insert_hold(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]

        pos = pinch_pose[:, :3]
        quat = pinch_pose[:, 3:]
        hold_steps = getattr(self, "insert_hold_steps", 10)

        return torch.cat(
            [
                pos.unsqueeze(1).repeat(1, hold_steps, 1),
                quat.unsqueeze(1).repeat(1, hold_steps, 1),
            ],
            dim=-1,
        )

    def _plan_done(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        end_pos = start_pos + torch.tensor(
            [0.0, 0.0, 0.1], device=start_pos.device, dtype=start_pos.dtype
        )

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos, end_pos, start_quat, start_quat, self.traj_steps
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_hold(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]

        pos = pinch_pose[:, :3]
        quat = pinch_pose[:, 3:]

        return torch.cat([pos.unsqueeze(1), quat.unsqueeze(1)], dim=-1)
