import torch

from assets.mjcf.robot.ur5e_xhand import XHAND_JOINT_ORDER
from src.utils.policy_helpers.xhand_tube_insert_base import (
    _XHandTubeInsertBase as _XHandTubeInsertBase,
)
from src.utils.policy_helpers.xhand_tube_insert_common import (
    _XHAND_INSERT_GRASP_QPOS as _XHAND_INSERT_GRASP_QPOS,
    _XHAND_INSERT_PALM_REL_POSE as _XHAND_INSERT_PALM_REL_POSE,
    XHandTubeInsertStage as XHandTubeInsertStage,
    sample_rack_tube_pos_from_site as sample_rack_tube_pos_from_site,
    sample_target_pos_from_site as sample_target_pos_from_site,
)
from src.utils.kinematics import batched_interpolate_linear
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_distance,
    quat_mul,
    quat_to_mat,
)


class XHandTubeInsertPolicy(_XHandTubeInsertBase):
    """Insert policy variant for UR5E+XHand.

    The inherited planner still moves the task frame named
    ``pinch_site_pose_w``. In the XHand task that observation is the XHand
    palm site, and the final action is 7D palm IK target plus 12 XHand joints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hand = None
        self.hand_dim = len(XHAND_JOINT_ORDER)
        self.target_tube_pose = None

    def reset(self):
        super().reset()
        self.hand = None
        if self.target_tube_pose is not None:
            self.target_tube_pose.zero_()

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if self.traj is None:
            return
        env_ids = torch.as_tensor(env_ids, device=self.traj.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        self.traj[env_ids] = 0.0
        self.traj_len[env_ids] = 0
        self.idx[env_ids] = 0
        self.stage[env_ids] = XHandTubeInsertStage.APPROACH.value
        if self.target_pos is not None:
            self.target_pos[env_ids] = 0.0
        if self.hand is not None:
            self.hand[env_ids] = self._xhand_open(
                env_ids.numel(),
                self.hand.device,
                self.hand.dtype,
            )
        if self.target_tube_pose is not None:
            self.target_tube_pose[env_ids] = 0.0

    def __call__(self, obs):
        pinch_pose = obs["pinch_site_pose_w"]
        B = pinch_pose.shape[0]
        device = pinch_pose.device
        dtype = pinch_pose.dtype

        if self.traj is None or self.traj.shape[0] != B or self.traj.device != device:
            self.init_buffers(obs)

        need_plan = self.traj_len == 0
        if need_plan.any():
            self._plan_stage_batch(obs, need_plan)

        batch_idx = torch.arange(B, device=device)
        target_pose = self.traj[batch_idx, self.idx]

        if self.server is not None:
            self.frame_handle.position = target_pose[0, :3].detach().cpu().numpy()
            self.frame_handle.wxyz = target_pose[0, 3:].detach().cpu().numpy()

        pos_err = torch.norm(target_pose[:, :3] - pinch_pose[:, :3], dim=-1)
        rot_err = quat_distance(target_pose[:, 3:], pinch_pose[:, 3:])
        last = self.idx == (self.traj_len - 1)

        pos_tol_nonlast = torch.full(
            (B,), float(self.pos_threshold), device=device, dtype=dtype
        )
        pos_tol_nonlast = torch.where(
            self.stage == XHandTubeInsertStage.INSERT.value,
            torch.full((B,), 0.01, device=device, dtype=dtype),
            pos_tol_nonlast,
        )
        advance = (pos_err < pos_tol_nonlast) & (~last)
        self.idx = self.idx + advance.long()

        pos_tol_last = torch.full(
            (B,), float(self.pos_threshold), device=device, dtype=dtype
        )
        # The new real-robot reset configuration leaves a small steady-state
        # Cartesian error at the top of the lift.  Keep all grasp/insert
        # tolerances unchanged and only allow that lift waypoint to advance.
        pos_tol_last = torch.where(
            self.stage == XHandTubeInsertStage.LIFT.value,
            torch.full((B,), 0.006, device=device, dtype=dtype),
            pos_tol_last,
        )
        pos_ok_last = pos_err < pos_tol_last
        rot_ok_last = rot_err < float(self.rot_threshold)
        done_ok = pos_ok_last & rot_ok_last
        done_ok = torch.where(
            self.stage == XHandTubeInsertStage.LIFT.value, pos_ok_last, done_ok
        )
        done_traj = self._stage_done_mask(obs, last & done_ok)

        if done_traj.any():
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
            self._plan_stage_batch(obs, done_traj)

        return torch.cat([target_pose, self.hand.to(dtype=target_pose.dtype)], dim=-1)

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
        self.hand = self._xhand_open(B, device, dtype)
        self.target_pos = torch.zeros((B, 3), device=device, dtype=dtype)
        self.target_tube_pose = torch.zeros((B, 7), device=device, dtype=dtype)

        self._plan_stage_batch(obs, torch.ones((B,), device=device, dtype=torch.bool))

    def apply_plan(
        self, idxs: torch.Tensor, traj: torch.Tensor, hand_value: torch.Tensor
    ):
        if idxs.numel() == 0:
            return
        L = traj.shape[1]
        self.traj[idxs] = 0.0
        self.traj[idxs, :L] = traj
        self.traj_len[idxs] = L
        self.idx[idxs] = 0
        self.hand[idxs] = hand_value.to(device=self.hand.device, dtype=self.hand.dtype)

    def _plan_stage_batch(self, obs, plan_mask: torch.Tensor):
        if not plan_mask.any():
            return

        B = self.stage.shape[0]
        device = self.traj.device
        dtype = self.traj.dtype
        open_hand = self._xhand_open(B, device, dtype)
        grasp_hand = self._xhand_grasp(B, device, dtype)

        stage_table = {
            XHandTubeInsertStage.APPROACH.value: (self._plan_approach, open_hand),
            XHandTubeInsertStage.ALIGN.value: (self._plan_align, open_hand),
            XHandTubeInsertStage.GRASP.value: (self._plan_grasp, grasp_hand),
            XHandTubeInsertStage.LIFT.value: (self._plan_lift, grasp_hand),
            XHandTubeInsertStage.MOVE.value: (self._plan_move, grasp_hand),
            XHandTubeInsertStage.PRE_INSERT.value: (
                self._plan_pre_insert,
                grasp_hand,
            ),
            XHandTubeInsertStage.INSERT.value: (self._plan_insert, grasp_hand),
            XHandTubeInsertStage.INSERT_HOLD.value: (
                self._plan_insert_hold,
                grasp_hand,
            ),
            XHandTubeInsertStage.RELEASE.value: (self._plan_release, open_hand),
            XHandTubeInsertStage.DONE.value: (self._plan_done, open_hand),
            XHandTubeInsertStage.HOLD.value: (self._plan_hold, open_hand),
        }

        for stage_val, (plan_fn, hand_target) in stage_table.items():
            m = plan_mask & (self.stage == stage_val)
            if not m.any():
                continue
            idxs = torch.where(m)[0]
            traj = plan_fn(obs, idxs)
            self.apply_plan(idxs, traj, hand_target[idxs])

    def _xhand_open(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros((batch_size, self.hand_dim), device=device, dtype=dtype)

    def _xhand_grasp(self, batch_size: int, device, dtype) -> torch.Tensor:
        hand = self._xhand_open(batch_size, device, dtype)
        for joint_id, joint_expr in enumerate(XHAND_JOINT_ORDER):
            for joint_suffix, qpos in _XHAND_INSERT_GRASP_QPOS.items():
                if joint_suffix in joint_expr:
                    hand[:, joint_id] = qpos
                    break
        return hand

    def _xhand_rel_pose(self, batch_size: int, device, dtype) -> torch.Tensor:
        return (
            torch.tensor(
                _XHAND_INSERT_PALM_REL_POSE,
                device=device,
                dtype=dtype,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

    def _tube_pose_to_palm_pose(self, tube_pose: torch.Tensor) -> torch.Tensor:
        rel_pose = self._xhand_rel_pose(
            tube_pose.shape[0],
            tube_pose.device,
            tube_pose.dtype,
        )
        return compose_pose_wxyz(tube_pose, rel_pose)

    def _upright_tube_quat(self, batch_size: int, device, dtype) -> torch.Tensor:
        quat = torch.zeros((batch_size, 4), device=device, dtype=dtype)
        quat[:, 0] = 1.0
        return quat

    def _normalize_vec(self, vec: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-8)
        return vec / norm

    def _quat_between_unit_vecs(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        cross = torch.cross(source, target, dim=-1)
        dot = (source * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        quat = torch.cat([1.0 + dot, cross], dim=-1)
        quat_norm = torch.linalg.norm(quat, dim=-1, keepdim=True)

        fallback_axis = torch.zeros_like(source)
        fallback_axis[:, 0] = 1.0
        fallback_axis = self._normalize_vec(torch.cross(source, fallback_axis, dim=-1))
        fallback = torch.cat([torch.zeros_like(dot), fallback_axis], dim=-1)

        return torch.where(quat_norm < 1e-6, fallback, quat / quat_norm.clamp_min(1e-8))

    def _upright_tube_pose_preserving_twist(
        self,
        current_tube_pose: torch.Tensor,
        target_tube_pose: torch.Tensor,
    ) -> torch.Tensor:
        current_quat = current_tube_pose[:, 3:]
        current_axis = quat_to_mat(current_quat)[:, :, 2]
        target_axis = torch.zeros_like(current_axis)
        target_axis[:, 2] = 1.0
        swing = self._quat_between_unit_vecs(
            self._normalize_vec(current_axis),
            target_axis,
        )

        target_pose = target_tube_pose.clone()
        target_pose[:, 3:] = self._normalize_vec(quat_mul(swing, current_quat))
        return target_pose

    def _retarget_current_tube_to_palm_pose(
        self,
        current_tube_pose: torch.Tensor,
        current_palm_pose: torch.Tensor,
        target_tube_pose: torch.Tensor,
    ) -> torch.Tensor:
        tube_delta = compose_pose_wxyz(
            target_tube_pose, pose_inv_wxyz(current_tube_pose)
        )
        return compose_pose_wxyz(tube_delta, current_palm_pose)

    def _compute_grasp_pose(self, cap_pose: torch.Tensor, body_pose: torch.Tensor):
        del cap_pose
        grasp_pose = self._tube_pose_to_palm_pose(body_pose)
        return grasp_pose[:, :3], grasp_pose[:, 3:]

    def _plan_move(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]
        tube_pose = obs["tube_body_pose_w"][idxs]

        if "cryovial_pose_w" in obs:
            target_tube_pose = obs["cryovial_pose_w"][idxs].clone()
            target_tube_pos = target_tube_pose[:, :3]
        elif "target_tube_pose_w" in obs:
            target_tube_pose = obs["target_tube_pose_w"][idxs].clone()
            target_tube_pos = target_tube_pose[:, :3]
        else:
            slot_pose = obs["slot1_site_pose_w"][idxs]
            target_tube_pos = sample_rack_tube_pos_from_site(slot_pose)
            target_tube_quat = self._upright_tube_quat(
                idxs.numel(),
                target_tube_pos.device,
                target_tube_pos.dtype,
            )
            target_tube_pose = torch.cat([target_tube_pos, target_tube_quat], dim=-1)
        target_tube_pose = self._upright_tube_pose_preserving_twist(
            tube_pose, target_tube_pose
        )
        target_tube_pre_pose = target_tube_pose.clone()
        target_tube_pre_pose[:, 2] += 0.10

        if self.target_pos is None or self.target_tube_pose is None:
            raise RuntimeError("XHand insert target buffers are not initialized")
        self.target_pos[idxs] = target_tube_pos
        self.target_tube_pose[idxs] = target_tube_pose

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        end_palm_pose = self._tube_pose_to_palm_pose(target_tube_pre_pose)
        end_pos = end_palm_pose[:, :3]
        end_quat = end_palm_pose[:, 3:]
        end_pos[:, 2] = start_pos[:, 2]

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos,
            end_pos,
            start_quat,
            end_quat,
            self.traj_steps,
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_pre_insert(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]
        tube_pose = obs["tube_body_pose_w"][idxs]

        if self.target_tube_pose is None:
            raise RuntimeError(
                "target_tube_pose is not initialized; MOVE must run before PRE_INSERT"
            )
        target_tube_pre_pose = self._upright_tube_pose_preserving_twist(
            tube_pose,
            self.target_tube_pose[idxs],
        )
        target_tube_pre_pose[:, 2] += 0.10
        target_palm_pose = self._retarget_current_tube_to_palm_pose(
            tube_pose,
            pinch_pose,
            target_tube_pre_pose,
        )

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        align_pos = start_pos.clone()
        align_pos[:, 0:2] = target_palm_pose[:, 0:2]

        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos,
            align_pos,
            start_quat,
            target_palm_pose[:, 3:],
            num_steps=5,
        )

        return torch.cat([pos_traj, quat_traj], dim=-1)

    def _plan_insert(self, obs, idxs: torch.Tensor):
        pinch_pose = obs["pinch_site_pose_w"][idxs]
        tube_pose = obs["tube_body_pose_w"][idxs]

        if self.target_tube_pose is None:
            raise RuntimeError(
                "target_tube_pose is not initialized; MOVE must run before INSERT"
            )
        target_tube_pose = self._upright_tube_pose_preserving_twist(
            tube_pose,
            self.target_tube_pose[idxs],
        )
        target_palm_pose = self._retarget_current_tube_to_palm_pose(
            tube_pose,
            pinch_pose,
            target_tube_pose,
        )

        start_pos = pinch_pose[:, :3]
        start_quat = pinch_pose[:, 3:]

        if self.traj_steps < 2:
            pos_traj, quat_traj = batched_interpolate_linear(
                start_pos,
                target_palm_pose[:, :3],
                start_quat,
                target_palm_pose[:, 3:],
                self.traj_steps,
            )
            return torch.cat([pos_traj, quat_traj], dim=-1)

        align_steps = min(5, self.traj_steps - 1)
        descend_steps = self.traj_steps - align_steps
        align_pose = target_palm_pose.clone()
        align_pose[:, 2] = start_pos[:, 2]

        align_pos_traj, align_quat_traj = batched_interpolate_linear(
            start_pos,
            align_pose[:, :3],
            start_quat,
            align_pose[:, 3:],
            align_steps,
        )
        descend_pos_traj, descend_quat_traj = batched_interpolate_linear(
            align_pose[:, :3],
            target_palm_pose[:, :3],
            align_pose[:, 3:],
            target_palm_pose[:, 3:],
            descend_steps,
        )

        return torch.cat(
            [
                torch.cat([align_pos_traj, align_quat_traj], dim=-1),
                torch.cat([descend_pos_traj[:, 1:], descend_quat_traj[:, 1:]], dim=-1),
            ],
            dim=1,
        )


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del action_shape, env, replay_path
    return XHandTubeInsertPolicy(
        traj_steps=40,
        device=device,
        pos_threshold=0.005,
        rot_threshold=0.05,
        server=server,
    )
