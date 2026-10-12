from __future__ import annotations

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import viser

from assets.mjcf.robot.ur5e_xhand import XHAND_JOINT_ORDER
from src.utils.policy_helpers.xhand_thermalcycler_close_common import (
    XHandThermalCyclerMode as XHandThermalCyclerMode,
)
from src.utils.policy_helpers.xhand_thermalcycler_close_plans import (
    ThermalCyclerMotionPlans,
)
from src.utils.kinematics import batched_interpolate_linear, qpos_linear_interpolate
from src.utils.observations import observation_tensor
from src.utils.quat_util import (
    compose_pose_wxyz,
)


class XHandThermalCyclerClosePolicy(ThermalCyclerMotionPlans):
    """Scripted UR5e+XHand thermal-cycler policy."""

    # Keep this False to let the thermal cycler move from MuJoCo contacts.
    # The old kinematic/demo behavior can be restored by setting it True.
    _DRIVE_CYCLER_JOINTS = False
    _DISABLE_CONTACTS = False

    # Measured with tmp/913_thermal_cycler_xhand_vis.py on 2026-06-02.
    # These are palm/right_hand_link poses expressed in the lever-site frame.
    _CLOSE_START_GRASP_POS = (-0.008569, -0.028471, 0.092388)
    _CLOSE_START_GRASP_QUAT = (-0.230179, 0.676555, -0.659345, -0.233568)
    # Measured with tmp/913_thermal_cycler_xhand_vis.py on 2026-06-04.
    _CLOSE_END_GRASP_POS = (0.007506, 0.101702, -0.047904)
    _CLOSE_END_GRASP_QUAT = (-0.415821, -0.548494, 0.604477, -0.401067)
    # Measured for the lid-force-knob spinner pre-open pose on 2026-06-10.
    _SPINNER_GRASP_LEVER_POS = (0.021301, -0.087130, -0.152465)
    _SPINNER_GRASP_LEVER_QUAT = (0.628702, 0.331797, -0.394554, 0.582213)

    _APPROACH_CLEARANCE_Z = 0.16
    _TARGET_CLEARANCE_Z = 0.12
    _SPINNER_CLEARANCE_Z = 0.12
    _SPINNER_UNLOCKED_QPOS = 0.0
    _SPINNER_LOCKED_QPOS = 1.5707963267948966
    _OPEN_SPINNER_STEPS = 90
    _OPEN_POST_SPINNER_LIFT_Z = 0.07
    _OPEN_PREINSERT_LOWER_Z = 0.06
    _OPEN_INSERT_READY_Z = 0.045
    _CLOSE_GRASP_BACKOFF_Y = 0.010
    _CLOSE_LID_STEPS = 150
    _CLOSE_LID_SETTLE_STEPS = 1
    _CLOSE_LATCH_STEPS = 55

    _CLOSE_START_GRIP_QPOS = {
        "thumb_bend_joint": 0.0,
        "thumb_rota_joint1": 0.845,
        "thumb_rota_joint2": 0.821,
        "index_bend_joint": 0.0,
        "index_joint1": 1.82,
        "index_joint2": 1.920,
        "mid_joint1": 1.82,
        "mid_joint2": 1.920,
        "ring_joint1": 1.82,
        "ring_joint2": 1.920,
        "pinky_joint1": 1.82,
        "pinky_joint2": 1.920,
    }
    _CLOSE_END_GRIP_QPOS = {
        "thumb_bend_joint": 0.0,
        "thumb_rota_joint1": 0.845,
        "thumb_rota_joint2": 0.784,
        "index_bend_joint": 0.0,
        "index_joint1": 1.526,
        "index_joint2": 0.500,
        "mid_joint1": 1.526,
        "mid_joint2": 0.500,
        "ring_joint1": 1.526,
        "ring_joint2": 0.500,
        "pinky_joint1": 1.526,
        "pinky_joint2": 0.500,
    }
    _OPEN_INSERT_GRIP_QPOS = {
        **_CLOSE_END_GRIP_QPOS,
        "index_joint2": 0.350,
        "mid_joint2": 0.350,
        "ring_joint2": 0.350,
        "pinky_joint2": 0.350,
    }
    _OPEN_START_GRIP_QPOS = {
        "thumb_bend_joint": 0.0,
        "thumb_rota_joint1": 0.845,
        "thumb_rota_joint2": 0.821,
        "index_bend_joint": 0.0,
        "index_joint1": 1.565,
        "index_joint2": 1.920,
        "mid_joint1": 1.565,
        "mid_joint2": 1.920,
        "ring_joint1": 1.565,
        "ring_joint2": 1.920,
        "pinky_joint1": 1.565,
        "pinky_joint2": 1.920,
    }
    _SPINNER_GRIP_QPOS = {
        "thumb_bend_joint": 1.342,
        "thumb_rota_joint1": -0.170,
        "thumb_rota_joint2": 1.830,
        "index_bend_joint": 0.0,
        "index_joint1": 1.545,
        "index_joint2": 0.958,
        "mid_joint1": 1.545,
        "mid_joint2": 0.948,
        "ring_joint1": 1.920,
        "ring_joint2": 1.920,
        "pinky_joint1": 1.920,
        "pinky_joint2": 1.920,
    }

    def __init__(
        self,
        action_shape,
        device,
        env,
        server: viser.ViserServer | None = None,
        loop: bool = False,
    ) -> None:
        self.device = device
        self.env = env
        self.server = server
        self.loop = bool(loop)
        self.num_envs, self.act_dim = action_shape
        self.pose_dim = 7
        self.hand_dim = len(XHAND_JOINT_ORDER)
        self.expected_action_dim = self.pose_dim + self.hand_dim
        if self.act_dim != self.expected_action_dim:
            raise ValueError(
                f"Expected action_dim={self.expected_action_dim} "
                f"(7D palm IK + {self.hand_dim} XHand qpos), got {self.act_dim}."
            )

        self.traj: torch.Tensor | None = None
        self.cycler_traj: torch.Tensor | None = None
        self.lever_follow_modes: list[list[str | None] | None] | None = None
        self.lever_follow_local_traj: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None
        self.traj_len: torch.Tensor | None = None
        self.mode: XHandThermalCyclerMode | None = None
        self.mode_by_env: list[XHandThermalCyclerMode | None] | None = None
        self._active_plan_env_ids: torch.Tensor | None = None
        self._cycler_joint_ids: torch.Tensor | None = None
        self._lever_site_id: int | None = None
        self._knob_site_id: int | None = None
        self._fk_data = mujoco.MjData(self.env.sim.mj_model)
        self.target_frame = None
        if self._DISABLE_CONTACTS:
            self._disable_contacts()
        if self.server is not None:
            self.target_frame = self.server.scene.add_frame(
                name="xhand_thermalcycler/target_palm",
                axes_length=0.08,
                axes_radius=0.005,
                origin_radius=0.01,
            )

        print(
            "Loaded thermal cycler XHand policy: "
            f"action_dim={self.act_dim}, loop={self.loop}, "
            f"drive_cycler_joints={self._DRIVE_CYCLER_JOINTS}, "
            f"disable_contacts={self._DISABLE_CONTACTS}"
        )

    def _disable_contacts(self) -> None:
        mj_model = self.env.sim.mj_model
        mj_model.geom_contype[:] = 0
        mj_model.geom_conaffinity[:] = 0
        mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        print("Disabled all MuJoCo contacts for thermal cycler XHand visual playback.")

    def reset(self) -> None:
        self.traj = None
        self.cycler_traj = None
        self.lever_follow_modes = None
        self.lever_follow_local_traj = None
        self.idx = None
        self.traj_len = None
        self.mode = None
        self.mode_by_env = None

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if self.traj is not None:
            self.traj[env_ids] = 0.0
        if self.cycler_traj is not None:
            self.cycler_traj[env_ids] = 0.0
        if self.lever_follow_local_traj is not None:
            self.lever_follow_local_traj[env_ids] = float("nan")
        if self.idx is not None:
            self.idx[env_ids] = 0
        if self.traj_len is not None:
            self.traj_len[env_ids] = 0
        if self.lever_follow_modes is not None:
            for env_id in env_ids.detach().cpu().tolist():
                self.lever_follow_modes[int(env_id)] = None
        if self.mode_by_env is not None:
            for env_id in env_ids.detach().cpu().tolist():
                self.mode_by_env[int(env_id)] = None

    def __call__(self, obs):
        if self.traj is None or self.idx is None or self.traj_len is None:
            self.init_buffers()

        need_plan = self.traj_len == 0
        if need_plan.any():
            self._plan(obs, torch.where(need_plan)[0])

        batch_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        step_idx = torch.minimum(self.idx, self.traj_len - 1)
        wrote_cycler_state = False
        if self._DRIVE_CYCLER_JOINTS and self.cycler_traj is not None:
            wrote_cycler_state = self._write_cycler_state(
                self.cycler_traj[batch_idx, step_idx]
            )

        action = self.traj[batch_idx, step_idx].clone()
        if self._DRIVE_CYCLER_JOINTS:
            self._apply_follow_targets(obs, action, step_idx, wrote_cycler_state)

        self._update_viewer(action)

        active = self.traj_len > 0
        at_end = self.idx >= self.traj_len - 1
        advance = active & ~at_end
        self.idx[advance] += 1
        if self.loop:
            loop_done = active & at_end
            if loop_done.any():
                self.idx[loop_done] = 0
                self.traj_len[loop_done] = 0
        return action

    def init_buffers(self, capacity: int = 1) -> None:
        self.traj = torch.zeros(
            (self.num_envs, capacity, self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.cycler_traj = torch.zeros(
            (self.num_envs, capacity, 3), device=self.device, dtype=torch.float32
        )
        self.lever_follow_local_traj = torch.full(
            (self.num_envs, capacity, 7),
            float("nan"),
            device=self.device,
            dtype=torch.float32,
        )
        self.idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.traj_len = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.lever_follow_modes = [None for _ in range(self.num_envs)]
        self.mode_by_env = [None for _ in range(self.num_envs)]

    def _ensure_plan_capacity(self, length: int) -> None:
        if (
            self.traj is None
            or self.cycler_traj is None
            or self.lever_follow_local_traj is None
        ):
            self.init_buffers(length)
            return
        current = self.traj.shape[1]
        if current >= length:
            return
        new_traj = torch.zeros(
            (self.num_envs, length, self.act_dim),
            device=self.device,
            dtype=self.traj.dtype,
        )
        new_cycler_traj = torch.zeros(
            (self.num_envs, length, 3), device=self.device, dtype=self.cycler_traj.dtype
        )
        new_follow_local = torch.full(
            (self.num_envs, length, 7),
            float("nan"),
            device=self.device,
            dtype=self.lever_follow_local_traj.dtype,
        )
        new_traj[:, :current] = self.traj
        new_cycler_traj[:, :current] = self.cycler_traj
        new_follow_local[:, :current] = self.lever_follow_local_traj
        self.traj = new_traj
        self.cycler_traj = new_cycler_traj
        self.lever_follow_local_traj = new_follow_local

    def _plan(self, obs, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        cycler_qpos_all = observation_tensor(
            obs, "cycler_qpos", self.device, self.num_envs
        )
        palm_pose_all = self._pose_obs(obs, "palm_pose_w")
        lever_pose_all = self._pose_obs(obs, "lever_site_pose_w")
        knob_pose_all = self._pose_obs(obs, "knob_site_pose_w")
        hand_all = self._current_hand(obs)

        cycler_qpos = cycler_qpos_all[env_ids]
        open_mask = cycler_qpos[:, 0] < -1.0
        if (~open_mask).any():
            close_env_ids = env_ids[~open_mask]
            self._plan_group(
                XHandThermalCyclerMode.CLOSE,
                close_env_ids,
                palm_pose_all[close_env_ids],
                lever_pose_all[close_env_ids],
                knob_pose_all[close_env_ids],
                cycler_qpos_all[close_env_ids, :3],
                hand_all[close_env_ids],
            )
        if open_mask.any():
            open_env_ids = env_ids[open_mask]
            self._plan_group(
                XHandThermalCyclerMode.OPEN,
                open_env_ids,
                palm_pose_all[open_env_ids],
                lever_pose_all[open_env_ids],
                knob_pose_all[open_env_ids],
                cycler_qpos_all[open_env_ids, :3],
                hand_all[open_env_ids],
            )

    def _plan_group(
        self,
        mode: XHandThermalCyclerMode,
        env_ids: torch.Tensor,
        palm_pose: torch.Tensor,
        lever_pose: torch.Tensor,
        knob_pose: torch.Tensor,
        cycler_qpos: torch.Tensor,
        hand: torch.Tensor,
    ) -> None:
        self._active_plan_env_ids = env_ids
        try:
            (
                action_traj,
                cycler_traj,
                follow_modes,
                follow_local_traj,
            ) = self._build_mode_plan(
                mode, palm_pose, lever_pose, knob_pose, cycler_qpos, hand
            )
        finally:
            self._active_plan_env_ids = None

        if cycler_traj.shape[1] != action_traj.shape[1]:
            raise RuntimeError(
                "Thermal cycler joint trajectory length does not match hand trajectory: "
                f"{cycler_traj.shape[1]} != {action_traj.shape[1]}"
            )
        if len(follow_modes) != action_traj.shape[1]:
            raise RuntimeError(
                "Thermal cycler follow-mode length does not match hand trajectory: "
                f"{len(follow_modes)} != {action_traj.shape[1]}"
            )
        if follow_local_traj.shape[1] != action_traj.shape[1]:
            raise RuntimeError(
                "Thermal cycler local follow trajectory length does not match hand trajectory: "
                f"{follow_local_traj.shape[1]} != {action_traj.shape[1]}"
            )

        length = action_traj.shape[1]
        self._ensure_plan_capacity(length)
        assert self.traj is not None
        assert self.cycler_traj is not None
        assert self.lever_follow_local_traj is not None
        assert self.idx is not None
        assert self.traj_len is not None
        assert self.lever_follow_modes is not None
        assert self.mode_by_env is not None

        self.traj[env_ids] = 0.0
        self.cycler_traj[env_ids] = 0.0
        self.lever_follow_local_traj[env_ids] = float("nan")
        self.traj[env_ids, :length] = action_traj
        self.cycler_traj[env_ids, :length] = cycler_traj
        self.lever_follow_local_traj[env_ids, :length] = follow_local_traj
        self.idx[env_ids] = 0
        self.traj_len[env_ids] = length
        for env_id in env_ids.detach().cpu().tolist():
            self.lever_follow_modes[int(env_id)] = list(follow_modes)
            self.mode_by_env[int(env_id)] = mode
        self.mode = mode
        print(
            f"Thermal cycler XHand planned {mode.value} trajectory "
            f"for {env_ids.numel()} env(s) with {length} control points."
        )

    def _build_mode_plan(
        self,
        mode: XHandThermalCyclerMode,
        palm_pose: torch.Tensor,
        lever_pose: torch.Tensor,
        knob_pose: torch.Tensor,
        cycler_qpos: torch.Tensor,
        hand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str | None], torch.Tensor]:
        if mode == XHandThermalCyclerMode.CLOSE:
            return self._plan_close(
                palm_pose,
                lever_pose,
                hand,
            )
        return self._plan_open(
            palm_pose,
            lever_pose,
            knob_pose,
            cycler_qpos,
            hand,
        )

    def _apply_follow_targets(
        self,
        obs,
        action: torch.Tensor,
        step_idx: torch.Tensor,
        wrote_cycler_state: bool,
    ) -> None:
        if self.lever_follow_modes is None and self.lever_follow_local_traj is None:
            return
        if wrote_cycler_state:
            self.env.sim.forward()
        lever_pose = self._current_lever_pose(obs)
        for env_id in range(self.num_envs):
            step = int(step_idx[env_id].item())
            follow_mode = None
            if (
                self.lever_follow_modes is not None
                and self.lever_follow_modes[env_id] is not None
            ):
                follow_mode = self.lever_follow_modes[env_id][step]
            follow_local_pose = None
            if self.lever_follow_local_traj is not None:
                candidate = self.lever_follow_local_traj[env_id, step]
                if not torch.isnan(candidate).any():
                    follow_local_pose = candidate.unsqueeze(0)
            if follow_local_pose is not None:
                action[env_id : env_id + 1, :7] = compose_pose_wxyz(
                    lever_pose[env_id : env_id + 1],
                    follow_local_pose,
                )
            elif follow_mode is not None:
                action[env_id : env_id + 1, :7] = self._lever_pose_to_palm_pose(
                    lever_pose[env_id : env_id + 1],
                    mode=follow_mode,
                )

    def _concat_segments(self, *segments) -> torch.Tensor:
        pieces = []
        for i, (start_pose, end_pose, start_hand, end_hand, steps) in enumerate(
            segments
        ):
            pos_traj, quat_traj = batched_interpolate_linear(
                start_pos=start_pose[:, :3],
                end_pos=end_pose[:, :3],
                start_quat=F.normalize(start_pose[:, 3:7], dim=-1),
                end_quat=F.normalize(end_pose[:, 3:7], dim=-1),
                num_steps=steps,
            )
            hand_traj = qpos_linear_interpolate(start_hand, end_hand, steps)
            segment = torch.cat((pos_traj, quat_traj, hand_traj), dim=-1)
            if i > 0:
                segment = segment[:, 1:]
            pieces.append(segment)
        return torch.cat(pieces, dim=1)

    def _compose_lever_local_pose_traj(
        self,
        lever_pose_traj: torch.Tensor,
        local_pose_traj: torch.Tensor,
    ) -> torch.Tensor:
        if lever_pose_traj.shape[:2] != local_pose_traj.shape[:2]:
            raise RuntimeError(
                "Lever and local pose trajectories must have matching batch/step dimensions: "
                f"{lever_pose_traj.shape[:2]} != {local_pose_traj.shape[:2]}"
            )
        return torch.stack(
            [
                compose_pose_wxyz(lever_pose_traj[:, step], local_pose_traj[:, step])
                for step in range(lever_pose_traj.shape[1])
            ],
            dim=1,
        )

    def _lever_guided_world_pose_traj(
        self,
        lever_pose_traj: torch.Tensor,
        start_palm_pose: torch.Tensor,
        end_palm_pose: torch.Tensor,
    ) -> torch.Tensor:
        if lever_pose_traj.ndim != 3 or lever_pose_traj.shape[-1] != 7:
            raise RuntimeError(
                f"Expected lever pose trajectory shape (B, T, 7), got {tuple(lever_pose_traj.shape)}"
            )

        steps = lever_pose_traj.shape[1] - 1
        t = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=self.device,
            dtype=torch.float32,
        )[None, :, None]
        start_offset = start_palm_pose[:, :3] - lever_pose_traj[:, 0, :3]
        end_offset = end_palm_pose[:, :3] - lever_pose_traj[:, -1, :3]
        offset_traj = (1.0 - t) * start_offset[:, None, :] + t * end_offset[:, None, :]
        pos_traj = lever_pose_traj[:, :, :3] + offset_traj
        _, quat_traj = batched_interpolate_linear(
            start_pos=start_palm_pose[:, :3],
            end_pos=end_palm_pose[:, :3],
            start_quat=F.normalize(start_palm_pose[:, 3:7], dim=-1),
            end_quat=F.normalize(end_palm_pose[:, 3:7], dim=-1),
            num_steps=steps,
        )
        return torch.cat((pos_traj, quat_traj), dim=-1)

    def _concat_joint_segments(self, *segments) -> torch.Tensor:
        pieces = []
        for i, (start_qpos, end_qpos, steps) in enumerate(segments):
            segment = qpos_linear_interpolate(start_qpos, end_qpos, steps)
            if i > 0:
                segment = segment[:, 1:]
            pieces.append(segment)
        return torch.cat(pieces, dim=1)

    def _concat_follow_modes(self, *segments) -> list[str | None]:
        modes: list[str | None] = []
        for i, (mode, steps) in enumerate(segments):
            count = steps + 1 if i == 0 else steps
            modes.extend([mode] * count)
        return modes

    def _concat_local_follow_segments(self, *segments) -> torch.Tensor:
        pieces = []
        for i, (start_pose, end_pose, steps) in enumerate(segments):
            if start_pose is None or end_pose is None:
                segment = torch.full(
                    (self._plan_batch_size(), steps + 1, 7),
                    float("nan"),
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                pos_traj, quat_traj = batched_interpolate_linear(
                    start_pos=start_pose[:, :3],
                    end_pos=end_pose[:, :3],
                    start_quat=F.normalize(start_pose[:, 3:7], dim=-1),
                    end_quat=F.normalize(end_pose[:, 3:7], dim=-1),
                    num_steps=steps,
                )
                segment = torch.cat((pos_traj, quat_traj), dim=-1)
            if i > 0:
                segment = segment[:, 1:]
            pieces.append(segment)
        return torch.cat(pieces, dim=1)

    def _cycler_qpos(self, qpos: tuple[float, float, float]) -> torch.Tensor:
        return (
            torch.tensor(qpos, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self._plan_batch_size(), 1)
        )

    def _plan_batch_size(self) -> int:
        if self._active_plan_env_ids is not None:
            return int(self._active_plan_env_ids.numel())
        return self.num_envs

    def _pose_obs(self, obs, key: str) -> torch.Tensor:
        pose = observation_tensor(obs, key, self.device, self.num_envs)
        if pose.ndim == 3:
            pose = pose[:, 0, :]
        pose = pose[:, :7]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _current_hand(self, obs) -> torch.Tensor:
        qpos = observation_tensor(obs, "current_qpos", self.device, self.num_envs)
        return qpos[:, -self.hand_dim :]

    def _current_lever_pose(self, obs) -> torch.Tensor:
        lever_pose = self._lever_pose_from_scene()
        if lever_pose is None:
            lever_pose = self._pose_obs(obs, "lever_site_pose_w")
        return lever_pose

    def _lever_pose_from_scene(self) -> torch.Tensor | None:
        try:
            cycler = self.env.scene["thermal_cycler"]
        except KeyError:
            return None
        site_id = self._resolve_lever_site_id()
        if site_id is None:
            return None
        pose = cycler.data.site_pose_w[:, site_id, :7].to(
            device=self.device, dtype=torch.float32
        )
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _cycler_lever_pose_for_qpos_traj(self, qpos_traj: torch.Tensor) -> torch.Tensor:
        return self._cycler_site_pose_for_qpos_traj(
            qpos_traj,
            self._resolve_lever_site_id(),
            "lever",
        )

    def _cycler_knob_pose_for_qpos_traj(self, qpos_traj: torch.Tensor) -> torch.Tensor:
        return self._cycler_site_pose_for_qpos_traj(
            qpos_traj,
            self._resolve_knob_site_id(),
            "knob",
        )

    def _cycler_site_pose_for_qpos_traj(
        self,
        qpos_traj: torch.Tensor,
        site_id: int | None,
        site_label: str,
    ) -> torch.Tensor:
        try:
            cycler = self.env.scene["thermal_cycler"]
        except KeyError as exc:
            raise RuntimeError(
                f"Thermal cycler entity is required to sample {site_label} FK."
            ) from exc
        joint_ids = self._resolve_cycler_joint_ids()
        if joint_ids is None or site_id is None:
            raise RuntimeError(
                f"Could not resolve thermal cycler joint/site ids for {site_label} FK."
            )

        if self._active_plan_env_ids is None:
            env_ids = torch.arange(
                qpos_traj.shape[0], device=self.device, dtype=torch.long
            )
        else:
            env_ids = self._active_plan_env_ids
        indexing = cycler.data.indexing
        qpos_adrs = indexing.joint_q_adr[joint_ids].detach().cpu().numpy()
        global_site_id = int(indexing.site_ids[site_id].item())
        live_data = self.env.sim.data
        qpos_traj_np = qpos_traj.detach().cpu().numpy()
        env_ids_list = env_ids.detach().cpu().tolist()
        poses = np.empty(
            (qpos_traj_np.shape[0], qpos_traj_np.shape[1], 7),
            dtype=np.float32,
        )

        for batch_id, env_id in enumerate(env_ids_list):
            self._fk_data.qpos[:] = live_data.qpos[env_id].detach().cpu().numpy()
            if self.env.sim.mj_model.nmocap:
                self._fk_data.mocap_pos[:] = (
                    live_data.mocap_pos[env_id].detach().cpu().numpy()
                )
                self._fk_data.mocap_quat[:] = (
                    live_data.mocap_quat[env_id].detach().cpu().numpy()
                )

            for step in range(qpos_traj_np.shape[1]):
                self._fk_data.qpos[qpos_adrs] = qpos_traj_np[batch_id, step]
                mujoco.mj_kinematics(self.env.sim.mj_model, self._fk_data)

                quat = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, self._fk_data.site_xmat[global_site_id])
                poses[batch_id, step, :3] = self._fk_data.site_xpos[global_site_id]
                poses[batch_id, step, 3:7] = quat

        return torch.as_tensor(poses, device=self.device, dtype=torch.float32)

    def _write_cycler_state(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> bool:
        try:
            cycler = self.env.scene["thermal_cycler"]
        except KeyError:
            return False
        joint_ids = self._resolve_cycler_joint_ids()
        if joint_ids is None:
            return False
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        qpos = qpos.to(device=self.device, dtype=cycler.data.joint_pos.dtype)
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0).repeat(env_ids.numel(), 1)
        if qpos.shape[0] != env_ids.numel():
            raise RuntimeError(
                "Thermal cycler qpos batch does not match env_ids: "
                f"{qpos.shape[0]} != {env_ids.numel()}"
            )
        if qvel is None:
            qvel = torch.zeros_like(qpos)
        else:
            qvel = qvel.to(device=self.device, dtype=cycler.data.joint_vel.dtype)
            if qvel.ndim == 1:
                qvel = qvel.unsqueeze(0).repeat(env_ids.numel(), 1)
            if qvel.shape[0] != env_ids.numel():
                raise RuntimeError(
                    "Thermal cycler qvel batch does not match env_ids: "
                    f"{qvel.shape[0]} != {env_ids.numel()}"
                )
        cycler.write_joint_state_to_sim(
            qpos, qvel, env_ids=env_ids, joint_ids=joint_ids
        )
        return True

    def _resolve_cycler_joint_ids(self) -> torch.Tensor | None:
        if self._cycler_joint_ids is not None:
            return self._cycler_joint_ids
        try:
            cycler = self.env.scene["thermal_cycler"]
        except KeyError:
            return None
        target_names = ("lid", "lid-lever", "lid-force-knob")
        try:
            joint_ids, _ = cycler.find_joints(target_names, preserve_order=True)
        except Exception:
            joint_ids = []
        if len(joint_ids) != len(target_names):
            joint_names = getattr(cycler, "joint_names", None)
            if joint_names is None:
                return None
            joint_ids = []
            for target_name in target_names:
                match = next(
                    (
                        joint_id
                        for joint_id, joint_name in enumerate(joint_names)
                        if joint_name == target_name
                        or str(joint_name).endswith(target_name)
                    ),
                    None,
                )
                if match is None:
                    return None
                joint_ids.append(match)
        self._cycler_joint_ids = torch.tensor(
            joint_ids, device=self.device, dtype=torch.long
        )
        return self._cycler_joint_ids

    def _resolve_lever_site_id(self) -> int | None:
        if self._lever_site_id is not None:
            return self._lever_site_id
        self._lever_site_id = self._resolve_cycler_site_id("lid-lever")
        return self._lever_site_id

    def _resolve_knob_site_id(self) -> int | None:
        if self._knob_site_id is not None:
            return self._knob_site_id
        self._knob_site_id = self._resolve_cycler_site_id("lid-force-knob")
        return self._knob_site_id

    def _resolve_cycler_site_id(self, target_name: str) -> int | None:
        try:
            cycler = self.env.scene["thermal_cycler"]
        except KeyError:
            return None
        try:
            site_ids, _ = cycler.find_sites([target_name], preserve_order=True)
        except Exception:
            site_ids = []
        if not site_ids:
            site_names = getattr(cycler, "site_names", None)
            if site_names is None:
                return None
            for site_id, site_name in enumerate(site_names):
                if site_name == target_name or str(site_name).endswith(target_name):
                    return site_id
            return None
        return int(site_ids[0])

    def _open_hand(self) -> torch.Tensor:
        return torch.zeros((self._plan_batch_size(), self.hand_dim), device=self.device)

    def _pregrasp_hand(self, mode: str) -> torch.Tensor:
        hand = self._open_hand()
        grip = self._grip_hand(mode)
        for joint_id, joint_expr in enumerate(XHAND_JOINT_ORDER):
            if joint_expr.replace(".*", "").startswith("thumb_"):
                hand[:, joint_id] = grip[:, joint_id]
        return hand

    def _grip_hand(self, mode: str) -> torch.Tensor:
        if mode == "close_start":
            targets = self._CLOSE_START_GRIP_QPOS
        elif mode == "spinner":
            targets = self._SPINNER_GRIP_QPOS
        elif mode == "open_start":
            targets = self._OPEN_START_GRIP_QPOS
        elif mode == "open_insert":
            targets = self._OPEN_INSERT_GRIP_QPOS
        elif mode == "close_end":
            targets = self._CLOSE_END_GRIP_QPOS
        else:
            raise ValueError(f"Unknown XHand grip mode: {mode}")
        values = []
        for joint_expr in XHAND_JOINT_ORDER:
            values.append(targets[joint_expr.replace(".*", "")])
        return (
            torch.tensor(values, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(
                self._plan_batch_size(),
                1,
            )
        )

    def _lever_pose_to_palm_pose(
        self, lever_pose: torch.Tensor, mode: str
    ) -> torch.Tensor:
        return compose_pose_wxyz(
            lever_pose, self._lever_local_pose(mode, batch_size=lever_pose.shape[0])
        )

    def _lever_local_pose(
        self, mode: str, batch_size: int | None = None
    ) -> torch.Tensor:
        if mode == "side_detach":
            rel_pos = self._offset_pos(self._CLOSE_START_GRASP_POS, (0.0, 0.046, 0.0))
            rel_quat = self._const_quat(
                self._CLOSE_START_GRASP_QUAT, batch_size=batch_size
            )
        elif mode == "side_grip":
            rel_pos = self._CLOSE_START_GRASP_POS
            rel_quat = self._const_quat(
                self._CLOSE_START_GRASP_QUAT, batch_size=batch_size
            )
        elif mode == "side_grip_backoff":
            rel_pos = self._offset_pos(
                self._CLOSE_START_GRASP_POS,
                (0.0, self._CLOSE_GRASP_BACKOFF_Y, 0.0),
            )
            rel_quat = self._const_quat(
                self._CLOSE_START_GRASP_QUAT, batch_size=batch_size
            )
        elif mode == "close_end_grip":
            rel_pos = self._CLOSE_END_GRASP_POS
            rel_quat = self._const_quat(
                self._CLOSE_END_GRASP_QUAT, batch_size=batch_size
            )
        elif mode == "spinner_grip":
            rel_pos = self._SPINNER_GRASP_LEVER_POS
            rel_quat = self._const_quat(
                self._SPINNER_GRASP_LEVER_QUAT, batch_size=batch_size
            )
        else:
            raise ValueError(f"Unknown lever palm mode: {mode}")
        return self._local_pose(rel_pos, rel_quat, batch_size=batch_size)

    def _with_world_delta(
        self,
        pose: torch.Tensor,
        delta: tuple[float, float, float],
    ) -> torch.Tensor:
        out = pose.clone()
        out[:, :3] = pose[:, :3] + torch.tensor(
            delta, device=self.device, dtype=torch.float32
        )
        return out

    def _local_pose(
        self,
        pos: tuple[float, float, float],
        quat: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = self._plan_batch_size()
        local = torch.zeros((batch_size, 7), device=self.device, dtype=torch.float32)
        local[:, :3] = torch.tensor(pos, device=self.device, dtype=torch.float32)
        local[:, 3:7] = quat
        return local

    def _const_quat(
        self,
        quat: tuple[float, float, float, float],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = self._plan_batch_size()
        out = (
            torch.tensor(quat, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        return F.normalize(out, dim=-1)

    def _offset_pos(
        self,
        pos: tuple[float, float, float],
        delta: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return tuple(float(a + b) for a, b in zip(pos, delta, strict=True))

    def _update_viewer(self, action: torch.Tensor) -> None:
        if self.target_frame is not None:
            pose = action[0, :7].detach().cpu().numpy()
            self.target_frame.position = pose[:3]
            self.target_frame.wxyz = pose[3:7]


class XHandThermalCyclerCloseVisualPolicy(XHandThermalCyclerClosePolicy):
    """Full scripted motion with contacts disabled for visual trajectory playback."""

    _DRIVE_CYCLER_JOINTS = True
    _DISABLE_CONTACTS = True


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return XHandThermalCyclerClosePolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        loop=False,
        server=server,
    )
