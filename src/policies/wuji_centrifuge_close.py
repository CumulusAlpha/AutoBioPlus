from __future__ import annotations

import re
from enum import Enum

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from assets.mjcf.robot.ur5e_wuji import WUJI_JOINT_ORDER
from src.utils.kinematics import batched_interpolate_linear, qpos_linear_interpolate
from src.utils.observations import observation_tensor
from src.utils.quat_util import compose_pose_wxyz

GRASP_POSE_LID = (
    0.465299,
    -0.088913,
    0.056548,
    -0.030148,
    -0.711907,
    0.041464,
    0.7004,
)
PREGRASP_POSE_LID = (
    0.623067,
    -0.090699,
    0.058558,
    -0.030148,
    -0.711907,
    0.041464,
    0.7004,
)
WUJI_GRASP_JOINT_QPOS = {
    "right_finger1_joint1": 1.028,
    "right_finger1_joint2": -0.1387,
    "right_finger1_joint3": 1.095,
    "right_finger1_joint4": 0.127,
    "right_finger2_joint1": 0.126,
    "right_finger2_joint2": -0.008,
    "right_finger2_joint3": 0.0,
    "right_finger2_joint4": 0.0,
    "right_finger3_joint1": 0.12,
    "right_finger3_joint2": 0.0,
    "right_finger3_joint3": 0.0,
    "right_finger3_joint4": 0.0,
    "right_finger4_joint1": 0.198,
    "right_finger4_joint2": 0.0,
    "right_finger4_joint3": 0.0,
    "right_finger4_joint4": 0.0,
    "right_finger5_joint1": 0.052,
    "right_finger5_joint2": 0.0,
    "right_finger5_joint3": 0.0,
    "right_finger5_joint4": 0.0,
}


class WujiCentrifugeCloseState(Enum):
    MOVE_TO_PREGRASP = "move to pregrasp pose"
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    CLOSE_LID = "close the lid"
    WITHDRAW_HAND = "withdraw hand"


class WujiCentrifugeClosePolicy:
    """Five-state UR5e+Wuji policy for closing the 5910 Ri centrifuge lid."""

    _STATE_ORDER = tuple(WujiCentrifugeCloseState)

    def __init__(
        self,
        action_shape,
        device,
        env,
        move_steps: int = 120,
        pregrasp_steps: int = 30,
        grasp_steps: int = 60,
        close_steps: int = 180,
        withdraw_steps: int = 80,
        withdraw_height: float = 0.15,
        loop: bool = False,
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

        self.move_steps = max(1, int(move_steps))
        self.pregrasp_steps = max(1, int(pregrasp_steps))
        self.grasp_steps = max(1, int(grasp_steps))
        self.close_steps = max(1, int(close_steps))
        self.withdraw_steps = max(1, int(withdraw_steps))
        self.withdraw_height = float(withdraw_height)
        self.loop = bool(loop)
        self.server = server

        self.grasp_pose_lid = torch.tensor(
            GRASP_POSE_LID, device=device, dtype=torch.float32
        )
        self.grasp_pose_lid[3:7] = F.normalize(self.grasp_pose_lid[3:7], dim=0)
        self.grasp_hand = self._joint_targets(
            WUJI_GRASP_JOINT_QPOS, "WUJI_GRASP_JOINT_QPOS"
        )

        self.pregrasp_pose_lid = torch.tensor(
            PREGRASP_POSE_LID,
            device=device,
            dtype=torch.float32,
        )
        self.pregrasp_pose_lid[3:7] = F.normalize(self.pregrasp_pose_lid[3:7], dim=0)
        self.pregrasp_hand = self._joint_targets(
            WUJI_GRASP_JOINT_QPOS, "WUJI_GRASP_JOINT_QPOS"
        )

        self.centrifuge = env.scene["centrifuge"]
        lid_joint_ids, _ = self.centrifuge.find_joints(["lid"], preserve_order=True)
        lid_body_ids, _ = self.centrifuge.find_bodies(["lid"], preserve_order=True)
        if len(lid_joint_ids) != 1 or len(lid_body_ids) != 1:
            raise RuntimeError(
                "Could not uniquely resolve centrifuge lid joint and body."
            )
        self._lid_joint_id = lid_joint_ids[0]
        self._lid_body_id = lid_body_ids[0]
        self._lid_qpos_adr = int(
            self.centrifuge.data.indexing.joint_q_adr[self._lid_joint_id].item()
        )
        self._lid_global_body_id = int(
            self.centrifuge.data.indexing.body_ids[self._lid_body_id].item()
        )
        global_joint_id = int(
            self.centrifuge.data.indexing.joint_ids[self._lid_joint_id].item()
        )
        self._closed_lid_qpos = float(env.sim.mj_model.jnt_range[global_joint_id, 1])
        self._lid_lock_eq_id = self._resolve_lid_lock_eq_id()
        self._fk_data = mujoco.MjData(env.sim.mj_model)
        self._set_lid_lock_active(False)
        self._set_env_policy_complete(False)

        self.state = WujiCentrifugeCloseState.MOVE_TO_PREGRASP
        self._state_index = 0
        self._step_index = 0
        self._trajectories: dict[WujiCentrifugeCloseState, torch.Tensor] | None = (
            None
        )
        self._target_frame = None
        if server is not None:
            self._target_frame = server.scene.add_frame(
                name="wuji_centrifuge_close/target_palm",
                axes_length=0.08,
                axes_radius=0.004,
            )

        print(
            "Loaded Wuji centrifuge close policy: "
            "using built-in pregrasp/grasp poses, "
            f"closed_lid_qpos={self._closed_lid_qpos:.4f}"
        )

    def reset(self) -> None:
        self.state = WujiCentrifugeCloseState.MOVE_TO_PREGRASP
        self._state_index = 0
        self._step_index = 0
        self._trajectories = None
        self._set_lid_lock_active(False)
        self._set_env_policy_complete(False)

    def __call__(self, obs):
        if self._trajectories is None:
            self._trajectories = self._build_trajectories(obs)

        trajectory = self._trajectories[self.state]
        action = trajectory[:, self._step_index].clone()
        self._update_viewer(action)

        if self._step_index < trajectory.shape[1] - 1:
            self._step_index += 1
        elif self._state_index < len(self._STATE_ORDER) - 1:
            if self.state is WujiCentrifugeCloseState.CLOSE_LID:
                self._set_lid_lock_active(True)
            self._state_index += 1
            self.state = self._STATE_ORDER[self._state_index]
            self._step_index = 0
            print(f"Wuji centrifuge close state: {self.state.value}")
        elif self.loop:
            self._set_env_policy_complete(True)
            self.reset()
        else:
            self._set_env_policy_complete(True)
        return action

    def _build_trajectories(
        self, obs
    ) -> dict[WujiCentrifugeCloseState, torch.Tensor]:
        palm_pose = observation_tensor(
            obs, "palm_body_pose_w", self.device, self.num_envs
        )[:, :7]
        palm_pose[:, 3:7] = F.normalize(palm_pose[:, 3:7], dim=-1)
        current_qpos = observation_tensor(
            obs, "current_qpos", self.device, self.num_envs
        )
        current_hand = current_qpos[:, -self.hand_dim :]
        lid_qpos = observation_tensor(
            obs, "centrifuge_qpos", self.device, self.num_envs
        )[:, 0]
        lid_pose = self._current_lid_body_pose()
        grasp_pose = compose_pose_wxyz(
            lid_pose,
            self.grasp_pose_lid.unsqueeze(0).repeat(self.num_envs, 1),
        )
        pregrasp_pose = compose_pose_wxyz(
            lid_pose,
            self.pregrasp_pose_lid.unsqueeze(0).repeat(self.num_envs, 1),
        )
        grasp_hand = self.grasp_hand.unsqueeze(0).repeat(self.num_envs, 1)
        pregrasp_hand = self.pregrasp_hand.unsqueeze(0).repeat(self.num_envs, 1)

        move_to_pregrasp = self._interpolate_action(
            palm_pose, pregrasp_pose, current_hand, pregrasp_hand, self.move_steps
        )
        pregrasp = self._interpolate_action(
            pregrasp_pose,
            pregrasp_pose,
            pregrasp_hand,
            pregrasp_hand,
            self.pregrasp_steps,
        )
        grasp = self._interpolate_action(
            pregrasp_pose, grasp_pose, pregrasp_hand, grasp_hand, self.grasp_steps
        )

        lid_qpos_traj = qpos_linear_interpolate(
            lid_qpos[:, None],
            torch.full_like(lid_qpos[:, None], self._closed_lid_qpos),
            self.close_steps,
        )
        lid_pose_traj = self._lid_body_fk(lid_qpos_traj)
        local_grasp = self.grasp_pose_lid[None, None, :].repeat(
            self.num_envs, lid_pose_traj.shape[1], 1
        )
        close_pose = torch.stack(
            [
                compose_pose_wxyz(lid_pose_traj[:, step], local_grasp[:, step])
                for step in range(lid_pose_traj.shape[1])
            ],
            dim=1,
        )
        close = torch.cat(
            (
                close_pose,
                grasp_hand[:, None, :].repeat(1, close_pose.shape[1], 1),
            ),
            dim=-1,
        )

        withdraw_pose = close_pose[:, -1].clone()
        withdraw_pose[:, 2] += self.withdraw_height
        withdraw = self._interpolate_action(
            close_pose[:, -1],
            withdraw_pose,
            grasp_hand,
            current_hand,
            self.withdraw_steps,
        )
        return {
            WujiCentrifugeCloseState.MOVE_TO_PREGRASP: move_to_pregrasp,
            WujiCentrifugeCloseState.PREGRASP: pregrasp,
            WujiCentrifugeCloseState.GRASP: grasp,
            WujiCentrifugeCloseState.CLOSE_LID: close,
            WujiCentrifugeCloseState.WITHDRAW_HAND: withdraw,
        }

    def _interpolate_action(
        self,
        start_pose: torch.Tensor,
        end_pose: torch.Tensor,
        start_hand: torch.Tensor,
        end_hand: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        pos, quat = batched_interpolate_linear(
            start_pos=start_pose[:, :3],
            end_pos=end_pose[:, :3],
            start_quat=F.normalize(start_pose[:, 3:7], dim=-1),
            end_quat=F.normalize(end_pose[:, 3:7], dim=-1),
            num_steps=steps,
        )
        hand = qpos_linear_interpolate(start_hand, end_hand, steps)
        return torch.cat((pos, quat, hand), dim=-1)

    def _current_lid_body_pose(self) -> torch.Tensor:
        pose = self.centrifuge.data.body_link_pose_w[:, self._lid_body_id, :7].to(
            device=self.device, dtype=torch.float32
        )
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _lid_body_fk(self, lid_qpos_traj: torch.Tensor) -> torch.Tensor:
        live_data = self.env.sim.data
        qpos_np = lid_qpos_traj.detach().cpu().numpy()
        poses = np.empty((*qpos_np.shape[:2], 7), dtype=np.float32)
        for env_id in range(self.num_envs):
            self._fk_data.qpos[:] = live_data.qpos[env_id].detach().cpu().numpy()
            if self.env.sim.mj_model.nmocap:
                self._fk_data.mocap_pos[:] = (
                    live_data.mocap_pos[env_id].detach().cpu().numpy()
                )
                self._fk_data.mocap_quat[:] = (
                    live_data.mocap_quat[env_id].detach().cpu().numpy()
                )
            for step in range(qpos_np.shape[1]):
                self._fk_data.qpos[self._lid_qpos_adr] = qpos_np[env_id, step, 0]
                mujoco.mj_kinematics(self.env.sim.mj_model, self._fk_data)
                poses[env_id, step, :3] = self._fk_data.xpos[self._lid_global_body_id]
                poses[env_id, step, 3:7] = self._fk_data.xquat[self._lid_global_body_id]
        return torch.as_tensor(poses, device=self.device, dtype=torch.float32)

    def _resolve_lid_lock_eq_id(self) -> int:
        model = self.env.sim.mj_model
        for eq_id in range(model.neq):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
            if name == "lid-lock" or (name is not None and name.endswith("/lid-lock")):
                return int(eq_id)
        raise RuntimeError("Could not resolve centrifuge lid-lock equality.")

    def _set_lid_lock_active(self, active: bool) -> None:
        value = bool(active)
        try:
            self.env.sim.data.eq_active[:, self._lid_lock_eq_id] = value
        except Exception:
            pass
        if hasattr(self.env.sim, "mj_data"):
            self.env.sim.mj_data.eq_active[self._lid_lock_eq_id] = value

    def _set_env_policy_complete(
        self, value, env_ids: torch.Tensor | None = None
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        flag = getattr(self.env, "wuji_centrifuge_close_policy_complete", None)
        if flag is None or flag.shape[0] != self.num_envs:
            flag = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        if isinstance(value, torch.Tensor):
            value = value.to(device=self.device, dtype=torch.bool)
            if value.ndim == 0:
                flag[env_ids] = value
            elif value.shape[0] == self.num_envs:
                flag[env_ids] = value[env_ids]
            elif value.shape[0] == env_ids.numel():
                flag[env_ids] = value
            else:
                raise ValueError(
                    "Cannot set wuji_centrifuge_close_policy_complete for "
                    f"{env_ids.numel()} envs from value shape {tuple(value.shape)}."
                )
        else:
            flag[env_ids] = bool(value)
        self.env.wuji_centrifuge_close_policy_complete = flag

    def _joint_targets(self, targets: dict[str, float], source: str) -> torch.Tensor:
        values = []
        for expression in WUJI_JOINT_ORDER:
            match = next(
                (
                    value
                    for name, value in targets.items()
                    if re.fullmatch(expression, name)
                    or name.endswith(expression.removeprefix(".*"))
                ),
                None,
            )
            if match is None:
                raise KeyError(f"Missing Wuji joint {expression!r} in {source}")
            values.append(match)
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _update_viewer(self, action: torch.Tensor) -> None:
        if self._target_frame is None:
            return
        pose = action[0, :7].detach().cpu().numpy()
        self._target_frame.position = pose[:3]
        self._target_frame.wxyz = pose[3:7]


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return WujiCentrifugeClosePolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        loop=False,
        server=server,
    )
