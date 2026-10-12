from __future__ import annotations

from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
import viser

from assets.mjcf.robot.ur5e_xhand import XHAND_JOINT_ORDER
from src.tasks.mdp.events import get_mixer_ui_state
from src.utils.kinematics import batched_interpolate_linear, qpos_linear_interpolate
from src.utils.observations import observation_tensor
from src.utils.quat_util import (
    axis_angle_to_quat,
    compose_pose_wxyz,
    quat_distance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs"
XHAND_TWEEZER_REL_POSE_PREFIX = "931_xhand_tweezer_rel_pose_"
XHAND_PREGRASP_QPOS_FILE = (
    LOG_DIR / "931_xhand_tweezer_rel_pose_20260525_182118_460247.txt"
)

MIXER_BUTTON_SITE_NAMES = (
    "button-speed-up",
    "button-speed-down",
    "button-temp-up",
    "button-temp-down",
    "button-time-up",
    "button-time-down",
)


class XHandMixerPanelManipulateState(Enum):
    APPROACH = 0
    PRESS = 1
    WAIT_UI = 2
    RETREAT = 3
    FINAL_LIFT = 4
    DONE = 5


class XHandMixerPanelManipulatePolicy:
    """Event-driven UR5e+XHand policy that adjusts mixer UI targets with the index tip."""

    def __init__(
        self,
        action_shape,
        device,
        env,
        approach_steps: int = 80,
        press_steps: int = 35,
        hold_steps: int = 5,
        retreat_steps: int = 50,
        final_lift_steps: int = 50,
        approach_offset: tuple[float, float, float] = (0.0, 0.0, 0.055),
        press_offset: tuple[float, float, float] = (0.0048, 0.0, -0.006),
        retreat_offset: tuple[float, float, float] = (0.0, 0.0, -0.075),
        final_lift_offset: tuple[float, float, float] = (0.0, 0.0, 0.22),
        pos_tol: float = 0.01,
        rot_tol: float = 0.20,
        loop: bool = False,
        server: viser.ViserServer = None,
    ) -> None:
        self.device = device
        self.env = env
        self.server = server
        self.num_envs, self.act_dim = action_shape
        self.pose_dim = 7
        self.hand_dim = len(XHAND_JOINT_ORDER)
        self.expected_action_dim = self.pose_dim + self.hand_dim
        if self.act_dim != self.expected_action_dim:
            raise ValueError(
                f"Expected action_dim={self.expected_action_dim} "
                f"(7D right_index_tip IK + {self.hand_dim} XHand qpos), got {self.act_dim}."
            )

        self.approach_steps = max(1, int(approach_steps))
        self.press_steps = max(1, int(press_steps))
        self.hold_steps = max(1, int(hold_steps))
        self.retreat_steps = max(1, int(retreat_steps))
        self.final_lift_steps = max(1, int(final_lift_steps))
        self.loop = bool(loop)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)

        self.approach_offset = torch.tensor(
            approach_offset, device=device, dtype=torch.float32
        )
        self.press_offset = torch.tensor(
            press_offset, device=device, dtype=torch.float32
        )
        self.retreat_offset = torch.tensor(
            retreat_offset, device=device, dtype=torch.float32
        )
        self.final_lift_offset = torch.tensor(
            final_lift_offset, device=device, dtype=torch.float32
        )
        # tip-frame axes in button frame:
        # tip_x = button_y, tip_y = button_x, tip_z = -button_z
        self.button_tip_rot_offset = axis_angle_to_quat(
            torch.tensor([1.0, 1.0, 0.0], device=device, dtype=torch.float32),
            torch.tensor([torch.pi], device=device, dtype=torch.float32),
            1,
        )[0]
        self.point_hand_action = self._build_point_hand_action()
        self.current_tip_frame = None
        self.target_tip_frame = None
        self.sensor_frame = None
        self.debug_text = None
        if self.server is not None:
            self.current_tip_frame = self.server.scene.add_frame(
                name="xhand_mixerpanel_manipulate/current_index_tip",
                axes_length=0.05,
                axes_radius=0.004,
                origin_radius=0.01,
            )
            self.target_tip_frame = self.server.scene.add_frame(
                name="xhand_mixerpanel_manipulate/target_index_tip",
                axes_length=0.08,
                axes_radius=0.006,
                origin_radius=0.012,
            )
            self.sensor_frame = self.server.scene.add_frame(
                name="xhand_mixerpanel_manipulate/target_button_sensor",
                axes_length=0.06,
                axes_radius=0.005,
                origin_radius=0.011,
            )
            with self.server.gui.add_folder("XHand Mixer Debug"):
                self.debug_text = self.server.gui.add_text(
                    label="ui/info",
                    initial_value="",
                    multiline=True,
                    disabled=True,
                )

        self.traj: torch.Tensor | None = None
        self.traj_len: torch.Tensor | None = None
        self.idx: torch.Tensor | None = None
        self.state: torch.Tensor | None = None
        self.param_idx: torch.Tensor | None = None
        self.target_button_ids: torch.Tensor | None = None

        print(
            "Loaded XHand mixer button policy: "
            f"approach={self.approach_steps}, press={self.press_steps}, "
            f"hold={self.hold_steps}, retreat={self.retreat_steps}, lift={self.final_lift_steps}, "
            f"pos_tol={self.pos_tol}, rot_tol={self.rot_tol}, action_dim={self.act_dim}"
        )

    def reset(self) -> None:
        self.traj = None
        self.traj_len = None
        self.idx = None
        self.state = None
        self.param_idx = None
        self.target_button_ids = None

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
            self.state[env_ids] = XHandMixerPanelManipulateState.APPROACH.value
        if self.param_idx is not None:
            self.param_idx[env_ids] = 0
        if self.target_button_ids is not None:
            self.target_button_ids[env_ids] = -1

        if 0 in env_ids.detach().cpu().tolist() and self.debug_text is not None:
            self.debug_text.value = "RESET"

    def __call__(self, obs):
        if self.traj is None:
            self.init_buffers()

        self._skip_completed_params()
        need_plan = self.traj_len == 0
        if need_plan.any():
            self.plan_state_batch(obs, need_plan)

        self._sync_target_button_ids()

        batch_idx = torch.arange(self.num_envs, device=self.device)
        action = self.traj[batch_idx, self.idx]
        self._update_tip_frames(obs, action)
        self._update_debug_text(obs)

        done = self._state_event_reached(obs, action)
        if done.any():
            self.advance_state(done)
            replan = self.traj_len == 0
            if replan.any():
                self.plan_state_batch(obs, replan)

        active = ~done
        if active.any():
            self.idx[active] = torch.minimum(
                self.idx[active] + 1, self.traj_len[active] - 1
            )

        return action

    def init_buffers(self) -> None:
        max_len = (
            max(
                self.approach_steps,
                self.press_steps,
                self.hold_steps,
                self.retreat_steps,
                self.final_lift_steps,
            )
            + 1
        )
        self.traj = torch.zeros(
            (self.num_envs, max_len, self.act_dim), device=self.device
        )
        self.traj_len = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.state = torch.full(
            (self.num_envs,),
            XHandMixerPanelManipulateState.APPROACH.value,
            device=self.device,
            dtype=torch.long,
        )
        self.param_idx = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.target_button_ids = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )

    def plan_state_batch(self, obs, plan_mask: torch.Tensor) -> None:
        state_table = {
            XHandMixerPanelManipulateState.APPROACH.value: self._plan_approach,
            XHandMixerPanelManipulateState.PRESS.value: self._plan_press,
            XHandMixerPanelManipulateState.WAIT_UI.value: self._plan_wait_ui,
            XHandMixerPanelManipulateState.RETREAT.value: self._plan_retreat,
            XHandMixerPanelManipulateState.FINAL_LIFT.value: self._plan_final_lift,
            XHandMixerPanelManipulateState.DONE.value: self._plan_done,
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
            (self.num_envs, new_len, self.act_dim), device=self.device
        )
        self.traj[:, : old_traj.shape[1]] = old_traj

    def advance_state(self, done_mask: torch.Tensor) -> None:
        new_state = self.state.clone()
        approach_done = done_mask & (
            self.state == XHandMixerPanelManipulateState.APPROACH.value
        )
        press_done = done_mask & (
            self.state == XHandMixerPanelManipulateState.PRESS.value
        )
        hold_done = done_mask & (
            self.state == XHandMixerPanelManipulateState.WAIT_UI.value
        )
        retreat_done = done_mask & (
            self.state == XHandMixerPanelManipulateState.RETREAT.value
        )
        final_lift_done = done_mask & (
            self.state == XHandMixerPanelManipulateState.FINAL_LIFT.value
        )

        new_state[approach_done] = XHandMixerPanelManipulateState.PRESS.value
        new_state[press_done] = XHandMixerPanelManipulateState.WAIT_UI.value
        new_state[hold_done] = XHandMixerPanelManipulateState.RETREAT.value
        if self.loop:
            new_state[retreat_done] = XHandMixerPanelManipulateState.APPROACH.value
        elif retreat_done.any():
            self.param_idx[retreat_done] += 1
            new_state[retreat_done & (self.param_idx >= 3)] = (
                XHandMixerPanelManipulateState.FINAL_LIFT.value
            )
            new_state[retreat_done & (self.param_idx < 3)] = (
                XHandMixerPanelManipulateState.APPROACH.value
            )
            self.target_button_ids[retreat_done] = -1
        new_state[final_lift_done] = XHandMixerPanelManipulateState.DONE.value

        self.state = new_state
        replanned = done_mask & (
            self.loop
            | (~retreat_done)
            | (self.state != XHandMixerPanelManipulateState.RETREAT.value)
        )
        self.traj_len[replanned] = 0
        self.idx[done_mask] = 0
        done_final = (
            retreat_done
            & (~self.loop)
            & (self.state == XHandMixerPanelManipulateState.DONE.value)
        )
        self.idx[done_final] = self.traj_len[done_final] - 1

    def _state_event_reached(self, obs, target_action: torch.Tensor) -> torch.Tensor:
        pose_states = (
            self.state == XHandMixerPanelManipulateState.APPROACH.value
        ) | (self.state == XHandMixerPanelManipulateState.RETREAT.value)
        final_lift_states = (
            self.state == XHandMixerPanelManipulateState.FINAL_LIFT.value
        )
        button_states = self.state == XHandMixerPanelManipulateState.PRESS.value
        wait_states = self.state == XHandMixerPanelManipulateState.WAIT_UI.value
        done_states = self.state == XHandMixerPanelManipulateState.DONE.value

        done = done_states.clone()
        if pose_states.any() or final_lift_states.any():
            tip_pose = self._get_index_tip_pose_w(obs)
            pos_err = torch.linalg.norm(tip_pose[:, :3] - target_action[:, :3], dim=-1)
            rot_err = quat_distance(
                F.normalize(tip_pose[:, 3:7], dim=-1),
                F.normalize(target_action[:, 3:7], dim=-1),
            )
            pose_reached = (pos_err <= self.pos_tol) & (rot_err <= self.rot_tol)
            done = done | (pose_states & pose_reached)
            # FINAL_LIFT should not complete early on intermediate trajectory waypoints.
            at_last_step = self.idx >= (self.traj_len - 1)
            done = done | (final_lift_states & at_last_step & pose_reached)

        if button_states.any():
            done = done | (button_states & self._target_button_is_pressed(obs))
        if wait_states.any():
            done = done | (wait_states & self._current_param_is_done())
        return done

    def _plan_approach(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        self._latch_target_button_ids(idxs)
        start_pose = self._get_index_tip_pose_w(obs)[idxs]
        target_pose = self._target_tip_pose_w(obs, self.approach_offset)[idxs]
        return self.plan_action_traj(
            obs, idxs, start_pose, target_pose, self.approach_steps
        )

    def _plan_press(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._get_index_tip_pose_w(obs)[idxs]
        target_pose = self._target_tip_pose_w(obs, self.press_offset)[idxs]
        return self.plan_action_traj(
            obs, idxs, start_pose, target_pose, self.press_steps
        )

    def _plan_wait_ui(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        hold_pose = self._target_tip_pose_w(obs, self.press_offset)[idxs]
        return self.plan_action_traj(obs, idxs, hold_pose, hold_pose, self.hold_steps)

    def _plan_retreat(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._get_index_tip_pose_w(obs)[idxs]
        target_pose = self._target_tip_pose_w(obs, self.retreat_offset)[idxs]
        return self.plan_action_traj(
            obs, idxs, start_pose, target_pose, self.retreat_steps
        )

    def _plan_done(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        tip_pose = self._get_index_tip_pose_w(obs)[idxs]
        return self.plan_action_traj(obs, idxs, tip_pose, tip_pose, 1)

    def _plan_final_lift(self, obs, idxs: torch.Tensor) -> torch.Tensor:
        start_pose = self._get_index_tip_pose_w(obs)[idxs]
        target_pose = start_pose.clone()
        target_pose[:, :3] = start_pose[:, :3] + self.final_lift_offset.unsqueeze(0)
        return self.plan_action_traj(
            obs, idxs, start_pose, target_pose, self.final_lift_steps
        )

    def plan_action_traj(
        self,
        obs,
        idxs: torch.Tensor,
        start_pose_w: torch.Tensor,
        target_pose_w: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        pos_traj, quat_traj = batched_interpolate_linear(
            start_pos=start_pose_w[:, :3],
            end_pos=target_pose_w[:, :3],
            start_quat=F.normalize(start_pose_w[:, 3:7], dim=-1),
            end_quat=F.normalize(target_pose_w[:, 3:7], dim=-1),
            num_steps=steps,
        )
        pose_traj = torch.cat((pos_traj, quat_traj), dim=-1)

        start_hand = self._extract_current_hand_action(obs)[idxs]
        target_hand = self.point_hand_action.unsqueeze(0).repeat(idxs.numel(), 1)
        hand_traj = qpos_linear_interpolate(start_hand, target_hand, steps)
        return torch.cat((pose_traj, hand_traj), dim=-1)

    def _target_tip_pose_w(self, obs, offset: torch.Tensor) -> torch.Tensor:
        button_pose = self._get_target_button_pose_w(obs)
        local_pose = torch.cat(
            (
                offset.unsqueeze(0).repeat(self.num_envs, 1),
                self.button_tip_rot_offset.unsqueeze(0).repeat(self.num_envs, 1),
            ),
            dim=-1,
        )
        return compose_pose_wxyz(button_pose, local_pose)

    def _get_target_button_pose_w(self, obs) -> torch.Tensor:
        button_poses = observation_tensor(
            obs, "mixer_button_sites_pose_w", self.device, self.num_envs
        )
        if button_poses.ndim == 2 and button_poses.shape[-1] % 7 == 0:
            button_poses = button_poses.reshape(self.num_envs, -1, 7)
        if button_poses.ndim != 3:
            raise ValueError(
                f"Expected mixer_button_sites_pose_w to have shape (B, N, 7), got {tuple(button_poses.shape)}."
            )
        button_poses = self._canonicalize_button_pose_order(button_poses)
        target_ids = self._get_target_button_ids()
        batch_idx = torch.arange(self.num_envs, device=self.device)
        pose = button_poses[batch_idx, target_ids, :7]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _canonicalize_button_pose_order(
        self, button_poses: torch.Tensor
    ) -> torch.Tensor:
        """Normalize pose order to [speed-up, speed-down, temp-up, temp-down, time-up, time-down]."""
        if button_poses.shape[1] != 6:
            return button_poses
        canonical = button_poses.clone()
        # SceneEntityCfg resolves these sites in MuJoCo model order (down, up), while
        # the policy and touch observations use semantic order (up, down). On this
        # slanted panel, each "up" button sits higher in world z.
        for pair_start in (0, 2, 4):
            pair = button_poses[:, pair_start : pair_start + 2, :]
            up_is_first = pair[:, 0, 2] >= pair[:, 1, 2]
            canonical[:, pair_start, :] = torch.where(
                up_is_first[:, None],
                pair[:, 0, :],
                pair[:, 1, :],
            )
            canonical[:, pair_start + 1, :] = torch.where(
                up_is_first[:, None],
                pair[:, 1, :],
                pair[:, 0, :],
            )
        return canonical

    def _compute_target_button_ids(self) -> torch.Tensor:
        current = self._get_ui_state_tensor()
        target = self._get_ui_target_tensor()
        param_idx = self.param_idx.clamp(0, 2)
        batch_idx = torch.arange(self.num_envs, device=self.device)
        current_param = current[batch_idx, param_idx]
        target_param = target[batch_idx, param_idx]
        increase = target_param > current_param

        button_ids = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        freq = param_idx == 0
        temp = param_idx == 1
        time = param_idx == 2
        button_ids[freq & increase] = 0
        button_ids[freq & (~increase)] = 1
        button_ids[temp & increase] = 2
        button_ids[temp & (~increase)] = 3
        button_ids[time & increase] = 4
        button_ids[time & (~increase)] = 5
        return button_ids

    def _ensure_target_button_ids(self) -> None:
        if self.target_button_ids is None:
            self.target_button_ids = torch.full(
                (self.num_envs,),
                -1,
                device=self.device,
                dtype=torch.long,
            )

    def _latch_target_button_ids(self, idxs: torch.Tensor) -> None:
        self._ensure_target_button_ids()
        computed = self._compute_target_button_ids()
        self.target_button_ids[idxs] = computed[idxs]

    def _sync_target_button_ids(self) -> None:
        if self.state is None:
            return
        self._ensure_target_button_ids()
        computed = self._compute_target_button_ids()
        stable = self.state == XHandMixerPanelManipulateState.PRESS.value
        missing = self.target_button_ids < 0
        sync = (~stable) | missing
        self.target_button_ids[sync] = computed[sync]

    def _get_target_button_ids(self) -> torch.Tensor:
        computed = self._compute_target_button_ids()
        if self.target_button_ids is None:
            return computed
        latched = self.target_button_ids
        stable = self.state == XHandMixerPanelManipulateState.PRESS.value
        use_latched = stable & (latched >= 0)
        return torch.where(use_latched, latched, computed)

    def _target_button_is_pressed(
        self,
        obs=None,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_ids is None:
            target_ids = self._get_target_button_ids()
        touch_values = self._get_button_touch_values(obs)
        if touch_values is not None:
            batch_idx = torch.arange(self.num_envs, device=self.device)
            return touch_values[batch_idx, target_ids] > 0.5

        sim = self.env.unwrapped.sim if hasattr(self.env, "unwrapped") else self.env.sim
        pressed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        for button_id, button_name in enumerate(MIXER_BUTTON_SITE_NAMES):
            env_mask = target_ids == button_id
            if not env_mask.any():
                continue
            adr_dim = self._sensor_adr_dim(sim, button_name)
            if adr_dim is None:
                continue
            adr, dim = adr_dim
            values = sim.data.sensordata[:, adr : adr + dim]
            values = torch.as_tensor(values, device=self.device)
            pressed[env_mask] = values[: self.num_envs, 0][env_mask] > 0.5
        return pressed

    def _target_button_sensor_raw(
        self, target_ids: torch.Tensor, obs=None
    ) -> torch.Tensor:
        touch_values = self._get_button_touch_values(obs)
        if touch_values is not None:
            batch_idx = torch.arange(self.num_envs, device=self.device)
            return touch_values[batch_idx, target_ids]

        sim = self.env.unwrapped.sim if hasattr(self.env, "unwrapped") else self.env.sim
        raw = torch.full(
            (self.num_envs,), float("nan"), device=self.device, dtype=torch.float32
        )
        for button_id, button_name in enumerate(MIXER_BUTTON_SITE_NAMES):
            env_mask = target_ids == button_id
            if not env_mask.any():
                continue
            adr_dim = self._sensor_adr_dim(sim, button_name)
            if adr_dim is None:
                continue
            adr, dim = adr_dim
            values = sim.data.sensordata[:, adr : adr + dim]
            values = torch.as_tensor(values, device=self.device, dtype=torch.float32)
            raw[env_mask] = values[: self.num_envs, 0][env_mask]
        return raw

    def _get_button_touch_values(self, obs) -> torch.Tensor | None:
        if obs is None:
            return None
        try:
            values = observation_tensor(
                obs, "mixer_button_touch", self.device, self.num_envs
            )
        except KeyError:
            return None
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]
        if values.ndim != 2:
            values = values.reshape(self.num_envs, -1)
        if values.shape[-1] < len(MIXER_BUTTON_SITE_NAMES):
            return None
        return values[:, : len(MIXER_BUTTON_SITE_NAMES)]

    def _sensor_adr_dim(self, sim, sensor_name: str) -> tuple[int, int] | None:
        for name in (sensor_name, f"mixer/{sensor_name}"):
            try:
                sensor = sim.mj_model.sensor(name)
                return int(sensor.adr[0]), int(sensor.dim[0])
            except KeyError:
                continue
        return None

    def _get_index_tip_pose_w(self, obs) -> torch.Tensor:
        pose = observation_tensor(obs, "index_tip_pose_w", self.device, self.num_envs)[
            :, :7
        ]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _update_tip_frames(self, obs, _target_action: torch.Tensor) -> None:
        if self.current_tip_frame is None or self.target_tip_frame is None:
            return

        current_pose = self._get_index_tip_pose_w(obs)[0, :7].detach().cpu()
        target_pose = (
            self._target_tip_pose_w(obs, torch.zeros(3, device=self.device))[0, :7]
            .detach()
            .cpu()
        )
        sensor_pose = self._get_target_button_pose_w(obs)[0, :7].detach().cpu()
        self.current_tip_frame.position = current_pose[:3].numpy()
        self.current_tip_frame.wxyz = current_pose[3:7].numpy()
        self.target_tip_frame.position = target_pose[:3].numpy()
        self.target_tip_frame.wxyz = target_pose[3:7].numpy()
        if self.sensor_frame is not None:
            self.sensor_frame.position = sensor_pose[:3].numpy()
            self.sensor_frame.wxyz = sensor_pose[3:7].numpy()

    def _update_debug_text(self, obs) -> None:
        if self.debug_text is None:
            return
        if self.state is None or self.param_idx is None:
            return

        current = self._get_ui_state_tensor()
        target = self._get_ui_target_tensor()
        button_ids = self._get_target_button_ids()
        computed_button_ids = self._compute_target_button_ids()
        latched_button_ids = self.target_button_ids
        all_sensor_raw = self._get_button_touch_values(obs)
        if all_sensor_raw is not None:
            batch_idx = torch.arange(self.num_envs, device=self.device)
            sensor_raw = all_sensor_raw[batch_idx, button_ids]
            pressed = sensor_raw > 0.5
        else:
            sensor_raw = self._target_button_sensor_raw(button_ids, obs)
            pressed = self._target_button_is_pressed(obs, target_ids=button_ids)
        env_id = 0
        state_name = XHandMixerPanelManipulateState(
            int(self.state[env_id].item())
        ).name
        all_state_names = [
            XHandMixerPanelManipulateState(int(v)).name
            for v in self.state.detach().cpu().tolist()
        ]
        param_id = int(self.param_idx[env_id].item())
        button_id = int(button_ids[env_id].item())
        computed_button_id = int(computed_button_ids[env_id].item())
        latched_button_id = (
            int(latched_button_ids[env_id].item())
            if latched_button_ids is not None
            else -1
        )
        button_name = MIXER_BUTTON_SITE_NAMES[button_id]
        is_pressed = bool(pressed[env_id].item())
        raw_val = float(sensor_raw[env_id].item())
        if all_sensor_raw is not None:
            raw_values = all_sensor_raw[env_id].detach().cpu().tolist()
            raw_line = (
                "raw [spd_up,spd_down,temp_up,temp_down,time_up,time_down] = "
                f"[{raw_values[0]:.2f}, {raw_values[1]:.2f}, {raw_values[2]:.2f}, "
                f"{raw_values[3]:.2f}, {raw_values[4]:.2f}, {raw_values[5]:.2f}]"
            )
        else:
            raw_line = (
                "raw [spd_up,spd_down,temp_up,temp_down,time_up,time_down] = <missing>"
            )

        cur_freq, cur_temp, cur_time = current[env_id].detach().cpu().tolist()
        tar_freq, tar_temp, tar_time = target[env_id].detach().cpu().tolist()
        self.debug_text.value = (
            f"env={env_id} state={state_name} param_idx={param_id}\n"
            f"all_env_states={all_state_names}\n"
            f"active_button_id={button_id} latched_button_id={latched_button_id} "
            f"computed_button_id={computed_button_id}\n"
            f"button={button_name} pressed={is_pressed} sensor_raw={raw_val:.4f}\n"
            f"{raw_line}\n"
            f"current [freq,temp,time] = [{cur_freq:.1f}, {cur_temp:.1f}, {cur_time:.1f}]\n"
            f"target  [freq,temp,time] = [{tar_freq:.1f}, {tar_temp:.1f}, {tar_time:.1f}]"
        )

    def _skip_completed_params(self) -> None:
        if self.param_idx is None or self.state is None or self.traj_len is None:
            return
        for _ in range(3):
            # Fast-forward is only for parameter-selection phases and only for the
            # first two parameters (freq/temp). The last parameter (time, idx=2)
            # must still go through WAIT_UI -> RETREAT -> FINAL_LIFT in advance_state().
            active = (
                (self.state == XHandMixerPanelManipulateState.APPROACH.value)
                | (self.state == XHandMixerPanelManipulateState.PRESS.value)
                | (self.state == XHandMixerPanelManipulateState.WAIT_UI.value)
            )
            skippable_param = self.param_idx < 2
            done = active & skippable_param & self._current_param_is_done()
            if not done.any():
                return
            self.param_idx[done] += 1
            # Never set DONE here. DONE must only be reached after FINAL_LIFT via
            # advance_state(), so we always restart from APPROACH for fast-forwarded
            # first/second parameters.
            self.state[done] = XHandMixerPanelManipulateState.APPROACH.value
            self.traj_len[done] = 0
            self.idx[done] = 0
            self.target_button_ids[done] = -1

    def _current_param_is_done(self) -> torch.Tensor:
        current = self._get_ui_state_tensor()
        target = self._get_ui_target_tensor()
        param_idx = self.param_idx.clamp(0, 2)
        batch_idx = torch.arange(self.num_envs, device=self.device)
        # Match dexbiolab thermal mixer expert logic: only done when value is exactly at target.
        return current[batch_idx, param_idx] == target[batch_idx, param_idx]

    def _get_ui_state_tensor(self) -> torch.Tensor:
        ui_state = get_mixer_ui_state(self.env)
        if ui_state is None:
            return (
                torch.tensor(
                    [900.0, 25.0, 60.0],
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
        return torch.stack(
            (
                ui_state["freq_rpm"][: self.num_envs],
                ui_state["temp_c"][: self.num_envs],
                ui_state["time_s"][: self.num_envs],
            ),
            dim=-1,
        ).to(device=self.device, dtype=torch.float32)

    def _get_ui_target_tensor(self) -> torch.Tensor:
        target = getattr(self.env, "mixer_ui_target_state", None)
        if target is None:
            return (
                torch.tensor(
                    [900.0, 25.0, 60.0],
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
        target = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        if target.ndim == 1:
            target = target.unsqueeze(0).repeat(self.num_envs, 1)
        target = target[: self.num_envs]
        # Env reset stores [time_s, temp_c, freq_rpm]; this policy adjusts [freq, temp, time].
        return target[:, [2, 1, 0]]

    def _extract_current_hand_action(self, obs) -> torch.Tensor:
        qpos = observation_tensor(obs, "current_qpos", self.device, self.num_envs)
        return qpos[:, -self.hand_dim :]

    def _build_point_hand_action(self) -> torch.Tensor:
        targets = {
            "thumb_bend_joint": 0.35,
            "thumb_rota_joint1": 0.35,
            "thumb_rota_joint2": 0.35,
            "index_bend_joint": 0.0,
            "index_joint1": 0.0,
            "index_joint2": 0.0,
            "mid_joint1": 1.20,
            "mid_joint2": 1.10,
            "ring_joint1": 1.20,
            "ring_joint2": 1.10,
            "pinky_joint1": 1.20,
            "pinky_joint2": 1.10,
        }
        values = []
        for joint_expr in XHAND_JOINT_ORDER:
            suffix = joint_expr.replace(".*", "")
            values.append(targets[suffix])
        return torch.tensor(values, device=self.device, dtype=torch.float32)


def make_policy(*, action_shape, device, env, server=None, replay_path=None):
    del replay_path
    return XHandMixerPanelManipulatePolicy(
        action_shape=action_shape,
        device=device,
        env=env,
        loop=False,
        server=server,
    )
