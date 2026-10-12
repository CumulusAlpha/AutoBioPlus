"""ThermalCyclerMotionPlans: extracted plans behavior for xhand_thermalcycler_close."""

from __future__ import annotations

import torch

from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
)


class ThermalCyclerMotionPlans:
    """Provide plans methods used by the task policy."""

    def _plan_close(
        self,
        palm_pose: torch.Tensor,
        lever_pose: torch.Tensor,
        hand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str | None], torch.Tensor]:
        close_pregrasp_hand = self._pregrasp_hand("close_start")
        close_start_hand = self._grip_hand("close_start")
        close_end_hand = self._grip_hand("close_end")
        spinner_pregrasp_hand = self._pregrasp_hand("spinner")
        spinner_grip_hand = self._grip_hand("spinner")
        spinner_preclose_hand = spinner_pregrasp_hand + 0.55 * (
            spinner_grip_hand - spinner_pregrasp_hand
        )
        spinner_release_hand = spinner_pregrasp_hand + 0.65 * (
            spinner_grip_hand - spinner_pregrasp_hand
        )

        lever_detach = self._lever_pose_to_palm_pose(lever_pose, mode="side_detach")
        lever_grip = self._lever_pose_to_palm_pose(lever_pose, mode="side_grip")
        lever_grip_backoff = self._lever_pose_to_palm_pose(
            lever_pose, mode="side_grip_backoff"
        )
        palm_clear = self._with_world_delta(
            palm_pose, (0.0, 0.0, self._APPROACH_CLEARANCE_Z)
        )
        lever_clear = self._with_world_delta(
            lever_detach, (0.0, 0.0, self._TARGET_CLEARANCE_Z)
        )
        side_grip_local = self._lever_local_pose("side_grip")
        side_grip_backoff_local = self._lever_local_pose("side_grip_backoff")
        close_end_grip_local = self._lever_local_pose("close_end_grip")
        lid_steps = self._CLOSE_LID_STEPS
        lid_settle_steps = self._CLOSE_LID_SETTLE_STEPS
        latch_steps = self._CLOSE_LATCH_STEPS
        spin_steps = self._OPEN_SPINNER_STEPS
        post_spin_lift_steps = 35

        open_q = self._cycler_qpos((0.0, 0.0, 0.0))
        lid_closed_q = self._cycler_qpos((-1.9, 0.0, 0.0))
        latched_q = self._cycler_qpos((-1.9, -0.94, 0.0))
        locked_latched_q = latched_q.clone()
        locked_latched_q[:, 2] = self._SPINNER_LOCKED_QPOS

        approach_traj = self._concat_segments(
            (palm_pose, palm_clear, hand, close_pregrasp_hand, 45),
            (palm_clear, lever_clear, close_pregrasp_hand, close_pregrasp_hand, 65),
            (lever_clear, lever_detach, close_pregrasp_hand, close_pregrasp_hand, 45),
            (
                lever_detach,
                lever_grip_backoff,
                close_pregrasp_hand,
                close_pregrasp_hand,
                35,
            ),
            (
                lever_grip_backoff,
                lever_grip_backoff,
                close_pregrasp_hand,
                close_start_hand,
                45,
            ),
            (lever_grip_backoff, lever_grip, close_start_hand, close_start_hand, 25),
        )
        cycler_traj = self._concat_joint_segments(
            (open_q, open_q, 45),
            (open_q, open_q, 65),
            (open_q, open_q, 45),
            (open_q, open_q, 35),
            (open_q, open_q, 45),
            (open_q, open_q, 25),
            (open_q, lid_closed_q, lid_steps),
            (lid_closed_q, lid_closed_q, lid_settle_steps),
            (lid_closed_q, latched_q, latch_steps),
        )
        lever_pose_traj = self._cycler_lever_pose_for_qpos_traj(cycler_traj)
        close_motion_hand_traj = self._concat_joint_segments(
            (close_start_hand, close_start_hand, lid_steps + lid_settle_steps),
            (close_start_hand, close_end_hand, latch_steps),
        )[:, 1:]
        close_motion_lever_traj = lever_pose_traj[
            :,
            approach_traj.shape[1] : approach_traj.shape[1]
            + close_motion_hand_traj.shape[1],
        ]
        latched_close_end_grip = compose_pose_wxyz(
            close_motion_lever_traj[:, -1], close_end_grip_local
        )
        close_motion_pose_traj = self._lever_guided_world_pose_traj(
            close_motion_lever_traj,
            lever_grip,
            latched_close_end_grip,
        )
        latched_lever_pose = lever_pose_traj[:, -1]
        open_start_spinner_grip = compose_pose_wxyz(
            latched_lever_pose,
            self._lever_local_pose("spinner_grip"),
        )
        knob_start_pose = self._cycler_knob_pose_for_qpos_traj(latched_q[:, None, :])[
            :, 0
        ]
        knob_end_pose = self._cycler_knob_pose_for_qpos_traj(
            locked_latched_q[:, None, :]
        )[:, 0]
        spinner_grip_local_to_knob = compose_pose_wxyz(
            pose_inv_wxyz(knob_end_pose),
            open_start_spinner_grip,
        )
        spinner_grip = compose_pose_wxyz(knob_start_pose, spinner_grip_local_to_knob)
        spinner_clear = self._with_world_delta(
            spinner_grip, (0.0, 0.0, self._SPINNER_CLEARANCE_Z)
        )
        spinner_near = self._with_world_delta(spinner_grip, (0.0, 0.0, 0.025))
        lever_release_lift = self._with_world_delta(
            latched_close_end_grip,
            (0.0, 0.0, self._TARGET_CLEARANCE_Z),
        )
        spinner_transition_traj = self._concat_segments(
            (
                latched_close_end_grip,
                lever_release_lift,
                close_end_hand,
                close_end_hand,
                45,
            ),
            (
                lever_release_lift,
                lever_release_lift,
                close_end_hand,
                spinner_pregrasp_hand,
                15,
            ),
            (
                lever_release_lift,
                spinner_clear,
                spinner_pregrasp_hand,
                spinner_preclose_hand,
                65,
            ),
            (spinner_clear, spinner_near, spinner_preclose_hand, spinner_grip_hand, 35),
            (spinner_near, spinner_grip, spinner_grip_hand, spinner_grip_hand, 10),
            (spinner_grip, spinner_grip, spinner_grip_hand, spinner_grip_hand, 30),
        )[:, 1:]

        spinner_spin_qpos_traj = self._concat_joint_segments(
            (latched_q, locked_latched_q, spin_steps),
        )[:, 1:]
        knob_spin_pose_traj = self._cycler_knob_pose_for_qpos_traj(
            spinner_spin_qpos_traj
        )
        spinner_spin_pose_traj = self._compose_lever_local_pose_traj(
            knob_spin_pose_traj,
            spinner_grip_local_to_knob[:, None, :].repeat(1, spin_steps, 1),
        )
        spinner_spin_hand_traj = self._concat_joint_segments(
            (spinner_grip_hand, spinner_grip_hand, spin_steps),
        )[:, 1:]
        spinner_spin_traj = torch.cat(
            (spinner_spin_pose_traj, spinner_spin_hand_traj), dim=-1
        )
        spinner_end_pose = spinner_spin_pose_traj[:, -1]
        spinner_lift = self._with_world_delta(
            spinner_end_pose,
            (0.0, 0.0, self._OPEN_POST_SPINNER_LIFT_Z),
        )
        spinner_post_traj = self._concat_segments(
            (
                spinner_end_pose,
                spinner_lift,
                spinner_grip_hand,
                spinner_release_hand,
                post_spin_lift_steps,
            ),
        )[:, 1:]

        post_approach_traj = torch.cat(
            (
                torch.cat((close_motion_pose_traj, close_motion_hand_traj), dim=-1),
                spinner_transition_traj,
                spinner_spin_traj,
                spinner_post_traj,
            ),
            dim=1,
        )
        action_traj = torch.cat((approach_traj, post_approach_traj), dim=1)
        cycler_traj = self._concat_joint_segments(
            (open_q, open_q, 45),
            (open_q, open_q, 65),
            (open_q, open_q, 45),
            (open_q, open_q, 35),
            (open_q, open_q, 45),
            (open_q, open_q, 25),
            (open_q, lid_closed_q, lid_steps),
            (lid_closed_q, lid_closed_q, lid_settle_steps),
            (lid_closed_q, latched_q, latch_steps),
            (latched_q, latched_q, 45),
            (latched_q, latched_q, 15),
            (latched_q, latched_q, 65),
            (latched_q, latched_q, 35),
            (latched_q, latched_q, 10),
            (latched_q, latched_q, 30),
            (latched_q, locked_latched_q, spin_steps),
            (locked_latched_q, locked_latched_q, post_spin_lift_steps),
        )
        follow_modes = self._concat_follow_modes(
            (None, 45),
            (None, 65),
            (None, 45),
            (None, 35),
            (None, 45),
            (None, 25),
            (None, lid_steps),
            (None, lid_settle_steps),
            (None, latch_steps),
            (None, 45),
            (None, 15),
            (None, 65),
            (None, 35),
            (None, 10),
            (None, 30),
            (None, spin_steps),
            (None, post_spin_lift_steps),
        )
        approach_local_traj = self._concat_local_follow_segments(
            (None, None, 45),
            (None, None, 65),
            (None, None, 45),
            (None, None, 35),
            (side_grip_backoff_local, side_grip_backoff_local, 45),
            (side_grip_backoff_local, side_grip_local, 25),
        )
        post_approach_local_traj = self._concat_local_follow_segments(
            (None, None, lid_steps),
            (None, None, lid_settle_steps),
            (None, None, latch_steps),
            (None, None, 45),
            (None, None, 15),
            (None, None, 65),
            (None, None, 35),
            (None, None, 10),
            (None, None, 30),
            (None, None, spin_steps),
            (None, None, post_spin_lift_steps),
        )[:, 1:]
        follow_local_traj = torch.cat(
            (approach_local_traj, post_approach_local_traj), dim=1
        )
        return action_traj, cycler_traj, follow_modes, follow_local_traj

    def _plan_open(
        self,
        palm_pose: torch.Tensor,
        lever_pose: torch.Tensor,
        knob_pose: torch.Tensor,
        cycler_qpos: torch.Tensor,
        hand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str | None], torch.Tensor]:
        spinner_pregrasp_hand = self._pregrasp_hand("spinner")
        spinner_grip_hand = self._grip_hand("spinner")
        spinner_preclose_hand = spinner_pregrasp_hand + 0.55 * (
            spinner_grip_hand - spinner_pregrasp_hand
        )
        spinner_release_hand = spinner_pregrasp_hand + 0.65 * (
            spinner_grip_hand - spinner_pregrasp_hand
        )
        open_insert_hand = self._grip_hand("open_insert")
        open_start_hand = self._grip_hand("open_start")
        close_start_hand = self._grip_hand("close_start")

        spinner_grip = self._lever_pose_to_palm_pose(lever_pose, mode="spinner_grip")
        spinner_clear = self._with_world_delta(
            spinner_grip, (0.0, 0.0, self._SPINNER_CLEARANCE_Z)
        )
        spinner_near = self._with_world_delta(spinner_grip, (0.0, 0.0, 0.025))
        close_end_grip = self._lever_pose_to_palm_pose(
            lever_pose, mode="close_end_grip"
        )
        palm_clear = self._with_world_delta(
            palm_pose, (0.0, 0.0, self._APPROACH_CLEARANCE_Z)
        )
        close_end_over_handle = close_end_grip.clone()
        close_end_insert_ready = self._with_world_delta(
            close_end_grip,
            (0.0, 0.0, self._OPEN_INSERT_READY_Z),
        )
        side_grip_local = self._lever_local_pose("side_grip")
        spin_steps = self._OPEN_SPINNER_STEPS
        post_spin_lift_steps = 35
        post_spin_transition_steps = 90
        latch_steps = self._CLOSE_LATCH_STEPS
        lid_settle_steps = self._CLOSE_LID_SETTLE_STEPS
        lid_steps = self._CLOSE_LID_STEPS

        latched_q = cycler_qpos.to(device=self.device, dtype=torch.float32).clone()
        spun_latched_q = latched_q.clone()
        spun_latched_q[:, 2] = self._SPINNER_UNLOCKED_QPOS
        lid_closed_q = spun_latched_q.clone()
        lid_closed_q[:, 0] = -1.9
        lid_closed_q[:, 1] = 0.0
        open_q = spun_latched_q.clone()
        open_q[:, 0] = 0.0
        open_q[:, 1] = 0.0

        spinner_approach_traj = self._concat_segments(
            (palm_pose, palm_clear, hand, spinner_pregrasp_hand, 45),
            (
                palm_clear,
                spinner_clear,
                spinner_pregrasp_hand,
                spinner_preclose_hand,
                65,
            ),
            (spinner_clear, spinner_near, spinner_preclose_hand, spinner_grip_hand, 35),
            (spinner_near, spinner_grip, spinner_grip_hand, spinner_grip_hand, 10),
        )
        spinner_spin_qpos_traj = self._concat_joint_segments(
            (latched_q, spun_latched_q, spin_steps),
        )[:, 1:]
        knob_spin_pose_traj = self._cycler_knob_pose_for_qpos_traj(
            spinner_spin_qpos_traj
        )
        spinner_grip_local_to_knob = compose_pose_wxyz(
            pose_inv_wxyz(knob_pose), spinner_grip
        )
        spinner_spin_pose_traj = self._compose_lever_local_pose_traj(
            knob_spin_pose_traj,
            spinner_grip_local_to_knob[:, None, :].repeat(1, spin_steps, 1),
        )
        spinner_spin_hand_traj = self._concat_joint_segments(
            (spinner_grip_hand, spinner_grip_hand, spin_steps),
        )[:, 1:]
        spinner_spin_traj = torch.cat(
            (spinner_spin_pose_traj, spinner_spin_hand_traj), dim=-1
        )
        spinner_traj = torch.cat((spinner_approach_traj, spinner_spin_traj), dim=1)
        spinner_end_pose = spinner_spin_pose_traj[:, -1]
        transfer_z = torch.maximum(
            spinner_end_pose[:, 2] - self._OPEN_PREINSERT_LOWER_Z,
            close_end_insert_ready[:, 2],
        )
        close_end_over_handle[:, 2] = transfer_z
        spinner_lift = self._with_world_delta(
            spinner_end_pose,
            (0.0, 0.0, self._OPEN_POST_SPINNER_LIFT_Z),
        )
        close_end_over_handle[:, 2] = torch.maximum(
            close_end_over_handle[:, 2], spinner_lift[:, 2]
        )

        approach_traj = self._concat_segments(
            (
                spinner_end_pose,
                spinner_lift,
                spinner_grip_hand,
                spinner_release_hand,
                post_spin_lift_steps,
            ),
            (
                spinner_lift,
                close_end_over_handle,
                spinner_release_hand,
                open_insert_hand,
                post_spin_transition_steps,
            ),
            (
                close_end_over_handle,
                close_end_insert_ready,
                open_insert_hand,
                open_insert_hand,
                45,
            ),
            (
                close_end_insert_ready,
                close_end_grip,
                open_insert_hand,
                open_insert_hand,
                35,
            ),
            (close_end_grip, close_end_grip, open_insert_hand, open_start_hand, 45),
        )
        post_spinner_approach_traj = approach_traj[:, 1:]
        cycler_traj = self._concat_joint_segments(
            (latched_q, latched_q, 45),
            (latched_q, latched_q, 65),
            (latched_q, latched_q, 45),
            (latched_q, spun_latched_q, spin_steps),
            (spun_latched_q, spun_latched_q, post_spin_lift_steps),
            (spun_latched_q, spun_latched_q, post_spin_transition_steps),
            (spun_latched_q, spun_latched_q, 45),
            (spun_latched_q, spun_latched_q, 35),
            (spun_latched_q, spun_latched_q, 45),
            (spun_latched_q, lid_closed_q, latch_steps),
            (lid_closed_q, lid_closed_q, lid_settle_steps),
            (lid_closed_q, open_q, lid_steps),
        )
        lever_pose_traj = self._cycler_lever_pose_for_qpos_traj(cycler_traj)
        if lever_pose_traj.shape[1] != cycler_traj.shape[1]:
            raise RuntimeError(
                "Open lever trajectory length does not match cycler trajectory: "
                f"{lever_pose_traj.shape[1]} != {cycler_traj.shape[1]}"
            )
        open_motion_hand_traj = self._concat_joint_segments(
            (open_start_hand, close_start_hand, latch_steps),
            (close_start_hand, close_start_hand, lid_settle_steps + lid_steps),
        )[:, 1:]
        open_motion_lever_traj = lever_pose_traj[
            :,
            spinner_traj.shape[1]
            + post_spinner_approach_traj.shape[1] : spinner_traj.shape[1]
            + post_spinner_approach_traj.shape[1]
            + open_motion_hand_traj.shape[1],
        ]
        opened_side_grip = compose_pose_wxyz(
            open_motion_lever_traj[:, -1], side_grip_local
        )
        open_motion_pose_traj = self._lever_guided_world_pose_traj(
            open_motion_lever_traj,
            close_end_grip,
            opened_side_grip,
        )
        open_motion_traj = torch.cat(
            (open_motion_pose_traj, open_motion_hand_traj), dim=-1
        )

        action_traj = torch.cat(
            (spinner_traj, post_spinner_approach_traj, open_motion_traj), dim=1
        )
        follow_modes = self._concat_follow_modes(
            (None, 45),
            (None, 65),
            (None, 45),
            (None, spin_steps),
            (None, post_spin_lift_steps),
            (None, post_spin_transition_steps),
            (None, 45),
            (None, 35),
            (None, 45),
            (None, latch_steps),
            (None, lid_settle_steps),
            (None, lid_steps),
        )
        follow_local_traj = self._concat_local_follow_segments(
            (None, None, 45),
            (None, None, 65),
            (None, None, 45),
            (None, None, spin_steps),
            (None, None, post_spin_lift_steps),
            (None, None, post_spin_transition_steps),
            (None, None, 45),
            (None, None, 35),
            (None, None, 45),
            (None, None, latch_steps),
            (None, None, lid_settle_steps),
            (None, None, lid_steps),
        )
        return action_traj, cycler_traj, follow_modes, follow_local_traj
