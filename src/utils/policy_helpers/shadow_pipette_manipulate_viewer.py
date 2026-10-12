"""ShadowPipetteViewer: extracted viewer behavior for shadow_pipette_manipulate."""

from __future__ import annotations

import torch

from src.utils.policy_helpers.shadow_pipette_manipulate_common import (
    ShadowPipetteManipulateState,
)
from src.tasks.success_diagnostics import success_diagnostics_for_envs
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
)


class ShadowPipetteViewer:
    """Provide viewer methods used by the task policy."""

    def _update_success_debug_text(self) -> None:
        if self.success_debug_text is None:
            return

        snapshot = success_diagnostics_for_envs(self.env, [0]).get(0)
        if snapshot is None:
            self.success_debug_text.value = "Waiting for first environment step..."
            return

        components = snapshot["components"]
        metrics = snapshot["metrics"]
        thresholds = snapshot["thresholds"]
        radial_dist = metrics["tip_cryovial_radial_dist_m"]
        axial_offset = metrics["tip_cryovial_axial_offset_m"]
        axis_tilt = metrics["tip_cryovial_axis_tilt_rad"]
        stable_steps = metrics["tip_inserted_stable_steps"]
        button_qpos = metrics["pipette_button_qpos_m"]
        max_radial_dist = thresholds["max_tip_cryovial_radial_dist_m"]
        min_axial_offset = thresholds["min_tip_cryovial_axial_offset_m"]
        max_axial_offset = thresholds["max_tip_cryovial_axial_offset_m"]
        max_axis_tilt = thresholds["max_tip_cryovial_axis_tilt_rad"]
        min_stable_steps = thresholds["min_tip_inserted_stable_steps"]
        max_button_qpos = thresholds["max_pressed_button_qpos_m"]

        def result(value: bool) -> str:
            return "PASS" if value else "FAIL"

        self.success_debug_text.value = (
            f"radial: {result(components['tip_radially_aligned'])} | "
            f"{radial_dist:.4f} <= {max_radial_dist:.4f} m\n"
            f"depth: {result(components['tip_depth_inserted'])} | "
            f"{min_axial_offset:.4f} <= {axial_offset:.4f} "
            f"<= {max_axial_offset:.4f} m\n"
            f"axis: {result(components['tip_axis_aligned'])} | "
            f"tilt={axis_tilt:.4f} <= {max_axis_tilt:.4f} rad\n"
            f"inserted_stable: {result(components['tip_inserted_stable'])} | "
            f"steps={stable_steps:.0f}/{min_stable_steps:.0f}\n"
            f"button_pressed_now: "
            f"{result(components['pipette_button_pressed_now'])} | "
            f"qpos={button_qpos:+.4f} < {max_button_qpos:+.4f} m\n"
            f"pressed_while_inserted_ever: "
            f"{result(components['pipette_button_pressed_while_inserted_ever'])}\n"
            f"released_while_inserted_now: "
            f"{result(components['pipette_button_released_while_inserted_now'])}\n"
            f"full_success: {result(components['full_success'])}"
        )

    def _update_target_frame(
        self,
        target_pose_w: torch.Tensor,
        idxs: torch.Tensor,
    ) -> None:
        if self.target_frame is None:
            return

        env0 = torch.where(idxs == 0)[0]
        pose_idx = int(env0[0].item()) if env0.numel() > 0 else 0
        target_pose = target_pose_w[pose_idx, :7].detach().cpu()
        self.target_frame.position = target_pose[:3].numpy()
        self.target_frame.wxyz = target_pose[3:7].numpy()

    def _update_cell_dish_frame(self, obs) -> None:
        if self.cell_dish_frame is None:
            return

        try:
            cell_dish_pose = self._get_cell_dish_pose_w(obs)[0, :7].detach().cpu()
        except KeyError:
            return

        self.cell_dish_frame.position = cell_dish_pose[:3].numpy()
        self.cell_dish_frame.wxyz = cell_dish_pose[3:7].numpy()

    def _update_cryovial_frame(self, obs) -> None:
        if self.cryovial_frame is None:
            return

        try:
            cryovial_pose = self._get_cryovial_pose_w(obs)[0, :7].detach().cpu()
        except KeyError:
            return

        self.cryovial_frame.position = cryovial_pose[:3].numpy()
        self.cryovial_frame.wxyz = cryovial_pose[3:7].numpy()

    def _update_stage_text(self) -> None:
        if self.stage_text is not None:
            self.stage_text.value = self._format_stage_text()

    def _format_stage_text(self) -> str:
        if self.state is None or self.state.numel() == 0:
            return "UNINITIALIZED"

        stage_values = self.state.detach().cpu().tolist()
        stage_names = [
            ShadowPipetteManipulateState(int(value)).name for value in stage_values
        ]
        env0_stage = stage_names[0]
        if len(stage_names) == 1 or all(name == env0_stage for name in stage_names):
            return self._format_error_text(env0_stage)

        summary = []
        for state in ShadowPipetteManipulateState:
            count = stage_names.count(state.name)
            if count > 0:
                summary.append(f"{state.name}: {count}")
        return self._format_error_text(f"env0: {env0_stage}\n" + "\n".join(summary))

    def _format_error_text(self, stage_text: str) -> str:
        if self._latest_pos_err is None or self._latest_rot_err is None:
            return "\n".join((stage_text, self._format_grasp_offset_text()))

        pos_err = self._latest_pos_err.detach().cpu()
        rot_err = self._latest_rot_err.detach().cpu()
        env0_pos_err = float(pos_err[0].item())
        env0_rot_err = float(rot_err[0].item())

        pos_tol, rot_tol = self._target_reach_tolerance()
        env0_pos_tol = float(pos_tol.detach().cpu()[0].item())
        env0_rot_tol = float(rot_tol.detach().cpu()[0].item())

        lines = [
            stage_text,
            self._format_grasp_offset_text(),
            f"env0 pos_err: {env0_pos_err:.5f} m / tol {env0_pos_tol:.5f}",
            f"env0 rot_err: {env0_rot_err:.5f} rad / tol {env0_rot_tol:.5f}",
        ]

        if pos_err.numel() > 1:
            lines.extend(
                (
                    f"max pos_err: {float(pos_err.max().item()):.5f} m",
                    f"max rot_err: {float(rot_err.max().item()):.5f} rad",
                )
            )

        return "\n".join(lines)

    def _format_grasp_offset_text(self) -> str:
        offset = self._current_grasp_pos_offset().detach().cpu()
        return (
            "grasp offset xyz: "
            f"{float(offset[0].item()):+.3f}, "
            f"{float(offset[1].item()):+.3f}, "
            f"{float(offset[2].item()):+.3f} m"
        )

    def _log_current_grasp_rel_pose(
        self,
        obs,
        idxs: torch.Tensor,
        palm_pose_w: torch.Tensor,
    ) -> None:
        if not self.log_grasp_rel_pose or self._logged_grasp_rel_pose:
            return

        env0 = torch.where(idxs == 0)[0]
        pose_idx = int(env0[0].item()) if env0.numel() > 0 else 0
        pipette_pose_w = self._get_pipette_pose_w(obs)[idxs][pose_idx : pose_idx + 1]
        palm_pose_w = palm_pose_w[pose_idx : pose_idx + 1]
        mocap_pose_w = compose_pose_wxyz(
            palm_pose_w, pose_inv_wxyz(self.mocap_to_palm_pose.unsqueeze(0))
        )
        rel_pose = (
            compose_pose_wxyz(pose_inv_wxyz(pipette_pose_w), mocap_pose_w)[0]
            .detach()
            .cpu()
        )
        pos = tuple(float(v) for v in rel_pose[:3])
        quat = tuple(float(v) for v in rel_pose[3:7])
        print(
            "Measured pipette-relative grasp pose for env0: "
            f"SHADOW_INIT_REL_POS = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}); "
            f"SHADOW_INIT_REL_QUAT = ({quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f})"
        )
        self._logged_grasp_rel_pose = True

    def _current_grasp_pos_offset(self) -> torch.Tensor:
        offset = torch.zeros(3, device=self.device, dtype=torch.float32)
        if not self.grasp_offset_sliders:
            return offset

        offset_values = [
            float(self.grasp_offset_sliders[axis].value) for axis in ("x", "y", "z")
        ]
        return torch.tensor(offset_values, device=self.device, dtype=torch.float32)
