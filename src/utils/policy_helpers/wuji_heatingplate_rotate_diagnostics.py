"""HeatingPlateDiagnostics: extracted diagnostics behavior for wuji_heatingplate_rotate."""

from __future__ import annotations

import torch

from src.tasks.success_diagnostics import success_diagnostics_for_envs
from src.utils.quat_util import (
    quat_rotate_vec,
)


class HeatingPlateDiagnostics:
    """Provide diagnostics methods used by the task policy."""

    def _update_success_debug_text(self) -> None:
        if self.success_debug_text is None:
            return

        snapshot = success_diagnostics_for_envs(self.env, [0]).get(0)
        if snapshot is None:
            self.success_debug_text.value = "Waiting for first environment step..."
            return

        components = snapshot["components"]
        metrics = snapshot["metrics"]
        metrics_max = snapshot["metrics_max"]
        thresholds = snapshot["thresholds"]
        xy_dist = metrics["beaker_plate_xy_dist_m"]
        height = metrics["beaker_plate_height_m"]
        knob_qpos = metrics["knob_qpos_rad"]
        max_knob_qpos = metrics_max["knob_qpos_rad"]
        hand_lift = metrics["hand_lift_m"]
        max_xy_dist = thresholds["max_plate_xy_dist_m"]
        min_height = thresholds["min_plate_height_m"]
        max_height = thresholds["max_plate_height_m"]
        min_rotation = thresholds["min_rotation_rad"]
        home_tolerance = thresholds["home_tolerance_rad"]
        min_hand_lift = thresholds["min_hand_lift_m"]

        def result(value: bool) -> str:
            return "PASS" if value else "FAIL"

        centered = xy_dist <= max_xy_dist
        height_ok = min_height <= height <= max_height
        self.success_debug_text.value = (
            f"centered: {result(centered)} | xy={xy_dist:.4f} <= {max_xy_dist:.4f} m\n"
            f"above_plate: {result(height_ok)} | z={height:.4f} in "
            f"[{min_height:.4f}, {max_height:.4f}] m\n"
            f"beaker_on_plate: {result(components['beaker_on_plate'])} | "
            "centered AND above_plate\n"
            f"knob_turned_on_seen: {result(components['knob_turned_on_seen'])} | "
            f"episode max={max_knob_qpos:+.4f} >= {min_rotation:.4f} rad\n"
            f"knob_at_home: {result(components['knob_at_home'])} | seen AND "
            f"abs({knob_qpos:+.4f}) <= {home_tolerance:.4f} rad\n"
            f"object_complete: {result(components['object_complete'])} | "
            "beaker_on_plate AND knob_at_home\n"
            f"previous_task_complete_seen: "
            f"{result(components['previous_task_complete_seen'])}\n"
            f"hand_lifted: {result(components['hand_lifted'])} | "
            f"relative dz={hand_lift:.4f} >= {min_hand_lift:.4f} m above knob "
            "after previous task\n"
            f"full_success: {result(components['full_success'])}"
        )

    def _init_contact_force_debug(self) -> None:
        try:
            model = self.env.sim.mj_model
        except Exception:
            return

        for geom_id in range(model.ngeom):
            geom_name = model.geom(geom_id).name or str(geom_id)
            body_id = int(model.geom_bodyid[geom_id])
            body_name = model.body(body_id).name or str(body_id)
            self._geom_id_to_name[geom_id] = geom_name
            self._geom_id_to_body_name[geom_id] = body_name

            short_body = body_name.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
            is_collision = (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
            )
            if short_body == "knob" and is_collision:
                self._knob_geom_ids.add(geom_id)
                self._knob_global_body_id = body_id
            if (
                "wuji:" in geom_name
                or "wuji:" in body_name
                or "wuji:right_" in geom_name
                or "wuji:right_" in body_name
            ) and is_collision:
                self._wuji_collision_geom_ids.add(geom_id)

    def _get_knob_contact_force_stats(self) -> dict[str, object]:
        expected_sign = self._current_expected_turn_sign()
        stats: dict[str, object] = {
            "num_contacts": 0,
            "normal_force_sum": 0.0,
            "sliding_force_sum": 0.0,
            "max_normal_force": 0.0,
            "max_sliding_force": 0.0,
            "total_axis_torque": 0.0,
            "friction_axis_torque": 0.0,
            "friction_axis_torque_abs_sum": 0.0,
            "contact_dims": [],
            "contact_names": "[]",
            "torque_contact_values": "[]",
            "knob_qpos": 0.0,
            "knob_qvel": 0.0,
            "expected_sign": expected_sign,
        }
        if not self._knob_geom_ids or not self._wuji_collision_geom_ids:
            return stats

        try:
            data = self.env.sim.data
            ncon = int(data.nacon[0].detach().cpu().item())
            if ncon <= 0:
                self._fill_knob_joint_state(stats)
                return stats

            contact_geom = data.contact.geom[:ncon].detach().cpu()
            contact_dim = data.contact.dim[:ncon].detach().cpu()
            contact_pos = data.contact.pos[:ncon].detach().cpu()
            contact_frame = data.contact.frame[:ncon].detach().cpu()
            contact_efc_address = data.contact.efc_address[:ncon].detach().cpu()
            efc_force = data.efc.force[0].detach().cpu()
        except Exception:
            return stats

        torque_ref = self._get_knob_torque_reference()
        if torque_ref is None:
            return stats
        knob_center, knob_axis = torque_ref

        normal_force_sum = 0.0
        sliding_force_sum = 0.0
        max_normal_force = 0.0
        max_sliding_force = 0.0
        total_axis_torque = 0.0
        friction_axis_torque = 0.0
        friction_axis_torque_abs_sum = 0.0
        contact_dims: list[int] = []
        contact_names: list[str] = []
        torque_values: list[tuple[str, float]] = []
        num_contacts = 0

        for contact_id in range(ncon):
            geom1 = int(contact_geom[contact_id, 0].item())
            geom2 = int(contact_geom[contact_id, 1].item())
            if geom1 in self._knob_geom_ids and geom2 in self._wuji_collision_geom_ids:
                knob_geom_id = geom1
                hand_geom_id = geom2
                force_sign_for_knob = -1.0
            elif (
                geom2 in self._knob_geom_ids and geom1 in self._wuji_collision_geom_ids
            ):
                knob_geom_id = geom2
                hand_geom_id = geom1
                force_sign_for_knob = 1.0
            else:
                continue

            del knob_geom_id
            dim = int(contact_dim[contact_id].item())
            addresses = [
                int(addr)
                for addr in contact_efc_address[contact_id, :dim].tolist()
                if 0 <= int(addr) < efc_force.numel()
            ]
            if not addresses:
                continue

            contact_forces = efc_force[addresses]
            abs_forces = contact_forces.abs()
            normal_force = float(abs_forces[0].item())
            sliding_force = (
                float(torch.linalg.norm(contact_forces[1:3]).item())
                if contact_forces.numel() > 1
                else 0.0
            )

            frame = contact_frame[contact_id].reshape(3, 3)[: min(dim, 3)]
            frame_forces = contact_forces[: frame.shape[0]]
            force_w = force_sign_for_knob * (frame * frame_forces.unsqueeze(-1)).sum(
                dim=0
            )
            sliding_force_w = torch.zeros(3, dtype=torch.float32)
            if frame.shape[0] > 1:
                tangent_forces = contact_forces[1 : frame.shape[0]]
                sliding_force_w = force_sign_for_knob * (
                    frame[1 : frame.shape[0]] * tangent_forces.unsqueeze(-1)
                ).sum(dim=0)

            lever_w = contact_pos[contact_id] - knob_center
            total_torque = float(
                torch.dot(torch.cross(lever_w, force_w, dim=0), knob_axis).item()
            )
            friction_torque = float(
                torch.dot(
                    torch.cross(lever_w, sliding_force_w, dim=0), knob_axis
                ).item()
            )

            num_contacts += 1
            normal_force_sum += normal_force
            sliding_force_sum += sliding_force
            max_normal_force = max(max_normal_force, normal_force)
            max_sliding_force = max(max_sliding_force, sliding_force)
            total_axis_torque += total_torque
            friction_axis_torque += friction_torque
            friction_axis_torque_abs_sum += abs(friction_torque)
            contact_dims.append(dim)

            contact_name = self._format_contact_geom_name(hand_geom_id)
            if contact_name not in contact_names:
                contact_names.append(contact_name)
            torque_values.append((contact_name, friction_torque))

        stats["num_contacts"] = num_contacts
        stats["normal_force_sum"] = normal_force_sum
        stats["sliding_force_sum"] = sliding_force_sum
        stats["max_normal_force"] = max_normal_force
        stats["max_sliding_force"] = max_sliding_force
        stats["total_axis_torque"] = total_axis_torque
        stats["friction_axis_torque"] = friction_axis_torque
        stats["friction_axis_torque_abs_sum"] = friction_axis_torque_abs_sum
        stats["contact_dims"] = contact_dims
        stats["contact_names"] = f"[{', '.join(contact_names[:8])}]" + (
            "..." if len(contact_names) > 8 else ""
        )
        if torque_values:
            per_contact = ", ".join(
                f"{name}:{value:+.6f}" for name, value in torque_values[:8]
            )
            if len(torque_values) > 8:
                per_contact += ", ..."
            stats["torque_contact_values"] = f"[{per_contact}]"
        self._fill_knob_joint_state(stats)
        return stats

    def _get_knob_torque_reference(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._knob_global_body_id is None:
            return None
        try:
            data = self.env.sim.data
            xpos = data.xpos.detach()
            if xpos.ndim == 3:
                knob_center = xpos[0, self._knob_global_body_id].detach().cpu()
            else:
                knob_center = xpos[self._knob_global_body_id].detach().cpu()
            plate_quat = self._heating_plate_pose()[0, 3:7].detach().cpu()
        except Exception:
            return None

        local_axis = self.knob_axis.detach().cpu()
        knob_axis = quat_rotate_vec(plate_quat.unsqueeze(0), local_axis.unsqueeze(0))[0]
        knob_axis = knob_axis / torch.clamp(torch.linalg.norm(knob_axis), min=1e-8)
        return knob_center, knob_axis

    def _fill_knob_joint_state(self, stats: dict[str, object]) -> None:
        try:
            qpos = self.heating_plate.data.joint_pos.detach()
            qvel = self.heating_plate.data.joint_vel.detach()
            stats["knob_qpos"] = float(qpos[0, 0].detach().cpu().item())
            stats["knob_qvel"] = float(qvel[0, 0].detach().cpu().item())
        except Exception:
            return

    def _format_contact_geom_name(self, geom_id: int) -> str:
        geom_name = self._geom_id_to_name.get(geom_id, "")
        body_name = self._geom_id_to_body_name.get(geom_id, "")
        short_geom = geom_name.rsplit("/", 1)[-1] if geom_name else str(geom_id)
        short_body = body_name.rsplit("/", 1)[-1] if body_name else ""
        if short_body and short_body != short_geom:
            return f"{short_geom}@{short_body}"
        return short_geom

    def _current_expected_turn_sign(self) -> float:
        if self.state is not None:
            active_angle = float(self._active_turn_angle[0].detach().cpu().item())
            if abs(active_angle) > 1e-7:
                return 1.0 if active_angle > 0.0 else -1.0
        return 1.0 if self.button_turn_angle >= 0.0 else -1.0

    def _quat_from_z_to_axis(self, axis: torch.Tensor) -> torch.Tensor:
        axis = axis.to(dtype=torch.float32)
        axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-8)
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        dot = torch.dot(z_axis, axis).clamp(-1.0, 1.0)
        if float(dot.item()) > 1.0 - 1e-6:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        if float(dot.item()) < -1.0 + 1e-6:
            return torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

        rot_axis = torch.cross(z_axis, axis, dim=0)
        rot_axis = rot_axis / torch.clamp(torch.linalg.norm(rot_axis), min=1e-8)
        angle = torch.acos(dot)
        half = 0.5 * angle
        return torch.cat(
            (
                torch.cos(half).reshape(1),
                rot_axis * torch.sin(half),
            )
        )
