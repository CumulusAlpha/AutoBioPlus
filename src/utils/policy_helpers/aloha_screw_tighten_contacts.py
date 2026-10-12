"""AlohaContactDiagnostics: extracted contacts behavior for aloha_screw_tighten."""

from __future__ import annotations

import torch

from src.utils.policy_helpers.aloha_screw_tighten_common import (
    CAP_COLLISION_GEOM_NAMES,
    CAP_ROTATION_AXIS_LOCAL,
    FINGERTIP_SPHERE_GEOM_NAMES,
    GRIPPER_COLLISION_GEOM_NAMES,
)
from src.utils.quat_util import (
    quat_rotate_vec,
)


class AlohaContactDiagnostics:
    """Provide contacts methods used by the task policy."""

    def _resolve_contact_geom_ids(self) -> None:
        if self.env is None:
            return

        try:
            import mujoco

            model = self.env.sim.mj_model
        except Exception:
            return

        self._geom_id_to_name.clear()
        self._geom_id_to_body_id.clear()
        self._cap_geom_ids.clear()
        self._cap_body_ids.clear()
        self._gripper_collision_geom_ids.clear()
        self._fingertip_sphere_geom_ids.clear()
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            self._geom_id_to_name[geom_id] = name
            body_id = int(model.geom_bodyid[geom_id])
            self._geom_id_to_body_id[geom_id] = body_id
            short_name = name.rsplit("/", 1)[-1]
            if short_name in CAP_COLLISION_GEOM_NAMES:
                self._cap_geom_ids.add(geom_id)
                self._cap_body_ids.add(body_id)
            if short_name in GRIPPER_COLLISION_GEOM_NAMES:
                self._gripper_collision_geom_ids.add(geom_id)
            if short_name in FINGERTIP_SPHERE_GEOM_NAMES:
                self._fingertip_sphere_geom_ids.add(geom_id)

    def _get_cap_contact_force_stats(self) -> dict[str, object]:
        stats: dict[str, object] = {
            "num_contacts": 0,
            "normal_force_sum": 0.0,
            "sliding_force_sum": 0.0,
            "extra_constraint_sum": 0.0,
            "max_normal_force": 0.0,
            "max_sliding_force": 0.0,
            "contact_dims": [],
            "contact_names": [],
            "torque_axis_sum": 0.0,
            "torque_axis_abs_sum": 0.0,
            "torque_axis_pos_sum": 0.0,
            "torque_axis_neg_sum": 0.0,
            "torque_contacts": 0,
            "torque_contact_values": "[]",
            "sphere_num_contacts": 0,
            "sphere_normal_force_sum": 0.0,
            "sphere_sliding_force_sum": 0.0,
            "sphere_extra_constraint_sum": 0.0,
            "sphere_max_normal_force": 0.0,
            "sphere_max_sliding_force": 0.0,
            "sphere_contact_dims": [],
            "sphere_contact_names": [],
            "sphere_torque_axis_sum": 0.0,
            "sphere_torque_axis_abs_sum": 0.0,
            "sphere_torque_axis_pos_sum": 0.0,
            "sphere_torque_axis_neg_sum": 0.0,
            "sphere_torque_contacts": 0,
            "sphere_torque_contact_values": "[]",
        }
        if not self._cap_geom_ids or (
            not self._gripper_collision_geom_ids and not self._fingertip_sphere_geom_ids
        ):
            return stats

        try:
            data = self.env.sim.data
            ncon = int(data.nacon[0].detach().cpu().item())
            if ncon <= 0:
                return stats

            contact_geom = data.contact.geom[:ncon].detach().cpu()
            contact_dim = data.contact.dim[:ncon].detach().cpu()
            contact_pos = data.contact.pos[:ncon].detach().cpu()
            contact_frame = data.contact.frame[:ncon].detach().cpu()
            contact_efc_address = data.contact.efc_address[:ncon].detach().cpu()
            efc_force = data.efc.force[0].detach().cpu()
        except Exception:
            return stats

        normal_force_sum = 0.0
        sliding_force_sum = 0.0
        extra_constraint_sum = 0.0
        max_normal_force = 0.0
        max_sliding_force = 0.0
        contact_dims: list[int] = []
        contact_names: list[str] = []
        torque_values: list[float] = []
        torque_names: list[str] = []
        sphere_torque_values: list[float] = []
        sphere_torque_names: list[str] = []
        num_contacts = 0
        sphere_normal_force_sum = 0.0
        sphere_sliding_force_sum = 0.0
        sphere_extra_constraint_sum = 0.0
        sphere_max_normal_force = 0.0
        sphere_max_sliding_force = 0.0
        sphere_contact_dims: list[int] = []
        sphere_contact_names: list[str] = []
        sphere_num_contacts = 0
        torque_ref = self._get_cap_torque_reference()

        for contact_id in range(ncon):
            geom1 = int(contact_geom[contact_id, 0].item())
            geom2 = int(contact_geom[contact_id, 1].item())
            sphere_geom_id = None
            if geom1 in self._cap_geom_ids and geom2 in self._fingertip_sphere_geom_ids:
                sphere_geom_id = geom2
            elif (
                geom2 in self._cap_geom_ids and geom1 in self._fingertip_sphere_geom_ids
            ):
                sphere_geom_id = geom1

            if (
                geom1 in self._cap_geom_ids
                and geom2 in self._gripper_collision_geom_ids
            ):
                gripper_geom_id = geom2
            elif (
                geom2 in self._cap_geom_ids
                and geom1 in self._gripper_collision_geom_ids
            ):
                gripper_geom_id = geom1
            else:
                gripper_geom_id = None

            if gripper_geom_id is None and sphere_geom_id is None:
                continue

            dim = int(contact_dim[contact_id].item())
            addresses = [
                int(addr)
                for addr in contact_efc_address[contact_id, :dim].tolist()
                if 0 <= int(addr) < efc_force.numel()
            ]
            if not addresses:
                continue

            forces = efc_force[addresses].abs()
            normal_force = float(forces[0].item())
            sliding_force = (
                float(torch.linalg.norm(forces[1:3]).item())
                if forces.numel() > 1
                else 0.0
            )
            extra_constraint = (
                float(torch.linalg.norm(forces[3:]).item())
                if forces.numel() > 3
                else 0.0
            )

            if gripper_geom_id is not None:
                num_contacts += 1
                normal_force_sum += normal_force
                sliding_force_sum += sliding_force
                extra_constraint_sum += extra_constraint
                max_normal_force = max(max_normal_force, normal_force)
                max_sliding_force = max(max_sliding_force, sliding_force)
                contact_dims.append(dim)
                contact_name = self._geom_id_to_name.get(
                    gripper_geom_id, str(gripper_geom_id)
                ).rsplit("/", 1)[-1]
                if contact_name not in contact_names:
                    contact_names.append(contact_name)
            else:
                contact_name = ""

            if sphere_geom_id is not None:
                sphere_num_contacts += 1
                sphere_normal_force_sum += normal_force
                sphere_sliding_force_sum += sliding_force
                sphere_extra_constraint_sum += extra_constraint
                sphere_max_normal_force = max(sphere_max_normal_force, normal_force)
                sphere_max_sliding_force = max(sphere_max_sliding_force, sliding_force)
                sphere_contact_dims.append(dim)
                sphere_contact_name = self._geom_id_to_name.get(
                    sphere_geom_id, str(sphere_geom_id)
                ).rsplit("/", 1)[-1]
                if sphere_contact_name not in sphere_contact_names:
                    sphere_contact_names.append(sphere_contact_name)

                if torque_ref is not None:
                    sphere_torque_value = self._estimate_contact_axis_torque(
                        contact_id=contact_id,
                        cap_geom_id=self._cap_geom_for_contact(geom1, geom2),
                        gripper_geom_id=sphere_geom_id,
                        contact_pos=contact_pos,
                        contact_frame=contact_frame,
                        contact_efc_address=contact_efc_address,
                        contact_dim=contact_dim,
                        efc_force=efc_force,
                        cap_center=torque_ref[0],
                        cap_axis=torque_ref[1],
                    )
                    if sphere_torque_value is not None:
                        sphere_torque_values.append(sphere_torque_value)
                        sphere_torque_names.append(sphere_contact_name)

            if gripper_geom_id is not None and torque_ref is not None:
                torque_value = self._estimate_contact_axis_torque(
                    contact_id=contact_id,
                    cap_geom_id=self._cap_geom_for_contact(geom1, geom2),
                    gripper_geom_id=gripper_geom_id,
                    contact_pos=contact_pos,
                    contact_frame=contact_frame,
                    contact_efc_address=contact_efc_address,
                    contact_dim=contact_dim,
                    efc_force=efc_force,
                    cap_center=torque_ref[0],
                    cap_axis=torque_ref[1],
                )
                if torque_value is not None:
                    torque_values.append(torque_value)
                    torque_names.append(contact_name)

        stats["num_contacts"] = num_contacts
        stats["normal_force_sum"] = normal_force_sum
        stats["sliding_force_sum"] = sliding_force_sum
        stats["extra_constraint_sum"] = extra_constraint_sum
        stats["max_normal_force"] = max_normal_force
        stats["max_sliding_force"] = max_sliding_force
        stats["contact_dims"] = contact_dims
        stats["contact_names"] = contact_names
        stats["sphere_num_contacts"] = sphere_num_contacts
        stats["sphere_normal_force_sum"] = sphere_normal_force_sum
        stats["sphere_sliding_force_sum"] = sphere_sliding_force_sum
        stats["sphere_extra_constraint_sum"] = sphere_extra_constraint_sum
        stats["sphere_max_normal_force"] = sphere_max_normal_force
        stats["sphere_max_sliding_force"] = sphere_max_sliding_force
        stats["sphere_contact_dims"] = sphere_contact_dims
        stats["sphere_contact_names"] = sphere_contact_names
        if torque_values:
            torque_sum = sum(torque_values)
            torque_abs_sum = sum(abs(v) for v in torque_values)
            torque_pos_sum = sum(v for v in torque_values if v > 0.0)
            torque_neg_sum = sum(v for v in torque_values if v < 0.0)
            torque_contact_values = ", ".join(
                f"{name}:{value:+.5f}"
                for name, value in zip(torque_names[:6], torque_values[:6])
            )
            if len(torque_values) > 6:
                torque_contact_values += ", ..."
            stats["torque_axis_sum"] = torque_sum
            stats["torque_axis_abs_sum"] = torque_abs_sum
            stats["torque_axis_pos_sum"] = torque_pos_sum
            stats["torque_axis_neg_sum"] = torque_neg_sum
            stats["torque_contacts"] = len(torque_values)
            stats["torque_contact_values"] = f"[{torque_contact_values}]"
        if sphere_torque_values:
            sphere_torque_sum = sum(sphere_torque_values)
            sphere_torque_abs_sum = sum(abs(v) for v in sphere_torque_values)
            sphere_torque_pos_sum = sum(v for v in sphere_torque_values if v > 0.0)
            sphere_torque_neg_sum = sum(v for v in sphere_torque_values if v < 0.0)
            sphere_torque_contact_values = ", ".join(
                f"{name}:{value:+.5f}"
                for name, value in zip(
                    sphere_torque_names[:8], sphere_torque_values[:8]
                )
            )
            if len(sphere_torque_values) > 8:
                sphere_torque_contact_values += ", ..."
            stats["sphere_torque_axis_sum"] = sphere_torque_sum
            stats["sphere_torque_axis_abs_sum"] = sphere_torque_abs_sum
            stats["sphere_torque_axis_pos_sum"] = sphere_torque_pos_sum
            stats["sphere_torque_axis_neg_sum"] = sphere_torque_neg_sum
            stats["sphere_torque_contacts"] = len(sphere_torque_values)
            stats["sphere_torque_contact_values"] = f"[{sphere_torque_contact_values}]"
        return stats

    def _cap_geom_for_contact(self, geom1: int, geom2: int) -> int | None:
        if geom1 in self._cap_geom_ids:
            return geom1
        if geom2 in self._cap_geom_ids:
            return geom2
        return None

    def _get_cap_torque_reference(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._cap_body_ids:
            return None
        cap_body_id = min(self._cap_body_ids)
        try:
            data = self.env.sim.data
            xpos = data.xpos.detach()
            xquat = data.xquat.detach()
            if xpos.ndim == 3:
                cap_center = xpos[0, cap_body_id].detach().cpu()
                cap_quat = xquat[0, cap_body_id].detach().cpu()
            else:
                cap_center = xpos[cap_body_id].detach().cpu()
                cap_quat = xquat[cap_body_id].detach().cpu()
        except Exception:
            try:
                mj_data = self.env.sim.mj_data
                cap_center = torch.as_tensor(
                    mj_data.xpos[cap_body_id], dtype=torch.float32
                )
                cap_quat = torch.as_tensor(
                    mj_data.xquat[cap_body_id], dtype=torch.float32
                )
            except Exception:
                return None

        local_axis = torch.tensor(CAP_ROTATION_AXIS_LOCAL, dtype=cap_center.dtype)
        cap_axis = quat_rotate_vec(cap_quat.unsqueeze(0), local_axis.unsqueeze(0))[0]
        cap_axis = cap_axis / torch.clamp(torch.linalg.norm(cap_axis), min=1e-8)
        return cap_center, cap_axis

    def _estimate_contact_axis_torque(
        self,
        contact_id: int,
        cap_geom_id: int | None,
        gripper_geom_id: int,
        contact_pos: torch.Tensor,
        contact_frame: torch.Tensor,
        contact_efc_address: torch.Tensor,
        contact_dim: torch.Tensor,
        efc_force: torch.Tensor,
        cap_center: torch.Tensor,
        cap_axis: torch.Tensor,
    ) -> float | None:
        if cap_geom_id is None:
            return None

        dim = int(contact_dim[contact_id].item())
        addresses = [
            int(addr)
            for addr in contact_efc_address[contact_id, :dim].tolist()
            if 0 <= int(addr) < efc_force.numel()
        ]
        if not addresses:
            return None

        frame = contact_frame[contact_id].reshape(3, 3)[: min(dim, 3)]
        forces = efc_force[addresses[: frame.shape[0]]]
        force_sign_for_cap = (
            -1.0
            if self._contact_geom_order_is_cap_first(
                contact_id, cap_geom_id, gripper_geom_id
            )
            else 1.0
        )
        force_w = force_sign_for_cap * (frame * forces.unsqueeze(-1)).sum(dim=0)
        lever_w = contact_pos[contact_id] - cap_center
        return float(torch.dot(torch.cross(lever_w, force_w, dim=0), cap_axis).item())

    def _contact_geom_order_is_cap_first(
        self, contact_id: int, cap_geom_id: int, gripper_geom_id: int
    ) -> bool:
        try:
            geom_pair = self.env.sim.data.contact.geom[contact_id].detach().cpu()
            return (
                int(geom_pair[0].item()) == cap_geom_id
                and int(geom_pair[1].item()) == gripper_geom_id
            )
        except Exception:
            return True
