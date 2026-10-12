"""WujiTweezersDiagnostics: extracted diagnostics behavior for wuji_tweezers_manipulate."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from assets.mjcf.robot import UR5E_JOINT_ORDER, WUJI_JOINT_ORDER
from src.utils.policy_helpers.wuji_tweezers_manipulate_common import (
    WujiTweezersManipulateState,
    _available_contact_count,
)
from src.tasks.success_diagnostics import success_diagnostics_for_envs
from src.utils.quat_util import (
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_distance,
)


class WujiTweezersDiagnostics:
    """Provide diagnostics methods used by the task policy."""

    def _init_pose_print_debug(self) -> None:
        if self.server is None:
            return

        with self.server.gui.add_folder("Wuji Tweezers Debug"):
            print_state_button = self.server.gui.add_button(
                label="Print robot/tweezers state"
            )

            @print_state_button.on_click
            def _(_event):
                self._print_robot_tweezers_state()

    def _print_robot_tweezers_state(self) -> None:
        try:
            model = self.env.sim.mj_model
            data = self.env.sim.data
        except Exception as exc:
            print(
                f"[wuji_tweezers_manipulate] Could not access MuJoCo model/data: {exc}"
            )
            return

        print("\n[wuji_tweezers_manipulate] Current robot/tweezers state")
        print("UR5e joints:")
        for joint_name, value in self._read_named_joint_qpos(
            model, data, UR5E_JOINT_ORDER
        ):
            print(f"  {joint_name}: {value:+.6f}")

        print("Wuji hand joints:")
        for joint_name, value in self._read_named_joint_qpos(
            model, data, WUJI_JOINT_ORDER
        ):
            print(f"  {joint_name}: {value:+.6f}")

        self._print_latest_action()

        base_pose = self._read_body_pose(model, data, ("robot/base", "base"))
        if base_pose is None:
            print("Robot base pose: unavailable")
        else:
            pos, quat = base_pose
            print(
                "Robot base pose xyz+quat(wxyz): "
                f"[{pos[0]:+.6f}, {pos[1]:+.6f}, {pos[2]:+.6f}, "
                f"{quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}]"
            )

        tweezers_pose = self._read_body_pose(
            model, data, ("tweezers/tweezers", "tweezers")
        )
        if tweezers_pose is None:
            print("Tweezers pose: unavailable")
        else:
            pos, quat = tweezers_pose
            print(
                "Tweezers pose xyz+quat(wxyz): "
                f"[{pos[0]:+.6f}, {pos[1]:+.6f}, {pos[2]:+.6f}, "
                f"{quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}]"
            )

        hinge = self._read_single_joint_qpos(model, data, "tweezers_hinge")
        if hinge is None:
            print("Tweezers current angle: unavailable")
        else:
            print(
                "Tweezers current angle: "
                f"{hinge:+.6f} rad ({np.degrees(hinge):+.3f} deg)"
            )

    def _print_latest_action(self) -> None:
        if self._latest_action is None:
            print("Latest action: unavailable (policy has not produced an action yet)")
            return

        action = self._latest_action[0].detach().cpu()
        palm_action = action[: self.pose_dim].numpy()
        print(
            "Latest action palm target xyz+quat(wxyz): "
            f"[{palm_action[0]:+.6f}, {palm_action[1]:+.6f}, {palm_action[2]:+.6f}, "
            f"{palm_action[3]:+.6f}, {palm_action[4]:+.6f}, "
            f"{palm_action[5]:+.6f}, {palm_action[6]:+.6f}]"
        )

        print("Wuji hand action qpos:")
        hand_action = action[self.pose_dim : self.pose_dim + self.hand_dim].numpy()
        for joint_name, value in zip(WUJI_JOINT_ORDER, hand_action, strict=False):
            print(f"  {self._strip_regex_prefix(joint_name)}: {float(value):+.6f}")

    def _read_named_joint_qpos(
        self, model, data, joint_names
    ) -> list[tuple[str, float]]:
        values: list[tuple[str, float]] = []
        for joint_name in joint_names:
            qpos = self._read_single_joint_qpos(model, data, joint_name)
            values.append(
                (
                    self._strip_regex_prefix(joint_name),
                    float("nan") if qpos is None else qpos,
                )
            )
        return values

    def _read_single_joint_qpos(self, model, data, joint_name: str) -> float | None:
        joint_id = self._find_joint_id(model, joint_name)
        if joint_id is None:
            return None

        try:
            qpos_addr = int(model.jnt_qposadr[joint_id])
            qpos = data.qpos
            if qpos.ndim == 2:
                return float(qpos[0, qpos_addr].detach().cpu().item())
            return float(qpos[qpos_addr].detach().cpu().item())
        except Exception:
            return None

    def _find_joint_id(self, model, joint_name: str) -> int | None:
        target = self._strip_regex_prefix(joint_name)
        for joint_id in range(model.njnt):
            name = model.joint(joint_id).name or ""
            if (
                name == target
                or name.endswith(f"/{target}")
                or name.endswith(f":{target}")
            ):
                return joint_id
        return None

    def _strip_regex_prefix(self, name: str) -> str:
        return name[2:] if name.startswith(".*") else name

    def _read_body_pose(
        self,
        model,
        data,
        body_names: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        body_id = self._find_body_id(model, body_names)
        if body_id is None:
            return None

        try:
            xpos = data.xpos
            xquat = data.xquat
            if xpos.ndim == 3:
                pos = xpos[0, body_id].detach().cpu().numpy()
                quat = xquat[0, body_id].detach().cpu().numpy()
            else:
                pos = xpos[body_id].detach().cpu().numpy()
                quat = xquat[body_id].detach().cpu().numpy()
            return pos, quat
        except Exception:
            return None

    def _find_body_id(self, model, body_names: tuple[str, ...]) -> int | None:
        for body_id in range(model.nbody):
            name = model.body(body_id).name or ""
            if name in body_names:
                return body_id

        suffixes = tuple(f"/{name}" for name in body_names)
        for body_id in range(model.nbody):
            name = model.body(body_id).name or ""
            if name.endswith(suffixes):
                return body_id
        return None

    def _debug_print_pose_error(
        self,
        obs,
        palm_pose_w: torch.Tensor,
        target_action: torch.Tensor,
        pos_err: torch.Tensor,
        rot_err: torch.Tensor,
    ) -> None:
        self._debug_step += 1
        if self._debug_step % self._debug_print_interval != 0:
            return

        env_id = 0
        pos_delta = (palm_pose_w[env_id, :3] - target_action[env_id, :3]).detach().cpu()
        palm_pos = palm_pose_w[env_id, :3].detach().cpu()
        target_pos = target_action[env_id, :3].detach().cpu()

        tweezers_pose_w = self._get_tweezers_pose_w(obs)
        tweezers_in_palm = compose_pose_wxyz(
            pose_inv_wxyz(palm_pose_w),
            tweezers_pose_w,
        )
        target_tweezers_in_palm = compose_pose_wxyz(
            pose_inv_wxyz(target_action[:, :7]),
            tweezers_pose_w,
        )
        rel_pos_delta = (
            (tweezers_in_palm[env_id, :3] - target_tweezers_in_palm[env_id, :3])
            .detach()
            .cpu()
        )
        rel_pos_err = torch.linalg.norm(
            tweezers_in_palm[:, :3] - target_tweezers_in_palm[:, :3],
            dim=-1,
        )
        rel_rot_err = quat_distance(
            F.normalize(tweezers_in_palm[:, 3:7], dim=-1),
            F.normalize(target_tweezers_in_palm[:, 3:7], dim=-1),
        )

        state_name = (
            WujiTweezersManipulateState(int(self.state[env_id].item())).name
            if self.state is not None
            else "UNKNOWN"
        )
        traj_idx = int(self.idx[env_id].item()) if self.idx is not None else -1
        traj_len = (
            int(self.traj_len[env_id].item()) if self.traj_len is not None else -1
        )
        print(
            "[wuji_tweezers_manipulate] "
            f"step={self._debug_step} state={state_name} idx={traj_idx}/{traj_len} "
            f"palm-target-pos=({pos_delta[0]:+.6f}, {pos_delta[1]:+.6f}, {pos_delta[2]:+.6f}) "
            f"pos_err={float(pos_err[env_id]):.6f} rot_err={float(rot_err[env_id]):.6f} "
            f"tweezers-in-palm-target-pos=({rel_pos_delta[0]:+.6f}, {rel_pos_delta[1]:+.6f}, {rel_pos_delta[2]:+.6f}) "
            f"rel_pos_err={float(rel_pos_err[env_id]):.6f} rel_rot_err={float(rel_rot_err[env_id]):.6f} "
            f"palm_pos=({palm_pos[0]:+.6f}, {palm_pos[1]:+.6f}, {palm_pos[2]:+.6f}) "
            f"target_pos=({target_pos[0]:+.6f}, {target_pos[1]:+.6f}, {target_pos[2]:+.6f})"
        )

    def _init_success_debug(self) -> None:
        if self.server is None:
            return
        with self.server.gui.add_folder("Wuji Tweezers Success Debug"):
            self._success_debug_text = self.server.gui.add_text(
                label="env 0 check_success",
                initial_value="Waiting for first environment step...",
                multiline=True,
                disabled=True,
            )

    def _update_success_debug_text(self) -> None:
        if self._success_debug_text is None:
            return

        snapshot = success_diagnostics_for_envs(self.env, [0]).get(0)
        if snapshot is None:
            self._success_debug_text.value = "Waiting for first environment step..."
            return

        components = snapshot["components"]
        metrics = snapshot["metrics"]
        thresholds = snapshot["thresholds"]
        outer_radius = metrics["tube_outer_radius_m"]
        lowest_height = metrics["tube_lowest_height_m"]
        highest_height = metrics["tube_highest_height_m"]
        stable_steps = int(metrics["stable_steps"])
        inner_radius = thresholds["beaker_inner_radius_m"]
        bottom_height = thresholds["beaker_interior_bottom_height_m"]
        rim_height = thresholds["beaker_rim_height_m"]
        containment_tolerance = thresholds["containment_tolerance_m"]
        bottom_tolerance = thresholds["bottom_tolerance_m"]
        required_stability = int(thresholds["required_stability_steps"])
        max_palm_dist = thresholds["max_tweezers_palm_dist_m"]
        min_contact_force = thresholds["min_hand_contact_force_n"]

        def result(value: bool) -> str:
            return "PASS" if value else "FAIL"

        stable = stable_steps >= required_stability
        self._success_debug_text.value = (
            f"tube_within_beaker_radius: "
            f"{result(components['tube_within_beaker_radius'])} | "
            f"outer_r={outer_radius:.4f} <= "
            f"{inner_radius + containment_tolerance:.4f} m\n"
            f"tube_above_beaker_bottom: "
            f"{result(components['tube_above_beaker_bottom'])} | "
            f"lowest_z={lowest_height:.4f} >= "
            f"{bottom_height - bottom_tolerance:.4f} m\n"
            f"tube_below_beaker_rim: "
            f"{result(components['tube_below_beaker_rim'])} | "
            f"highest_z={highest_height:.4f} <= "
            f"{rim_height + containment_tolerance:.4f} m\n"
            f"object_complete_now: {result(components['object_complete'])} | "
            "radius AND bottom AND rim\n"
            f"tweezers_in_hand (diagnostic only): "
            f"{result(components['tweezers_in_hand'])} | palm distance <= "
            f"{max_palm_dist:.4f} m AND contact force >= {min_contact_force:.3f} N\n"
            f"stable: {result(stable)} | "
            f"{stable_steps}/{required_stability} consecutive steps\n"
            f"full_success: {result(components['full_success'])}"
        )

    def _init_contact_force_debug(self) -> None:
        if self.server is not None:
            with self.server.gui.add_folder("Wuji Tweezers Contact"):
                self._contact_force_text = self.server.gui.add_text(
                    label="Friction / Gravity",
                    initial_value="waiting for contact...",
                    multiline=True,
                    disabled=True,
                )

        try:
            model = self.env.sim.mj_model
        except Exception:
            return

        for geom_id in range(model.ngeom):
            geom_name = model.geom(geom_id).name or str(geom_id)
            body_id = int(model.geom_bodyid[geom_id])
            body_name = model.body(body_id).name or str(body_id)
            self._geom_id_to_body_id[geom_id] = body_id
            self._geom_id_to_name[geom_id] = geom_name

            if "tweezers/" in geom_name or body_name.startswith("tweezers/"):
                self._tweezers_geom_ids.add(geom_id)
            if (
                "wuji:" in geom_name
                or "wuji:" in body_name
                or "wuji:right_" in geom_name
            ):
                if (
                    int(model.geom_contype[geom_id]) != 0
                    or int(model.geom_conaffinity[geom_id]) != 0
                ):
                    self._wuji_collision_geom_ids.add(geom_id)

        tweezers_root_ids = [
            body_id
            for body_id in range(model.nbody)
            if (model.body(body_id).name or "").startswith("tweezers/")
        ]
        if tweezers_root_ids:
            root_id = min(tweezers_root_ids)
            self._tweezers_body_ids = {
                body_id
                for body_id in range(model.nbody)
                if body_id == root_id
                or self._body_has_ancestor(model, body_id, root_id)
            }

        if self._tweezers_body_ids:
            mass = sum(
                float(model.body_mass[body_id]) for body_id in self._tweezers_body_ids
            )
        else:
            mass = 0.0
        gravity = torch.as_tensor(model.opt.gravity, dtype=torch.float32)
        self._tweezers_weight_n = mass * float(torch.linalg.norm(gravity).item())

    def _body_has_ancestor(self, model, body_id: int, ancestor_id: int) -> bool:
        parent_id = int(model.body_parentid[body_id])
        while parent_id > 0:
            if parent_id == ancestor_id:
                return True
            parent_id = int(model.body_parentid[parent_id])
        return False

    def _update_contact_force_display(self) -> None:
        if self._contact_force_text is None:
            return

        stats = self._get_tweezers_contact_force_stats()
        gravity = float(stats["gravity_force"])
        sliding = float(stats["sliding_force_sum"])
        support = float(stats["vertical_friction_support"])
        ratio = sliding / gravity if gravity > 1e-8 else 0.0
        support_ratio = support / gravity if gravity > 1e-8 else 0.0

        self._contact_force_text.value = (
            f"contacts: {stats['num_contacts']}\n"
            f"gravity: {gravity:.4f} N\n"
            f"sliding friction sum: {sliding:.4f} N ({ratio:.2f}x gravity)\n"
            f"upward friction support: {support:.4f} N ({support_ratio:.2f}x gravity)\n"
            f"normal force sum: {float(stats['normal_force_sum']):.4f} N\n"
            f"extra friction constraints: {float(stats['extra_constraint_sum']):.4f} N\n"
            f"contact geoms: {stats['contact_names']}"
        )

    def _get_tweezers_contact_force_stats(self) -> dict[str, object]:
        stats: dict[str, object] = {
            "num_contacts": 0,
            "normal_force_sum": 0.0,
            "sliding_force_sum": 0.0,
            "vertical_friction_support": 0.0,
            "extra_constraint_sum": 0.0,
            "gravity_force": self._tweezers_weight_n,
            "contact_names": "[]",
        }
        if not self._tweezers_geom_ids or not self._wuji_collision_geom_ids:
            return stats

        try:
            model = self.env.sim.mj_model
            data = self.env.sim.data
            ncon = _available_contact_count(data)
            if ncon <= 0:
                return stats

            contact_geom = data.contact.geom[:ncon].detach().cpu()
            contact_dim = data.contact.dim[:ncon].detach().cpu()
            contact_frame = data.contact.frame[:ncon].detach().cpu()
            contact_efc_address = data.contact.efc_address[:ncon].detach().cpu()
            efc_force = data.efc.force[0].detach().cpu()
            gravity = torch.as_tensor(model.opt.gravity, dtype=torch.float32)
        except Exception:
            return stats

        gravity_norm = float(torch.linalg.norm(gravity).item())
        up_axis = (
            -gravity / gravity_norm
            if gravity_norm > 1e-8
            else torch.tensor([0.0, 0.0, 1.0])
        )
        normal_force_sum = 0.0
        sliding_force_sum = 0.0
        extra_constraint_sum = 0.0
        vertical_friction_support = 0.0
        contact_names: list[str] = []
        num_contacts = 0

        for contact_id in range(ncon):
            geom1 = int(contact_geom[contact_id, 0].item())
            geom2 = int(contact_geom[contact_id, 1].item())
            geom1_is_tweezers = geom1 in self._tweezers_geom_ids
            geom2_is_tweezers = geom2 in self._tweezers_geom_ids
            geom1_is_wuji = geom1 in self._wuji_collision_geom_ids
            geom2_is_wuji = geom2 in self._wuji_collision_geom_ids

            if geom1_is_tweezers and geom2_is_wuji:
                hand_geom_id = geom2
                force_sign_for_tweezers = -1.0
            elif geom2_is_tweezers and geom1_is_wuji:
                hand_geom_id = geom1
                force_sign_for_tweezers = 1.0
            else:
                continue

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
            extra_constraint = (
                float(torch.linalg.norm(contact_forces[3:]).item())
                if contact_forces.numel() > 3
                else 0.0
            )

            frame = contact_frame[contact_id].reshape(3, 3)[: min(dim, 3)]
            sliding_force_w = torch.zeros(3, dtype=torch.float32)
            if frame.shape[0] > 1:
                tangent_forces = contact_forces[1 : frame.shape[0]]
                sliding_force_w = force_sign_for_tweezers * (
                    frame[1 : frame.shape[0]] * tangent_forces.unsqueeze(-1)
                ).sum(dim=0)
            upward_support = max(0.0, float(torch.dot(sliding_force_w, up_axis).item()))

            num_contacts += 1
            normal_force_sum += normal_force
            sliding_force_sum += sliding_force
            extra_constraint_sum += extra_constraint
            vertical_friction_support += upward_support

            contact_name = self._geom_id_to_name.get(
                hand_geom_id, str(hand_geom_id)
            ).rsplit("/", 1)[-1]
            if contact_name not in contact_names:
                contact_names.append(contact_name)

        stats["num_contacts"] = num_contacts
        stats["normal_force_sum"] = normal_force_sum
        stats["sliding_force_sum"] = sliding_force_sum
        stats["vertical_friction_support"] = vertical_friction_support
        stats["extra_constraint_sum"] = extra_constraint_sum
        stats["contact_names"] = f"[{', '.join(contact_names[:8])}]" + (
            "..." if len(contact_names) > 8 else ""
        )
        return stats
