"""Random, zero-action and Viser joint-slider policies."""

from __future__ import annotations

import numpy as np
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv


class PolicyRandom:
    def __init__(self, action_shape, device):
        self.action_shape = action_shape
        self.device = device

    def __call__(self, obs):

        del obs
        return 2 * torch.rand(self.action_shape, device=self.device) - 1


class PolicyNoOp:
    def __init__(self, action_shape, device):
        self.action_shape = action_shape
        self.device = device

    def __call__(self, obs):
        del obs
        return torch.zeros(self.action_shape, device=self.device)


class PolicyViserSlider:
    def __init__(
        self, action_shape, device, server: viser.ViserServer, env: ManagerBasedRlEnv
    ):
        self.action_shape = action_shape
        self.device = device
        self.env = env
        self.num_envs, self.act_dim = action_shape
        if server is None:
            self.server = viser.ViserServer()
        else:
            self.server = server

        self.action_entity_names = {
            term.cfg.entity_name
            for term in self.env.unwrapped.action_manager._terms.values()
            if getattr(term.cfg, "entity_name", None) is not None
        }
        self._slider_meta = self._build_slider_meta()
        self.mocap_handles = {}
        self.mocap_targets = {}
        self.mocap_pose_texts = {}

        # Create sliders (without creating another server)
        tabs = self.server.gui.add_tab_group()
        self.sliders = []
        with tabs.add_tab("User Part", icon=viser.Icon.ALIEN):
            with self.server.gui.add_folder("🎮 Action Control"):
                for group_name, metas in self._group_slider_meta():
                    with self.server.gui.add_folder(group_name):
                        for meta in metas:
                            slider = self.server.gui.add_slider(
                                label=meta["label"],
                                min=meta["min"],
                                max=meta["max"],
                                step=meta["step"],
                                initial_value=meta["initial"],
                            )
                            self.sliders.append(slider)
                print_control_button = self.server.gui.add_button(
                    label="Print joint/tendon controls"
                )

                @print_control_button.on_click
                def _(_event):
                    self._print_current_slider_controls()

            self._add_mocap_transform_controls()
            if self.mocap_targets:
                with self.server.gui.add_folder("Mocap Pose"):
                    for entity_name in self.mocap_targets:
                        self.mocap_pose_texts[entity_name] = self.server.gui.add_text(
                            label=f"{entity_name}: xyz + quat(wxyz)",
                            initial_value=self._format_mocap_pose(
                                self._get_current_control_pose(entity_name)
                            ),
                            multiline=True,
                            disabled=True,
                        )
                    print_pose_button = self.server.gui.add_button(
                        label="Print transform_control poses"
                    )

                    @print_pose_button.on_click
                    def _(_event):
                        self._print_current_control_poses()

        self._install_reset_hook()
        print(f"✅ Viser slider policy initialized with {self.act_dim} dims")

    def _group_slider_meta(self):
        groups = []
        current_group = None
        current_metas = []
        for meta in self._slider_meta:
            group = meta["group"]
            if current_group is None:
                current_group = group
            if group != current_group:
                groups.append((current_group, current_metas))
                current_group = group
                current_metas = []
            current_metas.append(meta)
        if current_group is not None:
            groups.append((current_group, current_metas))
        return groups

    def _add_mocap_transform_controls(self):
        for entity_name, entity in self.env.scene.entities.items():
            if self.action_entity_names and entity_name not in self.action_entity_names:
                continue
            if not getattr(entity, "is_mocap", False):
                continue

            pose = self._get_current_mocap_pose(entity)
            self.mocap_handles[entity_name] = entity
            self.mocap_targets[entity_name] = self.server.scene.add_transform_controls(
                name=f"/mocap/{entity_name}",
                scale=0.18,
                line_width=3.0,
                depth_test=False,
                position=pose[:3],
                wxyz=pose[3:],
            )

    def _build_slider_meta(self):
        from mjlab.actuator.actuator import TransmissionType
        from mjlab.envs.mdp.actions import JointPositionAction

        mj_model = self.env.unwrapped.sim.mj_model
        action_manager = self.env.unwrapped.action_manager

        metas = []
        for term_name, term in action_manager._terms.items():
            entity_name = getattr(term.cfg, "entity_name", None)
            group = f"{term_name} ({entity_name})" if entity_name else term_name

            target_names = getattr(term, "target_names", None)
            target_ids = getattr(term, "target_ids", None)
            entity = getattr(term, "_entity", None)
            if target_names is None or target_ids is None or entity is None:
                # Fallback: allocate generic sliders if term doesn't expose targets
                for i in range(term.action_dim):
                    metas.append(
                        {
                            "group": group,
                            "label": f"action[{i}]",
                            "min": -1.0,
                            "max": 1.0,
                            "step": 0.01,
                            "initial": 0.0,
                            "scale": 1.0,
                            "offset": 0.0,
                            "kind": "action",
                            "term": term_name,
                            "entity": entity_name,
                        }
                    )
                continue

            # Resolve per-dim scale/offset (can be float or tensor)
            scale = term.scale
            offset = term.offset

            if isinstance(scale, (float, int)):
                scale_list = [float(scale)] * len(target_names)
            else:
                scale_list = scale[0].detach().cpu().numpy().tolist()

            if isinstance(offset, (float, int)):
                offset_list = [float(offset)] * len(target_names)
            else:
                offset_list = offset[0].detach().cpu().numpy().tolist()

            for i, (name, local_id) in enumerate(
                zip(target_names, target_ids.tolist(), strict=False)
            ):
                min_v, max_v = -1.0, 1.0
                if term.cfg.transmission_type == TransmissionType.JOINT:
                    kind = "joint"
                    global_id = int(entity.data.indexing.joint_ids[local_id].item())
                    if mj_model.jnt_limited[global_id]:
                        min_v = float(mj_model.jnt_range[global_id, 0])
                        max_v = float(mj_model.jnt_range[global_id, 1])
                elif term.cfg.transmission_type == TransmissionType.TENDON:
                    kind = "tendon"
                    global_id = int(entity.data.indexing.tendon_ids[local_id].item())
                    if mj_model.tendon_limited[global_id]:
                        min_v = float(mj_model.tendon_range[global_id, 0])
                        max_v = float(mj_model.tendon_range[global_id, 1])
                else:
                    kind = "action"

                # Pick a reasonable step based on range
                step = 0.001
                if max_v > min_v:
                    step = max(0.001, (max_v - min_v) / 200.0)

                # Initial value: prefer env default state for joint-position actions.
                if term.cfg.transmission_type == TransmissionType.JOINT and isinstance(
                    term, JointPositionAction
                ):
                    init = float(entity.data.default_joint_pos[0, local_id].item())
                else:
                    init = offset_list[i]
                if not (min_v <= init <= max_v):
                    init = 0.5 * (min_v + max_v)

                metas.append(
                    {
                        "group": group,
                        "label": name,
                        "min": min_v,
                        "max": max_v,
                        "step": step,
                        "initial": init,
                        "scale": float(scale_list[i]),
                        "offset": float(offset_list[i]),
                        "kind": kind,
                        "term": term_name,
                        "entity": entity_name,
                    }
                )

        # Safety: ensure we match action dim
        if len(metas) != self.act_dim:
            if len(metas) < self.act_dim:
                for i in range(self.act_dim - len(metas)):
                    metas.append(
                        {
                            "group": "extra",
                            "label": f"action_{len(metas)}",
                            "min": -1.0,
                            "max": 1.0,
                            "step": 0.01,
                            "initial": 0.0,
                            "scale": 1.0,
                            "offset": 0.0,
                            "kind": "action",
                            "term": "extra",
                            "entity": None,
                        }
                    )
            else:
                metas = metas[: self.act_dim]

        return metas

    def _get_current_mocap_pose(self, entity):
        data = entity.data
        mocap_id = data.indexing.mocap_id
        if mocap_id is not None:
            pos = data.data.mocap_pos[0, mocap_id].detach().cpu().numpy()
            quat = data.data.mocap_quat[0, mocap_id].detach().cpu().numpy()
            return np.concatenate([pos, quat], axis=0)

        pose = data.root_link_pose_w[0].detach().cpu().numpy()
        return pose

    def _get_current_control_pose(self, entity_name):
        target = self.mocap_targets.get(entity_name)
        if target is None:
            return self._get_current_mocap_pose(self.mocap_handles[entity_name])

        pos = np.asarray(target.position, dtype=np.float32)
        quat = np.asarray(target.wxyz, dtype=np.float32)
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-8:
            quat = quat / quat_norm
        return np.concatenate([pos, quat], axis=0)

    def _format_mocap_pose(self, pose):
        return (
            f"xyz:  {pose[0]: .6f}  {pose[1]: .6f}  {pose[2]: .6f}\n"
            f"quat: {pose[3]: .6f}  {pose[4]: .6f}  {pose[5]: .6f}  {pose[6]: .6f}"
        )

    def _update_mocap_pose_text(self, entity_name, pose):
        pose_text = self.mocap_pose_texts.get(entity_name)
        if pose_text is not None:
            pose_text.value = self._format_mocap_pose(pose)

    def _print_current_control_poses(self):
        for entity_name in self.mocap_targets:
            pose = self._get_current_control_pose(entity_name)
            print(f"[transform_control_pose:{entity_name}]")
            print(f"pos:  [{pose[0]:.9f}, {pose[1]:.9f}, {pose[2]:.9f}]")
            print(f"quat: [{pose[3]:.9f}, {pose[4]:.9f}, {pose[5]:.9f}, {pose[6]:.9f}]")

    def _print_current_slider_controls(self):
        print("[slider_policy_controls]")
        for slider, meta in zip(self.sliders, self._slider_meta, strict=False):
            scale = meta["scale"] if abs(meta["scale"]) > 1e-8 else 1.0
            control_value = float(slider.value)
            raw_action = (control_value - meta["offset"]) / scale
            kind = meta.get("kind", "action")
            print(
                f"{kind}:{meta['label']}: {control_value:.9f} "
                f"(raw_action={raw_action:.9f}, term={meta.get('term')}, entity={meta.get('entity')})"
            )

    def _write_mocap_pose_from_control(self):
        for entity_name, entity in self.mocap_handles.items():
            control_pose = self._get_current_control_pose(entity_name)
            pose = torch.tensor(
                control_pose,
                device=self.device,
                dtype=torch.float32,
            ).repeat(self.num_envs, 1)
            entity.write_mocap_pose_to_sim(pose)
            self._update_mocap_pose_text(entity_name, control_pose)

    def _get_slider_action(self):
        values = []
        for slider, meta in zip(self.sliders, self._slider_meta, strict=False):
            # Convert slider physical value back to raw action
            scale = meta["scale"] if abs(meta["scale"]) > 1e-8 else 1.0
            raw = (slider.value - meta["offset"]) / scale
            values.append(raw)

        action = torch.tensor(
            values,
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)
        return action

    def __call__(self, obs):
        self._write_mocap_pose_from_control()
        return self._get_slider_action()

    def _reset_sliders_to_initial(self):
        for slider, meta in zip(self.sliders, self._slider_meta, strict=False):
            slider.value = float(meta["initial"])
        for entity_name, target in self.mocap_targets.items():
            pose = self._get_current_mocap_pose(self.mocap_handles[entity_name])
            target.position = pose[:3]
            target.wxyz = pose[3:]
            self._update_mocap_pose_text(entity_name, pose)

    def _install_reset_hook(self):
        if getattr(self.env, "_viser_slider_reset_hook_installed", False):
            return

        original_reset = self.env.reset

        def reset_with_slider_sync(*args, **kwargs):
            out = original_reset(*args, **kwargs)
            self._reset_sliders_to_initial()
            self.env.action_manager.process_action(self._get_slider_action())
            return out

        self.env.reset = reset_with_slider_sync
        self.env._viser_slider_reset_hook_installed = True
