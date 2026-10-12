from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action whose dimensions follow ``actuator_names`` order."""

    preserve_order: bool = True

    def build(self, env: "ManagerBasedRlEnv") -> "OrderedJointPositionAction":
        return OrderedJointPositionAction(self, env)


class OrderedJointPositionAction(JointPositionAction):
    """Resolve joint targets in config order without patching mjlab."""

    cfg: OrderedJointPositionActionCfg

    def _find_targets(
        self, cfg: OrderedJointPositionActionCfg
    ) -> tuple[list[int], list[str]]:
        if not cfg.preserve_order:
            return super()._find_targets(cfg)
        return _find_actuated_joints_in_requested_order(
            self._entity,
            cfg.actuator_names,
        )


def _find_actuated_joints_in_requested_order(
    entity,
    name_keys: str | Sequence[str],
) -> tuple[list[int], list[str]]:
    """Resolve actuated joint ids, preserving the user-provided key order."""

    actuated_joint_names = set()
    for actuator in entity.actuators:
        actuated_joint_names.update(actuator.target_names)

    actuated_in_model_order = [
        name for name in entity.joint_names if name in actuated_joint_names
    ]
    _, matched_joint_names = entity.find_joints(
        name_keys,
        joint_subset=actuated_in_model_order,
        preserve_order=True,
    )

    name_to_entity_idx = {name: idx for idx, name in enumerate(entity.joint_names)}
    joint_ids = [name_to_entity_idx[name] for name in matched_joint_names]
    return joint_ids, matched_joint_names


@dataclass(kw_only=True)
class MocapPoseActionCfg(ActionTermCfg):
    """Relative mocap pose action: delta xyz plus delta wxyz quaternion."""

    pos_scale: float = 1.0
    quat_scale: float = 1.0
    default_pose: tuple[float, float, float, float, float, float, float] | None = None
    normalize_quat: bool = True

    def build(self, env: "ManagerBasedRlEnv") -> "MocapPoseAction":
        return MocapPoseAction(self, env)


class MocapPoseAction(ActionTerm):
    """Write a 7D policy action directly to a mocap body."""

    cfg: MocapPoseActionCfg

    def __init__(self, cfg: MocapPoseActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        if cfg.default_pose is None:
            default_pose = self._entity.data.root_link_pose_w.clone()
        else:
            default_pose = torch.tensor(cfg.default_pose, device=self.device, dtype=torch.float32)
            default_pose = default_pose.unsqueeze(0).repeat(self.num_envs, 1)
        self._default_pose = default_pose

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:, :3] = (
            self._default_pose[:, :3] + actions[:, :3] * float(self.cfg.pos_scale)
        )
        quat = self._default_pose[:, 3:7] + actions[:, 3:7] * float(self.cfg.quat_scale)
        if self.cfg.normalize_quat:
            quat = torch.nn.functional.normalize(quat, dim=-1)
        self._processed_actions[:, 3:7] = quat

    def apply_actions(self) -> None:
        self._entity.write_mocap_pose_to_sim(self._processed_actions)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._default_pose[env_ids]
        self._entity.write_mocap_pose_to_sim(self._processed_actions[env_ids], env_ids=env_ids)
