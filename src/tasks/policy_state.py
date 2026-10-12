"""Canonical robot state used by collection and learned-policy evaluation."""

from __future__ import annotations

from typing import Any

import torch

# Most tasks expose their controlled robot as ``scene["robot"]``. These two
# tasks control two independent entities, whose joint positions form one policy
# state in this explicit order.
DUAL_ARM_STATE_ENTITIES: dict[str, tuple[str, str]] = {
    "Aloha-Screw-Tighten": ("aloha_1", "aloha_2"),
    "Shadow-Tube-Rotate": ("robot", "tube_holder_robot"),
}


def joint_position_tensor(scene: Any, entity_name: str) -> torch.Tensor:
    value = scene[entity_name].data.joint_pos
    if hasattr(value, "_tensor"):
        value = value._tensor
    if isinstance(value, torch.Tensor):
        tensor = value
    elif hasattr(value, "__dlpack__"):
        tensor = torch.utils.dlpack.from_dlpack(value)
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim != 2:
        raise ValueError(
            f"Scene entity {entity_name!r} joint_pos must have shape "
            f"(num_envs, joints), got {tuple(tensor.shape)}."
        )
    return tensor


def policy_joint_positions(env: Any, task_id: str) -> torch.Tensor:
    """Return the qpos policy state in the collector's canonical entity order."""
    entity_names = DUAL_ARM_STATE_ENTITIES.get(task_id, ("robot",))
    states = [joint_position_tensor(env.scene, name) for name in entity_names]
    batch_sizes = {int(state.shape[0]) for state in states}
    if len(batch_sizes) != 1:
        shapes = dict(zip(entity_names, (tuple(state.shape) for state in states), strict=True))
        raise ValueError(f"Policy-state entity batch sizes do not match: {shapes}.")
    return torch.cat(states, dim=-1) if len(states) > 1 else states[0]
