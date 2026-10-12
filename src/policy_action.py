"""Canonical raw action layouts shared by rendering and policy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


_SINGLE_POSE_POSITION_SLICES = ((0, 3),)
_SINGLE_POSE_QUATERNION_SLICES = ((3, 7),)
_XHAND_TUBE_INSERT_TASK_ID = "XHand-Tube-Insert"
_XHAND_TUBE_INSERT_ACTION_TRANSFORM = "right_palm_site_to_real_ur_tcp"

# Fixed poses fitted in assets/mjcf/robot/ur5e_xhand.xml. The environment controls
# xhand:right_palm; these first map through the simulated UR5e wrist frame.
_RIGHT_PALM_TO_WRIST_3_POS = (
    0.00039365418650055367,
    -0.0004044972371598953,
    -0.12549931,
)
_RIGHT_PALM_TO_WRIST_3_QUAT = (
    0.6532814824381811,
    0.653281482438181,
    0.27059805007311605,
    0.27059805007311605,
)
_WRIST_3_TO_RIGHT_PALM_POS = (
    7.667194649993189e-06,
    0.12549931,
    -0.0005643782840839793,
)
_WRIST_3_TO_RIGHT_PALM_QUAT = (
    0.6532814824381811,
    -0.653281482438181,
    -0.27059805007311605,
    -0.27059805007311605,
)

# Map the simulated task world into the real UR controller base frame. The
# simulated robot root is at (0, 0.30, 0.824), Rz(pi), while the real MJCF base
# matches the controller after Rz(pi / 2).
_SIM_WORLD_TO_REAL_BASE_POS = (-0.3, 0.0, -0.824)
_SIM_WORLD_TO_REAL_BASE_QUAT = (
    0.7071067811865476,
    0.0,
    0.0,
    -0.7071067811865476,
)
_REAL_BASE_TO_SIM_WORLD_POS = (0.0, 0.3, 0.824)
_REAL_BASE_TO_SIM_WORLD_QUAT = (
    0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865476,
)

# The real UR API's configured TCP, recovered from the recorded qpos/action
# pairs: 10 cm along wrist +Y and Rx(-pi / 2).
_WRIST_3_TO_REAL_TCP_POS = (0.0, 0.1, 0.0)
_WRIST_3_TO_REAL_TCP_QUAT = (
    0.7071067811865476,
    -0.7071067811865476,
    0.0,
    0.0,
)
_REAL_TCP_TO_WRIST_3_POS = (0.0, 0.0, -0.1)
_REAL_TCP_TO_WRIST_3_QUAT = (
    0.7071067811865476,
    0.7071067811865476,
    0.0,
    0.0,
)

# Direct fixed mount transforms used by the current policy path. These are the
# compositions right_palm -> wrist_3 -> real controller TCP and its inverse.
_RIGHT_PALM_TO_REAL_TCP_POS = (
    0.0003936541865005398,
    -0.00040449723715988144,
    -0.02549931,
)
_RIGHT_PALM_TO_REAL_TCP_QUAT = (
    0.9238795325112765,
    0.0,
    0.0,
    0.38268343236511465,
)
_REAL_TCP_TO_RIGHT_PALM_POS = (
    7.667194649993189e-06,
    0.0005643782840839551,
    0.02549931,
)
_REAL_TCP_TO_RIGHT_PALM_QUAT = (
    0.9238795325112765,
    0.0,
    0.0,
    -0.38268343236511465,
)

TASK_ACTION_POSITION_SLICES: dict[str, tuple[tuple[int, int], ...]] = {
    "Aloha-Screw-Tighten": ((0, 3), (8, 11)),
    "Shadow-Centrifuge-Insert": _SINGLE_POSE_POSITION_SLICES,
    "Shadow-Pipette-Insert": _SINGLE_POSE_POSITION_SLICES,
    "Shadow-Pipette-Manipulate": _SINGLE_POSE_POSITION_SLICES,
    "Shadow-Tube-Insert": _SINGLE_POSE_POSITION_SLICES,
    "Shadow-Tube-Rotate": ((0, 3), (25, 28)),
    "Wuji-Centrifuge-Close": _SINGLE_POSE_POSITION_SLICES,
    "Wuji-HeatingPlate-Rotate": _SINGLE_POSE_POSITION_SLICES,
    "Wuji-Tweezers-Manipulate": _SINGLE_POSE_POSITION_SLICES,
    "XHand-Bunsen-Manipulate": _SINGLE_POSE_POSITION_SLICES,
    "XHand-MixerPanel-Manipulate": _SINGLE_POSE_POSITION_SLICES,
    "XHand-ThermalCycler-Close": _SINGLE_POSE_POSITION_SLICES,
    "XHand-ThermalCycler-Open": _SINGLE_POSE_POSITION_SLICES,
    "XHand-Tube-Insert": _SINGLE_POSE_POSITION_SLICES,
}

TASK_ACTION_QUATERNION_SLICES: dict[str, tuple[tuple[int, int], ...]] = {
    task_id: _SINGLE_POSE_QUATERNION_SLICES
    for task_id in TASK_ACTION_POSITION_SLICES
}
TASK_ACTION_QUATERNION_SLICES["Aloha-Screw-Tighten"] = ((3, 7), (11, 15))
TASK_ACTION_QUATERNION_SLICES["Shadow-Tube-Rotate"] = ((3, 7), (28, 32))


def action_position_slices(task_id: str) -> tuple[tuple[int, int], ...]:
    try:
        return TASK_ACTION_POSITION_SLICES[task_id]
    except KeyError as exc:
        raise ValueError(f"No raw action layout is registered for task {task_id!r}.") from exc


def action_quaternion_slices(task_id: str) -> tuple[tuple[int, int], ...]:
    try:
        return TASK_ACTION_QUATERNION_SLICES[task_id]
    except KeyError as exc:
        raise ValueError(f"No raw action layout is registered for task {task_id!r}.") from exc


def validate_action_layout_attrs(attrs: Any, task_id: str) -> None:
    """Require serialized action-layout metadata to match the task registry."""
    for attr_name, expected in (
        ("action_position_slices", action_position_slices(task_id)),
        ("action_quaternion_slices", action_quaternion_slices(task_id)),
    ):
        if attr_name not in attrs:
            raise KeyError(f"Dataset has no required {attr_name!r} attribute.")
        actual = tuple(tuple(int(index) for index in value) for value in attrs[attr_name])
        if actual != expected:
            raise ValueError(
                f"{task_id!r} {attr_name} is {actual}, expected {expected}."
            )


def action_transform_name(task_id: str) -> str:
    if task_id == _XHAND_TUBE_INSERT_TASK_ID:
        return _XHAND_TUBE_INSERT_ACTION_TRANSFORM
    return "none"


def _compose_first_action_pose(
    action: Any,
    relative_pos: tuple[float, float, float],
    relative_quat: tuple[float, float, float, float],
) -> Any:
    output = (
        action.clone()
        if isinstance(action, torch.Tensor)
        else np.array(action, copy=True)
    )
    quat = action[..., 3:7]

    if isinstance(action, torch.Tensor):
        pos_offset = action.new_tensor(relative_pos).expand_as(action[..., :3])
        quat_offset = action.new_tensor(relative_quat).expand_as(quat)
        twice_cross = 2.0 * torch.cross(quat[..., 1:4], pos_offset, dim=-1)
        rotated_offset = (
            pos_offset
            + quat[..., :1] * twice_cross
            + torch.cross(quat[..., 1:4], twice_cross, dim=-1)
        )
    else:
        pos_offset = np.asarray(relative_pos, dtype=action.dtype)
        quat_offset = np.asarray(relative_quat, dtype=action.dtype)
        twice_cross = 2.0 * np.cross(quat[..., 1:4], pos_offset)
        rotated_offset = (
            pos_offset
            + quat[..., :1] * twice_cross
            + np.cross(quat[..., 1:4], twice_cross)
        )
    output[..., :3] = action[..., :3] + rotated_offset

    w1, x1, y1, z1 = (quat[..., index] for index in range(4))
    w2, x2, y2, z2 = (quat_offset[..., index] for index in range(4))
    components = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    output[..., 3:7] = (
        torch.stack(components, dim=-1)
        if isinstance(action, torch.Tensor)
        else np.stack(components, axis=-1)
    )
    return output


def _left_compose_first_action_pose(
    action: Any,
    parent_pos: tuple[float, float, float],
    parent_quat: tuple[float, float, float, float],
) -> Any:
    """Apply a fixed world-frame transform before the first action pose."""
    output = action.clone() if isinstance(action, torch.Tensor) else np.array(action, copy=True)
    pos = action[..., :3]
    quat = action[..., 3:7]

    if isinstance(action, torch.Tensor):
        pos_offset = action.new_tensor(parent_pos)
        quat_offset = action.new_tensor(parent_quat)
        twice_cross = 2.0 * torch.cross(quat_offset[1:4].expand_as(pos), pos, dim=-1)
        rotated_pos = (
            pos
            + quat_offset[:1] * twice_cross
            + torch.cross(quat_offset[1:4].expand_as(pos), twice_cross, dim=-1)
        )
    else:
        pos_offset = np.asarray(parent_pos, dtype=action.dtype)
        quat_offset = np.asarray(parent_quat, dtype=action.dtype)
        quat_vector = np.broadcast_to(quat_offset[1:4], pos.shape)
        twice_cross = 2.0 * np.cross(quat_vector, pos)
        rotated_pos = pos + quat_offset[0] * twice_cross + np.cross(quat_vector, twice_cross)
    output[..., :3] = rotated_pos + pos_offset

    w1, x1, y1, z1 = quat_offset
    w2, x2, y2, z2 = (quat[..., index] for index in range(4))
    components = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    output[..., 3:7] = (
        torch.stack(components, dim=-1)
        if isinstance(action, torch.Tensor)
        else np.stack(components, axis=-1)
    )
    return output


def simulation_wrist_action_to_real_tcp_action(action: Any) -> Any:
    """Convert simulated-world wrist targets to real-base UR TCP targets."""
    real_wrist = _left_compose_first_action_pose(
        action,
        _SIM_WORLD_TO_REAL_BASE_POS,
        _SIM_WORLD_TO_REAL_BASE_QUAT,
    )
    return _compose_first_action_pose(
        real_wrist,
        _WRIST_3_TO_REAL_TCP_POS,
        _WRIST_3_TO_REAL_TCP_QUAT,
    )


def simulation_palm_action_to_real_tcp_action(action: Any) -> Any:
    """Convert legacy simulated-world palm targets to real-base UR TCP targets."""
    real_palm = _left_compose_first_action_pose(
        action,
        _SIM_WORLD_TO_REAL_BASE_POS,
        _SIM_WORLD_TO_REAL_BASE_QUAT,
    )
    return _compose_first_action_pose(
        real_palm,
        _RIGHT_PALM_TO_REAL_TCP_POS,
        _RIGHT_PALM_TO_REAL_TCP_QUAT,
    )


def real_tcp_action_to_simulation_wrist_action(action: Any) -> Any:
    """Convert real-base UR TCP targets back to simulated-world wrist targets."""
    real_wrist = _compose_first_action_pose(
        action,
        _REAL_TCP_TO_WRIST_3_POS,
        _REAL_TCP_TO_WRIST_3_QUAT,
    )
    return _left_compose_first_action_pose(
        real_wrist,
        _REAL_BASE_TO_SIM_WORLD_POS,
        _REAL_BASE_TO_SIM_WORLD_QUAT,
    )


def simulation_world_vector_to_real_base_vector(vector: Any) -> Any:
    """Rotate displacement vectors from simulation world axes to real base axes."""
    if vector.shape[-1] != 3:
        raise ValueError(f"Expected vectors with trailing dimension 3, got {vector.shape}.")
    if isinstance(vector, torch.Tensor):
        return torch.stack((vector[..., 1], -vector[..., 0], vector[..., 2]), dim=-1)
    vector = np.asarray(vector)
    return np.stack((vector[..., 1], -vector[..., 0], vector[..., 2]), axis=-1)


def environment_action_to_policy_action(action: Any, task_id: str) -> Any:
    """Convert the environment control pose to the pose learned by the policy."""
    if task_id != _XHAND_TUBE_INSERT_TASK_ID:
        return action
    return _compose_first_action_pose(
        action,
        _RIGHT_PALM_TO_REAL_TCP_POS,
        _RIGHT_PALM_TO_REAL_TCP_QUAT,
    )


def policy_action_to_environment_action(action: Any, task_id: str) -> Any:
    """Convert a learned-policy pose to the pose expected by the environment."""
    if task_id != _XHAND_TUBE_INSERT_TASK_ID:
        return action
    return _compose_first_action_pose(
        action,
        _REAL_TCP_TO_RIGHT_PALM_POS,
        _REAL_TCP_TO_RIGHT_PALM_QUAT,
    )


def shift_action_positions(
    action: Any,
    origins: Any,
    task_id: str,
    *,
    scale: float,
) -> Any:
    """Shift every absolute Cartesian action branch by its environment origin."""
    if action.ndim != 2:
        raise ValueError(f"Action must have shape (batch, dim), got {tuple(action.shape)}.")
    slices = action_position_slices(task_id)
    if not slices:
        return action

    if isinstance(action, torch.Tensor):
        adjusted = action.clone()
        origin_values = torch.as_tensor(origins, device=action.device, dtype=action.dtype)
    else:
        adjusted = np.array(action, copy=True)
        origin_values = np.asarray(origins, dtype=adjusted.dtype)
    if origin_values.shape != (action.shape[0], 3):
        raise ValueError(
            f"Environment origins must have shape ({action.shape[0]}, 3), "
            f"got {tuple(origin_values.shape)}."
        )
    for start, stop in slices:
        if stop - start != 3 or start < 0 or stop > action.shape[1]:
            raise ValueError(
                f"Invalid Cartesian action slice ({start}, {stop}) for "
                f"{task_id!r} action dim {action.shape[1]}."
            )
        adjusted[:, start:stop] += origin_values * scale
    return adjusted
