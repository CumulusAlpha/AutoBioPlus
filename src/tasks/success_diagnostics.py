"""Task-level success diagnostics retained across automatic environment resets."""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from typing import Any

import torch

from src.utils.quat_util import quat_rotate_vec

_STATE_ATTR = "_success_diagnostics"


def bool_vector(env: Any, value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=env.device, dtype=torch.bool).reshape(-1)
    if tensor.shape != (env.num_envs,):
        raise ValueError(
            f"Success diagnostic {name!r} has shape {tuple(tensor.shape)}, "
            f"expected ({env.num_envs},)."
        )
    return tensor.detach().clone()


def float_vector(env: Any, value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=env.device, dtype=torch.float32).reshape(-1)
    if tensor.shape != (env.num_envs,):
        raise ValueError(
            f"Success diagnostic {name!r} has shape {tuple(tensor.shape)}, "
            f"expected ({env.num_envs},)."
        )
    return tensor.detach().clone()


def record_success_diagnostics(
    env: Any,
    *,
    object_complete: Any,
    full_success: Any,
    final_motion_complete: Any | None = None,
    components: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> None:
    """Record current and episode-wide success components before auto-reset."""
    component_values = {
        "object_complete": bool_vector(
            env,
            object_complete,
            name="object_complete",
        ),
        "full_success": bool_vector(env, full_success, name="full_success"),
    }
    if final_motion_complete is not None:
        component_values["final_motion_complete"] = bool_vector(
            env,
            final_motion_complete,
            name="final_motion_complete",
        )
    for name, value in (components or {}).items():
        if name in component_values:
            raise ValueError(f"Duplicate success diagnostic component: {name!r}.")
        component_values[name] = bool_vector(env, value, name=name)

    metric_values = {
        name: float_vector(env, value, name=name)
        for name, value in (metrics or {}).items()
    }
    episode_length = getattr(env, "episode_length_buf", None)
    reset_mask = (
        torch.as_tensor(episode_length, device=env.device).reshape(-1) <= 1
        if episode_length is not None
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )

    state = getattr(env, _STATE_ATTR, None)
    if not isinstance(state, dict) or state.get("num_envs") != env.num_envs:
        state = {
            "num_envs": env.num_envs,
            "components": {},
            "components_ever": {},
            "metrics": {},
            "metrics_min": {},
            "metrics_max": {},
            "thresholds": {},
        }

    for name, value in component_values.items():
        previous = state["components_ever"].get(name)
        if previous is None or previous.shape != value.shape:
            previous = torch.zeros_like(value)
        state["components"][name] = value
        state["components_ever"][name] = torch.where(
            reset_mask,
            value,
            previous | value,
        )

    for name, value in metric_values.items():
        previous_min = state["metrics_min"].get(name)
        previous_max = state["metrics_max"].get(name)
        if previous_min is None or previous_min.shape != value.shape:
            previous_min = value
        if previous_max is None or previous_max.shape != value.shape:
            previous_max = value
        state["metrics"][name] = value
        state["metrics_min"][name] = torch.where(
            reset_mask,
            value,
            torch.minimum(previous_min, value),
        )
        state["metrics_max"][name] = torch.where(
            reset_mask,
            value,
            torch.maximum(previous_max, value),
        )

    state["thresholds"] = {
        name: float(value) for name, value in (thresholds or {}).items()
    }
    setattr(env, _STATE_ATTR, state)


def stable_success(
    env: Any,
    instantaneous_success: torch.Tensor,
    *,
    state_attr: str,
    required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require a physical terminal condition to persist for consecutive steps."""
    if required_steps < 1:
        raise ValueError(f"required_steps must be positive, got {required_steps}.")

    instantaneous_success = torch.as_tensor(
        instantaneous_success,
        device=env.device,
        dtype=torch.bool,
    ).reshape(-1)
    if instantaneous_success.shape != (env.num_envs,):
        raise ValueError(
            f"Success condition has shape {tuple(instantaneous_success.shape)}, "
            f"expected ({env.num_envs},)."
        )

    count = getattr(env, state_attr, None)
    if count is None or count.shape != instantaneous_success.shape:
        count = torch.zeros_like(instantaneous_success, dtype=torch.long)
    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is not None:
        reset_mask = torch.as_tensor(
            episode_length,
            device=env.device,
        ).reshape(-1) <= 1
        count = torch.where(reset_mask, torch.zeros_like(count), count)
    count = torch.where(
        instantaneous_success,
        count + 1,
        torch.zeros_like(count),
    )
    setattr(env, state_attr, count)
    return count >= required_steps, count


def axis_tilt_from_quats(
    quat: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """Return the angle between the local +Z axes of two WXYZ quaternions."""
    quat = torch.as_tensor(quat)
    target_quat = torch.as_tensor(
        target_quat,
        device=quat.device,
        dtype=quat.dtype,
    )
    if quat.shape != target_quat.shape or quat.shape[-1] != 4:
        raise ValueError(
            "Expected matching quaternion tensors ending in dimension 4, got "
            f"{tuple(quat.shape)} and {tuple(target_quat.shape)}."
        )
    local_z = torch.zeros((*quat.shape[:-1], 3), device=quat.device, dtype=quat.dtype)
    local_z[..., 2] = 1.0
    axis = quat_rotate_vec(quat, local_z)
    target_axis = quat_rotate_vec(target_quat, local_z)
    alignment = torch.sum(axis * target_axis, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(alignment)


def json_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def episode_outcome(components_ever: dict[str, bool]) -> str:
    object_complete = components_ever.get("object_complete", False)
    full_success = components_ever.get("full_success", False)
    if full_success:
        return "full_success"
    if "final_motion_complete" not in components_ever:
        return (
            "object_complete_without_full_success"
            if object_complete
            else "object_incomplete"
        )
    final_motion_complete = components_ever["final_motion_complete"]
    if object_complete and not final_motion_complete:
        return "object_complete_final_motion_incomplete"
    if object_complete:
        return "object_and_final_motion_never_coincided"
    if final_motion_complete:
        return "final_motion_complete_object_incomplete"
    return "object_incomplete"


def success_diagnostics_for_envs(
    env: Any,
    env_ids: Any,
) -> dict[int, dict[str, Any]]:
    """Return JSON-safe terminal snapshots captured before environment reset."""
    state = getattr(env, _STATE_ATTR, None)
    if not isinstance(state, dict):
        return {}
    ids = [
        int(value)
        for value in torch.as_tensor(env_ids, device="cpu").reshape(-1).tolist()
    ]
    result = {}
    for env_id in ids:
        components = {
            name: bool(value[env_id].item())
            for name, value in state["components"].items()
        }
        components_ever = {
            name: bool(value[env_id].item())
            for name, value in state["components_ever"].items()
        }
        result[env_id] = {
            "outcome": episode_outcome(components_ever),
            "components": components,
            "components_ever": components_ever,
            "unmet_at_terminal": sorted(
                name for name, complete in components.items() if not complete
            ),
            "never_satisfied": sorted(
                name for name, complete in components_ever.items() if not complete
            ),
            "regressed_before_terminal": sorted(
                name
                for name, complete in components.items()
                if components_ever.get(name, False) and not complete
            ),
            "metrics": {
                name: json_float(float(value[env_id].item()))
                for name, value in state["metrics"].items()
            },
            "metrics_min": {
                name: json_float(float(value[env_id].item()))
                for name, value in state["metrics_min"].items()
            },
            "metrics_max": {
                name: json_float(float(value[env_id].item()))
                for name, value in state["metrics_max"].items()
            },
            "thresholds": dict(state["thresholds"]),
        }
    return result


def update_success_diagnostic_counts(
    counts: MutableMapping[str, int],
    snapshot: dict[str, Any],
) -> None:
    """Accumulate episode outcomes for object-only versus full success."""
    if not snapshot:
        return
    ever = snapshot.get("components_ever", {})
    terminal = snapshot.get("components", {})
    object_complete = bool(ever.get("object_complete", False))
    full_success = bool(ever.get("full_success", False))
    outcome = snapshot.get("outcome") or episode_outcome(ever)
    increments = {
        "episodes_with_diagnostics": 1,
        f"outcome::{outcome}": 1,
        "object_complete_episodes": int(object_complete),
        "full_success_component_episodes": int(full_success),
        "object_complete_without_full_success": int(
            object_complete and not full_success
        ),
    }
    if "final_motion_complete" in ever:
        final_motion_complete = bool(ever["final_motion_complete"])
        increments.update(
            {
                "episodes_with_final_motion_gate": 1,
                "final_motion_complete_episodes": int(final_motion_complete),
                "object_complete_without_final_motion": int(
                    object_complete and not final_motion_complete
                ),
                "object_and_final_motion_never_coincided": int(
                    object_complete and final_motion_complete and not full_success
                ),
            }
        )
    for name, complete in terminal.items():
        increments[f"component_terminal::{name}"] = int(bool(complete))
    for name, complete in ever.items():
        increments[f"component_ever::{name}"] = int(bool(complete))
        increments[f"component_regressed::{name}"] = int(
            bool(complete) and not bool(terminal.get(name, False))
        )
    for name, increment in increments.items():
        counts[name] = counts.get(name, 0) + increment


def grouped_success_diagnostic_counts(
    counts: MutableMapping[str, int],
) -> dict[str, dict[str, int]]:
    """Group dynamic outcome/component counters for concise evaluator output."""
    prefixes = {
        "outcomes": "outcome::",
        "components_ever": "component_ever::",
        "components_terminal": "component_terminal::",
        "components_regressed": "component_regressed::",
    }
    return {
        group: {
            name.removeprefix(prefix): int(value)
            for name, value in sorted(counts.items())
            if name.startswith(prefix) and int(value) > 0
        }
        for group, prefix in prefixes.items()
    }
