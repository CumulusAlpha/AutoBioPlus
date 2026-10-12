"""Shared terminal-state extraction for collection and evaluation."""

from __future__ import annotations

from typing import Any

import torch

SUCCESS_TERM_NAMES = ("success", "succeed", "is_success")


def as_bool_vector(
    value: Any, num_envs: int, device: torch.device | str
) -> torch.Tensor:
    """Convert a termination value to a flat mask; missing values mean false."""
    if value is None:
        return torch.zeros(num_envs, device=device, dtype=torch.bool)
    return torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)


def extract_success(env: Any, extras: dict[str, Any]) -> torch.Tensor:
    """Read success from step extras or the environment's termination terms."""
    raw_env = env.unwrapped
    for key in SUCCESS_TERM_NAMES:
        if key in extras:
            value = as_bool_vector(extras[key], raw_env.num_envs, raw_env.device)
            if value.numel() == raw_env.num_envs:
                return value

    manager = raw_env.termination_manager
    for key in SUCCESS_TERM_NAMES:
        try:
            term = manager.get_term(key)
        except (KeyError, ValueError):
            # Tasks may use any one of these names, or have no success term.
            continue
        value = as_bool_vector(term, raw_env.num_envs, raw_env.device)
        if value.numel() == raw_env.num_envs:
            return value

    return torch.zeros(raw_env.num_envs, device=raw_env.device, dtype=torch.bool)


def extract_time_out(env: Any, extras: dict[str, Any]) -> torch.Tensor:
    """Read timeouts from step extras, falling back to the environment mask."""
    raw_env = env.unwrapped
    for key in ("time_outs", "timeout", "time_out"):
        if key in extras:
            value = as_bool_vector(extras[key], raw_env.num_envs, raw_env.device)
            if value.numel() == raw_env.num_envs:
                return value
    return as_bool_vector(raw_env.reset_time_outs, raw_env.num_envs, raw_env.device)
