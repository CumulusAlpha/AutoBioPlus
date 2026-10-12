"""Simulation integration for the shared DexBioLab language-goal contracts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from src.task_goals import MIXER_FREQ_VALUES
from src.task_goals import MIXER_TEMP_VALUES
from src.task_goals import MIXER_TIME_VALUES
from src.task_goals import TASK_GOAL_BALANCE_MODES
from src.task_goals import TASK_GOAL_DEFAULT_EPISODES_PER_VARIANT
from src.task_goals import TASK_GOAL_NAMES
from src.task_goals import TASK_GOAL_NORMALIZATION_BOUNDS
from src.task_goals import TASK_GOAL_VARIANTS
from src.task_goals import TASK_PROMPT_TEMPLATES
from src.task_goals import default_goal_episodes_per_variant
from src.task_goals import expected_collection_episodes
from src.task_goals import format_task_prompt
from src.task_goals import goal_balance_provenance
from src.task_goals import goal_variant_key
from src.task_goals import task_goal_balance_mode
from src.task_goals import task_goal_names
from src.task_goals import task_goal_variants
from src.task_goals import validate_balanced_goal_metadata
from src.task_goals import validate_goal_balance_contract
from src.task_goals import validate_goal_balance_provenance


_TASK_GOAL_PROVIDER_ATTR = "_task_goal_provider"


def install_task_goal_provider(env: Any, provider: Any | None) -> None:
    """Install a collection-only goal provider used by task reset events."""
    if provider is None:
        if hasattr(env, _TASK_GOAL_PROVIDER_ATTR):
            delattr(env, _TASK_GOAL_PROVIDER_ATTR)
        return
    setattr(env, _TASK_GOAL_PROVIDER_ATTR, provider)


def requested_task_goal_values(
    env: Any,
    task_id: str,
    env_ids: torch.Tensor,
) -> torch.Tensor | None:
    """Request reset goals from a collector, falling back to normal task sampling."""
    provider = getattr(env, _TASK_GOAL_PROVIDER_ATTR, None)
    if provider is None:
        return None
    values = torch.as_tensor(
        provider(task_id, env_ids),
        device=env.device,
        dtype=torch.float32,
    )
    expected = (int(env_ids.numel()), len(TASK_GOAL_NAMES[task_id]))
    if tuple(values.shape) != expected:
        raise ValueError(
            f"Goal provider returned shape {tuple(values.shape)} for {task_id!r}; "
            f"expected {expected}."
        )
    return values


def extract_task_goal_values(env: Any, task_id: str) -> torch.Tensor | None:
    """Read the current reset-selected goal for every vectorized environment."""
    num_envs = int(env.num_envs)
    device = env.device

    if task_id == "XHand-Tube-Insert":
        values = getattr(env, "xhand_tube_insert_target_indices", None)
    elif task_id == "Shadow-Tube-Insert":
        values = getattr(env, "shadow_tube_insert_target_indices", None)
    elif task_id == "XHand-MixerPanel-Manipulate":
        values = getattr(env, "mixer_ui_target_state", None)
    else:
        return None

    if values is None:
        raise RuntimeError(f"Episode goal for {task_id!r} was not initialized by reset.")
    values = torch.as_tensor(values, device=device)
    if values.ndim == 1:
        values = values.unsqueeze(0).expand(num_envs, -1)
    expected_dim = len(TASK_GOAL_NAMES[task_id])
    if values.ndim != 2 or values.shape[0] < num_envs or values.shape[1] != expected_dim:
        raise ValueError(
            f"Episode goal for {task_id!r} must have shape ({num_envs}, {expected_dim}), "
            f"got {tuple(values.shape)}."
        )
    return values[:num_envs].clone()


def normalized_task_goal_values(env: Any, task_id: str) -> torch.Tensor | None:
    """Read and normalize current per-environment goals for learned policies."""
    values = extract_task_goal_values(env, task_id)
    if values is None:
        return None
    bounds = TASK_GOAL_NORMALIZATION_BOUNDS[task_id]
    lower = torch.tensor([item[0] for item in bounds], device=values.device, dtype=torch.float32)
    upper = torch.tensor([item[1] for item in bounds], device=values.device, dtype=torch.float32)
    values = values.to(dtype=torch.float32)
    if torch.any(values < lower) or torch.any(values > upper):
        raise ValueError(f"Current goal for {task_id!r} is outside its normalization bounds.")
    return (values - lower) / (upper - lower)


def task_prompts_for_envs(
    env: Any,
    task_id: str,
    env_ids: Iterable[int] | torch.Tensor | None = None,
    *,
    override: str | None = None,
) -> list[str]:
    """Return prompts aligned with the requested vectorized environment lanes."""
    if env_ids is None:
        ids = list(range(int(env.num_envs)))
    elif isinstance(env_ids, torch.Tensor):
        ids = [int(value) for value in env_ids.detach().cpu().reshape(-1).tolist()]
    else:
        ids = [int(value) for value in env_ids]

    if override is not None:
        prompt = override.strip()
        if not prompt:
            raise ValueError("Prompt override cannot be empty.")
        return [prompt] * len(ids)

    goals = extract_task_goal_values(env, task_id)
    if goals is None:
        return [format_task_prompt(task_id)] * len(ids)
    goals_cpu = goals.detach().cpu()
    return [format_task_prompt(task_id, goals_cpu[env_id].tolist()) for env_id in ids]
