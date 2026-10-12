"""Deterministic per-environment episode quotas for vectorized evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def balanced_episode_targets(num_envs: int, num_episodes: int) -> np.ndarray:
    """Distribute evaluation episodes as evenly as possible across environments."""
    if num_envs < 1 or num_episodes < 1:
        raise ValueError("num_envs and num_episodes must be positive.")
    base, remainder = divmod(num_episodes, num_envs)
    targets = np.full(num_envs, base, dtype=np.int64)
    targets[:remainder] += 1
    return targets


def countable_finished_envs(
    finished_env_ids: Iterable[int],
    completed_per_env: np.ndarray,
    target_per_env: np.ndarray,
) -> list[int]:
    """Return terminal environments that still have an unfilled episode quota."""
    return [
        int(env_id)
        for env_id in finished_env_ids
        if completed_per_env[int(env_id)] < target_per_env[int(env_id)]
    ]

