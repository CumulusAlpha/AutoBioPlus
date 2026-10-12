"""Episode buffering and balanced reset-goal scheduling."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from src.task_goals import goal_variant_key, task_goal_balance_mode, task_goal_variants


class EpisodeBuffer:
    """Per-env time-major trajectory buffer."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def append(self, step: dict[str, Any]) -> None:
        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)


class BalancedGoalScheduler:
    """Assign reset goals while enforcing exact successful-episode quotas."""

    def __init__(
        self,
        task_id: str,
        *,
        episodes_per_variant: int,
        seed: int,
        existing_goals: Sequence[Sequence[float | int]],
    ) -> None:
        variants = task_goal_variants(task_id)
        if variants is None:
            raise ValueError(f"Task {task_id!r} has no balanced goal catalog.")
        if episodes_per_variant < 1:
            raise ValueError("episodes_per_variant must be positive.")

        self.task_id = task_id
        self.variants = tuple(goal_variant_key(values) for values in variants)
        self.variant_set = frozenset(self.variants)
        self.episodes_per_variant = int(episodes_per_variant)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.saved_counts = {variant: 0 for variant in self.variants}
        self.active_counts = {variant: 0 for variant in self.variants}
        self.active_by_env: dict[int, tuple[float, ...]] = {}

        for values in existing_goals:
            key = goal_variant_key(values)
            if key not in self.saved_counts:
                raise ValueError(
                    f"Existing collection contains non-canonical goal {key} for {task_id!r}."
                )
            self.saved_counts[key] += 1
        over_quota = {
            key: count
            for key, count in self.saved_counts.items()
            if count > self.episodes_per_variant
        }
        if over_quota:
            raise ValueError(
                "Existing collection exceeds the requested balanced quota; "
                f"recollect with --overwrite. First entries: {list(over_quota.items())[:5]}"
            )

    @property
    def target_episodes(self) -> int:
        return len(self.variants) * self.episodes_per_variant

    @property
    def completed_episodes(self) -> int:
        return sum(self.saved_counts.values())

    @property
    def complete(self) -> bool:
        return self.completed_episodes == self.target_episodes

    def __call__(self, task_id: str, env_ids: torch.Tensor) -> torch.Tensor:
        if task_id != self.task_id:
            raise ValueError(
                f"Goal scheduler for {self.task_id!r} received {task_id!r}."
            )

        assigned: list[tuple[float, ...]] = []
        for env_id in env_ids.detach().cpu().reshape(-1).tolist():
            env_id = int(env_id)
            previous = self.active_by_env.pop(env_id, None)
            if previous is not None:
                self.active_counts[previous] -= 1

            candidates = [
                variant
                for variant in self.variants
                if self.saved_counts[variant] < self.episodes_per_variant
            ]
            if not candidates:
                candidates = list(self.variants)
            scores = np.asarray(
                [
                    self.saved_counts[variant] + self.active_counts[variant]
                    for variant in candidates
                ],
                dtype=np.int64,
            )
            least_loaded = np.flatnonzero(scores == scores.min())
            selected = candidates[int(self.rng.choice(least_loaded))]
            self.active_by_env[env_id] = selected
            self.active_counts[selected] += 1
            assigned.append(selected)

        return torch.as_tensor(assigned, device=env_ids.device, dtype=torch.float32)

    def accept_success(self, values: Sequence[float | int]) -> bool:
        key = goal_variant_key(values)
        if key not in self.saved_counts:
            raise ValueError(f"Successful episode has non-canonical goal {key}.")
        if self.saved_counts[key] >= self.episodes_per_variant:
            return False
        self.saved_counts[key] += 1
        return True

    def metadata(self) -> dict[str, Any]:
        return {
            "goal_balanced": True,
            "goal_balance_mode": task_goal_balance_mode(self.task_id),
            "goal_variant_count": len(self.variants),
            "goal_episodes_per_variant": self.episodes_per_variant,
            "goal_balance_target_episodes": self.target_episodes,
            "goal_balance_seed": self.seed,
            "goal_variant_catalog": [list(values) for values in self.variants],
        }
