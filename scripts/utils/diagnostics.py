"""JSONL episode logging and evaluation diagnostic masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from mjlab.rl import RslRlVecEnvWrapper


def done_vector(done: Any, *, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(done, device=device, dtype=torch.bool).reshape(-1)


def _termination_vectors(env: RslRlVecEnvWrapper) -> dict[str, torch.Tensor]:
    manager = getattr(env.unwrapped, "termination_manager", None)
    if manager is None:
        return {}
    vectors = {}
    for name in getattr(manager, "_term_names", ()):
        try:
            value = torch.as_tensor(
                manager.get_term(name),
                device=env.unwrapped.device,
                dtype=torch.bool,
            ).reshape(-1)
        except Exception:
            continue
        if value.numel() == env.unwrapped.num_envs:
            vectors[str(name)] = value
    return vectors


def nonzero_dimension_counts(values: Any) -> dict[int, int]:
    return {
        index: int(value)
        for index, value in enumerate(np.asarray(values).reshape(-1))
        if int(value) > 0
    }


class JsonlDiagnostics:
    def __init__(self, path: Path | None) -> None:
        self.path = path.expanduser().resolve() if path is not None else None
        self._stream = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8")
            print(f"Episode diagnostics: {self.path}")

    def write(self, record: dict[str, Any]) -> None:
        if self._stream is None:
            return
        self._stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
