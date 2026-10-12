"""Observation access shared by scripted policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def observation_tensor(
    observations: Any,
    key: str,
    device: torch.device | str,
    num_envs: int,
    *,
    select_first: bool = False,
) -> torch.Tensor:
    """Read a flat or grouped observation as a float32 tensor.

    Supports dictionaries and TensorDict tuple keys. Unbatched vectors are
    repeated across environments, and singleton item axes are removed. Set
    ``select_first`` when a policy explicitly uses the first of multiple items;
    otherwise observations such as multiple button poses keep their item axis.
    """
    value = observations.get(key)
    if value is None:
        value = observations.get((key, key))
    if isinstance(value, Mapping):
        value = value.get(key)
    if value is None:
        raise KeyError(f"Observation missing {key!r}.")

    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).repeat(num_envs, 1)
    elif tensor.ndim == 3 and (select_first or tensor.shape[1] == 1):
        tensor = tensor[:, 0, :]
    return tensor
