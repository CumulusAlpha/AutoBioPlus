"""Array conversion for independent simulation-state snapshots."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def snapshot_to_numpy(value: Any, *, name: str | None = None) -> Any:
    """Copy a tensor or array to CPU without sharing simulation buffers.

    Scalars pass through unchanged. Object arrays are rejected because replay
    snapshots must be serializable without pickle.
    """
    # mjlab's TorchArray proxies delegate tensor methods, but their CUDA-backed
    # numpy() still requires an explicit detach and transfer to CPU.
    value = getattr(value, "_tensor", value)
    if isinstance(value, torch.Tensor):
        return np.array(value.detach().cpu().numpy(), copy=True)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value

    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "numpy"):
        array = value.numpy()
    elif hasattr(value, "__dlpack__"):
        array = torch.utils.dlpack.from_dlpack(value).detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.dtype == object:
        label = f" for {name!r}" if name is not None else ""
        raise TypeError(
            f"Could not convert value{label} of type {type(value)!r} "
            "to a pickle-safe numpy value."
        )
    return np.array(array, copy=True)
