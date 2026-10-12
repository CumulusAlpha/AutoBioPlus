"""Helpers for efficient open-loop policy evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TimestampedActionBuffer:
    """A contiguous action sequence indexed by simulation control step."""

    timestamp: int
    actions: torch.Tensor

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("Action-buffer timestamp must be non-negative.")
        if self.actions.ndim != 2:
            raise ValueError("Action buffer must have shape [horizon, action_dim].")
        if self.actions.shape[0] == 0:
            raise ValueError("Action buffer cannot be empty.")

    @property
    def end_timestamp(self) -> int:
        return self.timestamp + int(self.actions.shape[0])

    def discard_before(self, timestamp: int) -> TimestampedActionBuffer | None:
        """Drop actions older than ``timestamp``."""
        if timestamp <= self.timestamp:
            return self
        offset = timestamp - self.timestamp
        if offset >= self.actions.shape[0]:
            return None
        return TimestampedActionBuffer(timestamp, self.actions[offset:])

    def pop(self, timestamp: int) -> tuple[torch.Tensor, TimestampedActionBuffer | None]:
        """Pop the action for exactly ``timestamp``."""
        current = self.discard_before(timestamp)
        if current is None or current.timestamp != timestamp:
            raise RuntimeError(f"No buffered action exists for timestamp {timestamp}.")
        remaining = (
            TimestampedActionBuffer(timestamp + 1, current.actions[1:])
            if current.actions.shape[0] > 1
            else None
        )
        return current.actions[0], remaining


def slerp_quaternion_wxyz(
    old_quaternion: torch.Tensor,
    new_quaternion: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Interpolate quaternion batches along the shortest path on SO(3)."""
    old_quaternion = F.normalize(old_quaternion, dim=-1)
    new_quaternion = F.normalize(new_quaternion, dim=-1)

    dot = (old_quaternion * new_quaternion).sum(dim=-1, keepdim=True)
    new_quaternion = torch.where(dot < 0, -new_quaternion, new_quaternion)
    dot = dot.abs().clamp(max=1.0)

    linear = F.normalize(
        torch.lerp(old_quaternion, new_quaternion, weight),
        dim=-1,
    )
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    spherical = (
        torch.sin((1.0 - weight) * theta) / sin_theta.clamp_min(1e-8)
        * old_quaternion
        + torch.sin(weight * theta) / sin_theta.clamp_min(1e-8)
        * new_quaternion
    )
    spherical = F.normalize(spherical, dim=-1)
    return torch.where(dot > 0.9995, linear, spherical)


def merge_timestamped_action_chunk(
    old_buffer: TimestampedActionBuffer | None,
    new_chunk: torch.Tensor,
    *,
    new_timestamp: int,
    now_timestamp: int,
    quaternion_slices: tuple[tuple[int, int], ...] = (),
) -> TimestampedActionBuffer | None:
    """Merge a delayed action chunk into a time-aligned execution buffer.

    Actions that became stale while inference was running are discarded. In
    the overlap, interpolation stays away from both endpoints using
    ``(index + 1) / (overlap + 1)``, matching DexJoCo's asynchronous
    action-buffer behavior.
    """
    if new_chunk.ndim != 2:
        raise ValueError("New action chunk must have shape [horizon, action_dim].")
    if new_timestamp < 0 or now_timestamp < 0:
        raise ValueError("Action timestamps must be non-negative.")
    if new_chunk.shape[0] == 0:
        return (
            old_buffer.discard_before(now_timestamp)
            if old_buffer is not None
            else None
        )

    action_dim = int(new_chunk.shape[1])
    for start, stop in quaternion_slices:
        if stop - start != 4 or start < 0 or stop > action_dim:
            raise ValueError(
                f"Invalid quaternion slice ({start}, {stop}) for action dim "
                f"{action_dim}."
            )

    stale_prefix = max(0, now_timestamp - new_timestamp)
    if stale_prefix >= new_chunk.shape[0]:
        return (
            old_buffer.discard_before(now_timestamp)
            if old_buffer is not None
            else None
        )
    new_buffer = TimestampedActionBuffer(
        new_timestamp + stale_prefix,
        new_chunk[stale_prefix:],
    )
    old_buffer = (
        old_buffer.discard_before(now_timestamp)
        if old_buffer is not None
        else None
    )
    if old_buffer is None:
        return new_buffer
    if old_buffer.actions.shape[1] != action_dim:
        raise ValueError(
            f"Action dimensions differ: {old_buffer.actions.shape[1]} and "
            f"{action_dim}."
        )
    if old_buffer.actions.device != new_chunk.device:
        raise ValueError("Timestamped action chunks are on different devices.")
    if old_buffer.actions.dtype != new_chunk.dtype:
        raise ValueError("Timestamped action chunks have different dtypes.")

    start_timestamp = min(old_buffer.timestamp, new_buffer.timestamp)
    end_timestamp = max(old_buffer.end_timestamp, new_buffer.end_timestamp)
    if (
        old_buffer.end_timestamp < new_buffer.timestamp
        or new_buffer.end_timestamp < old_buffer.timestamp
    ):
        raise RuntimeError(
            "Timestamped action chunks contain an execution gap: "
            f"old=[{old_buffer.timestamp}, {old_buffer.end_timestamp}), "
            f"new=[{new_buffer.timestamp}, {new_buffer.end_timestamp})."
        )

    merged = torch.empty(
        (end_timestamp - start_timestamp, action_dim),
        device=new_chunk.device,
        dtype=new_chunk.dtype,
    )
    old_start = old_buffer.timestamp - start_timestamp
    old_stop = old_buffer.end_timestamp - start_timestamp
    new_start = new_buffer.timestamp - start_timestamp
    new_stop = new_buffer.end_timestamp - start_timestamp
    merged[old_start:old_stop] = old_buffer.actions

    overlap_start_ts = max(old_buffer.timestamp, new_buffer.timestamp)
    overlap_end_ts = min(old_buffer.end_timestamp, new_buffer.end_timestamp)
    overlap = overlap_end_ts - overlap_start_ts
    if new_buffer.timestamp < old_buffer.timestamp:
        prefix = old_buffer.timestamp - new_buffer.timestamp
        merged[new_start:old_start] = new_buffer.actions[:prefix]
    if new_buffer.end_timestamp > old_buffer.end_timestamp:
        suffix = new_buffer.end_timestamp - old_buffer.end_timestamp
        merged[old_stop:new_stop] = new_buffer.actions[-suffix:]

    if overlap > 0:
        old_overlap_start = overlap_start_ts - old_buffer.timestamp
        new_overlap_start = overlap_start_ts - new_buffer.timestamp
        merged_overlap_start = overlap_start_ts - start_timestamp
        old_overlap = old_buffer.actions[
            old_overlap_start : old_overlap_start + overlap
        ]
        new_overlap = new_buffer.actions[
            new_overlap_start : new_overlap_start + overlap
        ]
        weights = torch.linspace(
            1.0 / (overlap + 1),
            overlap / (overlap + 1),
            overlap,
            device=new_chunk.device,
            dtype=new_chunk.dtype,
        ).unsqueeze(-1)
        blended = torch.lerp(old_overlap, new_overlap, weights)
        for start, stop in quaternion_slices:
            blended[:, start:stop] = slerp_quaternion_wxyz(
                old_overlap[:, start:stop],
                new_overlap[:, start:stop],
                weights,
            )
        merged[
            merged_overlap_start : merged_overlap_start + overlap
        ] = blended

    return TimestampedActionBuffer(start_timestamp, merged)


def blend_action_chunks(
    old_chunk: torch.Tensor,
    new_chunk: torch.Tensor,
    quaternion_slices: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Blend a newly inferred chunk into the still-valid buffered tail."""
    if old_chunk.ndim != 2 or new_chunk.ndim != 2:
        raise ValueError("Action chunks must both have shape [horizon, action_dim].")
    if old_chunk.shape[1] != new_chunk.shape[1]:
        raise ValueError(
            f"Action chunk dimensions differ: {old_chunk.shape[1]} and "
            f"{new_chunk.shape[1]}."
        )
    if old_chunk.device != new_chunk.device:
        raise ValueError(
            f"Action chunks are on different devices: {old_chunk.device} and "
            f"{new_chunk.device}."
        )
    if old_chunk.dtype != new_chunk.dtype:
        raise ValueError(
            f"Action chunks have different dtypes: {old_chunk.dtype} and "
            f"{new_chunk.dtype}."
        )

    action_dim = int(new_chunk.shape[1])
    for start, stop in quaternion_slices:
        if stop - start != 4 or start < 0 or stop > action_dim:
            raise ValueError(
                f"Invalid quaternion slice ({start}, {stop}) for action dim "
                f"{action_dim}."
            )

    overlap = min(int(old_chunk.shape[0]), int(new_chunk.shape[0]))
    if overlap == 0:
        return new_chunk if new_chunk.shape[0] > 0 else old_chunk

    weights = torch.linspace(
        1.0 / (overlap + 1),
        overlap / (overlap + 1),
        overlap,
        device=new_chunk.device,
        dtype=new_chunk.dtype,
    ).unsqueeze(-1)
    blended = torch.lerp(old_chunk[:overlap], new_chunk[:overlap], weights)

    for start, stop in quaternion_slices:
        blended[:, start:stop] = slerp_quaternion_wxyz(
            old_chunk[:overlap, start:stop],
            new_chunk[:overlap, start:stop],
            weights,
        )

    tail = (
        new_chunk[overlap:]
        if new_chunk.shape[0] >= old_chunk.shape[0]
        else old_chunk[overlap:]
    )
    return torch.cat((blended, tail), dim=0)


def prepare_action_buffer(
    new_chunk: torch.Tensor,
    *,
    execution_horizon: int,
    blend: bool,
    old_chunk: torch.Tensor | None = None,
    quaternion_slices: tuple[tuple[int, int], ...] = (),
) -> torch.Tensor:
    """Build the action buffer for synchronous chunked evaluation."""
    if new_chunk.ndim != 2:
        raise ValueError("Action chunks must have shape [horizon, action_dim].")
    if not 1 <= execution_horizon <= new_chunk.shape[0]:
        raise ValueError(
            f"execution_horizon must be in [1, {new_chunk.shape[0]}], "
            f"got {execution_horizon}."
        )
    if blend:
        if old_chunk is not None and old_chunk.shape[0] > 0:
            return blend_action_chunks(old_chunk, new_chunk, quaternion_slices)
        return new_chunk

    # Canonical synchronous evaluation executes only this prefix. Keeping no
    # stale tail makes its disposal explicit and prevents accidental reuse.
    return new_chunk[:execution_horizon].clone()


class ChunkedSensingThrottle:
    """Skip camera sensing while a policy consumes a cached action chunk."""

    def __init__(self, env: Any, needs_replan: Callable[[], bool]) -> None:
        self._raw_env = env.unwrapped
        self._sim = self._raw_env.sim
        self._needs_replan = needs_replan
        self._original_sense = self._sim.sense
        self._installed = False

    @classmethod
    def install_if_supported(
        cls,
        env: Any,
        needs_replan: Callable[[], bool],
    ) -> ChunkedSensingThrottle | None:
        context = env.unwrapped.scene.sensor_context
        if context is None or not context.has_cameras or context.has_raycasts:
            return None
        throttle = cls(env, needs_replan)
        throttle.install()
        return throttle

    def install(self) -> None:
        if self._installed:
            return
        self._sim.sense = self._sense_if_needed
        self._installed = True

    def close(self) -> None:
        if not self._installed:
            return
        self._sim.sense = self._original_sense
        self._installed = False

    def _sense_if_needed(self) -> None:
        reset_buf = getattr(self._raw_env, "reset_buf", None)
        has_reset = bool(
            reset_buf is not None
            and torch.as_tensor(reset_buf, device=self._raw_env.device).any().item()
        )
        if self._needs_replan() or has_reset:
            self._original_sense()
