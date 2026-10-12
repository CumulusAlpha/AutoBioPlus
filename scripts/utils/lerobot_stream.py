"""Shared contracts for streaming rendered RGB into final LeRobot datasets."""

from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
import zarr

from src.policy_action import (
    action_transform_name,
    simulation_palm_action_to_real_tcp_action,
    simulation_world_vector_to_real_base_vector,
    simulation_wrist_action_to_real_tcp_action,
)
from src.task_goals import (
    goal_variant_key,
    task_goal_names,
    task_goal_variants,
    validate_balanced_goal_metadata,
)

STREAM_HEADER = struct.Struct("<qI")
STREAM_END = (-1, 0)
_XHAND_TUBE_INSERT_TASK_ID = "XHand-Tube-Insert"
_OLD_XHAND_WRIST_ACTION_TRANSFORM = "right_palm_site_to_wrist_3_link"


@dataclass(frozen=True)
class CollectionData:
    path: Path
    root: Any
    task_id: str
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    bounds: tuple[tuple[int, int], ...]
    prompts: tuple[str, ...]
    fps: int

    @property
    def num_frames(self) -> int:
        return int(self.state.shape[0])


def feature_matrix(value: Any, name: str, num_frames: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[0] != num_frames:
        raise ValueError(
            f"data/{name} must have shape ({num_frames}, D), got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"data/{name} contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float32)


def resolve_fps(attrs: dict[str, Any], override: int | None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError(f"FPS must be positive, got {override}.")
        return int(override)
    if "step_dt" not in attrs:
        raise ValueError("Collection has no step_dt attribute; pass --fps explicitly.")
    step_dt = float(attrs["step_dt"])
    exact = 1.0 / step_dt
    fps = round(exact)
    if (
        not np.isfinite(exact)
        or fps < 1
        or not np.isclose(exact, fps, atol=1e-6, rtol=0.0)
    ):
        raise ValueError(
            f"step_dt={step_dt!r} implies unsupported non-integer FPS {exact!r}."
        )
    return fps


def load_collection(
    path: Path,
    *,
    task_id: str,
    state_dim: int,
    action_dim: int,
    position_slices: tuple[tuple[int, int], ...],
    quaternion_slices: tuple[tuple[int, int], ...],
    max_frames: int | None = None,
    fps: int | None = None,
) -> CollectionData:
    """Load and normalize the non-image portion of a collected replay."""
    path = path.expanduser().resolve()
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    if attrs.get("created_by") != "scripts/collect.py":
        raise ValueError(
            "Input must be produced by scripts/collect.py; "
            f"created_by={attrs.get('created_by')!r}."
        )
    if str(attrs.get("task_id")) != task_id:
        raise ValueError(
            f"Collection task_id={attrs.get('task_id')!r}, expected {task_id!r}."
        )
    action_transform = attrs.get("action_transform")
    legacy_xhand_palm_action = (
        task_id == _XHAND_TUBE_INSERT_TASK_ID and action_transform == "none"
    )
    legacy_xhand_wrist_action = (
        task_id == _XHAND_TUBE_INSERT_TASK_ID
        and action_transform == _OLD_XHAND_WRIST_ACTION_TRANSFORM
    )
    if attrs.get("action_source_key") != "action" or (
        action_transform != action_transform_name(task_id)
        and not legacy_xhand_palm_action
        and not legacy_xhand_wrist_action
    ):
        raise ValueError(
            "Collection action transform does not match the task action schema."
        )
    if legacy_xhand_palm_action or legacy_xhand_wrist_action:
        warnings.warn(
            "Converting legacy XHand-Tube-Insert actions to real-base UR TCP actions.",
            stacklevel=2,
        )
    if attrs.get("state_source_key") not in {"current_qpos", "qpos"}:
        raise ValueError(
            "Collection must use qpos state; "
            f"state_source_key={attrs.get('state_source_key')!r}."
        )
    if attrs.get("action_env_origin_normalization") != "none":
        raise ValueError(
            "Collected actions must retain their episode environment origins."
        )

    for attr_name, expected in (
        ("action_position_slices", position_slices),
        ("action_quaternion_slices", quaternion_slices),
    ):
        actual = tuple(
            tuple(int(index) for index in pair) for pair in attrs.get(attr_name, ())
        )
        if actual != expected:
            raise ValueError(f"{task_id} {attr_name} is {actual}, expected {expected}.")

    if "data" not in root or "meta" not in root:
        raise KeyError("Collection must contain data and meta groups.")
    data = root["data"]
    meta = root["meta"]
    for key in ("state", "action", "timestamp"):
        if key not in data:
            raise KeyError(f"Collection has no data/{key} array.")
    for key in ("episode_ends", "episode_env_origins", "episode_prompts"):
        if key not in meta:
            raise KeyError(f"Collection has no meta/{key} array.")

    source_frames = int(data["state"].shape[0])
    lengths = {key: int(data[key].shape[0]) for key in ("state", "action", "timestamp")}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Collection arrays have mismatched frame counts: {lengths}.")
    selected_frames = source_frames
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError(f"max_frames must be positive, got {max_frames}.")
        selected_frames = min(source_frames, int(max_frames))

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    if (
        episode_ends.ndim != 1
        or episode_ends.size == 0
        or np.any(np.diff(episode_ends) <= 0)
        or int(episode_ends[-1]) != source_frames
    ):
        raise ValueError(
            "meta/episode_ends is inconsistent with the collection frame count."
        )
    output_ends = episode_ends[episode_ends <= selected_frames]
    if output_ends.size == 0 or int(output_ends[-1]) != selected_frames:
        output_ends = np.concatenate(
            (output_ends, np.asarray([selected_frames], dtype=np.int64))
        )
    starts = np.concatenate(([0], output_ends[:-1])).astype(np.int64)
    bounds = tuple(
        (int(start), int(end)) for start, end in zip(starts, output_ends, strict=True)
    )

    origins = np.asarray(meta["episode_env_origins"][:], dtype=np.float32)
    if origins.shape != (episode_ends.size, 3) or not np.isfinite(origins).all():
        raise ValueError(
            "meta/episode_env_origins must have shape "
            f"({episode_ends.size}, 3), got {origins.shape}."
        )
    frame_ids = np.arange(selected_frames, dtype=np.int64)
    episode_ids = np.searchsorted(episode_ends, frame_ids, side="right")
    frame_origins = origins[episode_ids]

    state = feature_matrix(data["state"][:selected_frames], "state", selected_frames)
    action = feature_matrix(data["action"][:selected_frames], "action", selected_frames)
    if state.shape[1] != state_dim or action.shape[1] != action_dim:
        raise ValueError(
            f"{task_id} state/action dimensions are {(state.shape[1], action.shape[1])}, "
            f"expected {(state_dim, action_dim)}."
        )
    if legacy_xhand_palm_action:
        action = simulation_palm_action_to_real_tcp_action(action)
    elif legacy_xhand_wrist_action:
        action = simulation_wrist_action_to_real_tcp_action(action)
    action = action.copy()
    legacy_sim_world = (
        task_id == _XHAND_TUBE_INSERT_TASK_ID
        and attrs.get("scene_coordinate_frame") != "real_ur_base"
    )
    normalized_origins = (
        simulation_world_vector_to_real_base_vector(frame_origins)
        if legacy_sim_world
        else frame_origins
    )
    for start, stop in position_slices:
        action[:, start:stop] -= normalized_origins

    timestamp = np.asarray(
        data["timestamp"][:selected_frames], dtype=np.float32
    ).reshape(-1)
    if timestamp.shape != (selected_frames,) or not np.isfinite(timestamp).all():
        raise ValueError(
            f"data/timestamp must be a finite ({selected_frames},) vector."
        )

    prompt_values = np.asarray(meta["episode_prompts"][:])
    if prompt_values.shape != episode_ends.shape:
        raise ValueError("meta/episode_prompts must contain one prompt per episode.")
    prompts = tuple(str(value).strip() for value in prompt_values[: len(bounds)])
    if any(not prompt for prompt in prompts):
        raise ValueError("meta/episode_prompts contains an empty prompt.")
    goal_names = task_goal_names(task_id)
    if goal_names is not None:
        for key in ("goal_names", "episode_goals"):
            if key not in meta:
                raise KeyError(f"Variable-goal collection has no meta/{key} array.")
        actual_names = tuple(str(value) for value in meta["goal_names"][:])
        if actual_names != goal_names:
            raise ValueError(f"Goal names are {actual_names}, expected {goal_names}.")
        episode_goals = np.asarray(meta["episode_goals"][:])
        expected_shape = (episode_ends.size, len(goal_names))
        if episode_goals.shape != expected_shape:
            raise ValueError(
                f"meta/episode_goals has shape {episode_goals.shape}, "
                f"expected {expected_shape}."
            )
        goal_balanced = attrs.get("goal_balanced")
        if goal_balanced is True:
            validate_balanced_goal_metadata(task_id, episode_goals, attrs)
        elif goal_balanced is False:
            catalog = task_goal_variants(task_id)
            assert catalog is not None
            allowed = {goal_variant_key(values) for values in catalog}
            unknown = [
                goal_variant_key(values)
                for values in episode_goals
                if goal_variant_key(values) not in allowed
            ]
            if unknown:
                raise ValueError(
                    f"Collection contains unknown goal variants: {unknown[:5]}."
                )
        else:
            raise ValueError(
                f"Variable-goal collection for {task_id!r} has no goal_balanced marker."
            )

    return CollectionData(
        path=path,
        root=root,
        task_id=task_id,
        state=state,
        action=action,
        timestamp=np.ascontiguousarray(timestamp),
        bounds=bounds,
        prompts=prompts,
        fps=resolve_fps(attrs, fps),
    )


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"RGB stream ended with {remaining} byte(s) still expected.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_rgb_batches(
    stream: BinaryIO,
    *,
    num_cameras: int,
    height: int,
    width: int,
) -> Iterator[tuple[int, np.ndarray]]:
    expected_start = 0
    frame_bytes = num_cameras * height * width * 3
    while True:
        start, length = STREAM_HEADER.unpack(read_exact(stream, STREAM_HEADER.size))
        if (start, length) == STREAM_END:
            break
        if start != expected_start:
            raise ValueError(
                f"RGB stream starts at frame {start}, expected {expected_start}."
            )
        if length < 1:
            raise ValueError(f"RGB stream batch length must be positive, got {length}.")
        payload = read_exact(stream, int(length) * frame_bytes)
        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(
            int(length), num_cameras, height, width, 3
        )
        yield int(start), rgb
        expected_start += int(length)


def write_rgb_batch(stream: BinaryIO, start: int, rgb: np.ndarray) -> None:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    stream.write(STREAM_HEADER.pack(int(start), int(rgb.shape[0])))
    stream.write(memoryview(rgb).cast("B"))


def finish_rgb_stream(stream: BinaryIO) -> None:
    stream.write(STREAM_HEADER.pack(*STREAM_END))
    stream.flush()
