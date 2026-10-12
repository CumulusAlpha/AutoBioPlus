"""Streaming Zarr writing and the collection data contract."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numcodecs import Blosc

from scripts.utils.collection_config import CollectConfig
from scripts.utils.collection_goals import BalancedGoalScheduler
from src.tasks.policy_action import (
    action_position_slices,
    action_quaternion_slices,
    action_transform_name,
    environment_action_to_policy_action,
    validate_action_layout_attrs,
)
from src.tasks.policy_state import DUAL_ARM_STATE_ENTITIES
from src.tasks.task_goals import format_task_prompt, task_goal_names

_XHAND_BUNSEN_TASK_ID = "XHand-Bunsen-Manipulate"

_XHAND_TUBE_INSERT_TASK_ID = "XHand-Tube-Insert"

_WUJI_HEATING_PLATE_TASK_ID = "Wuji-HeatingPlate-Rotate"


def stack_frames(values: list[Any]) -> np.ndarray:
    return np.stack(values, axis=0)


def select_state_key(arrays: dict[str, np.ndarray], requested: str | None) -> str:
    if requested is not None:
        if requested not in arrays:
            raise KeyError(f"Requested state key {requested!r} not found.")
        return requested
    for candidate in ("current_qpos", "qpos"):
        if candidate in arrays:
            return candidate
    raise KeyError("No state key found. Pass --state-key explicitly.")


def find_first_non_static_frame(action: np.ndarray) -> int:
    if len(action) < 2:
        return 0
    for idx in range(len(action) - 1):
        if not np.array_equal(action[idx], action[idx + 1]):
            return idx
    return 0


def normalize_time_major(
    array: np.ndarray,
    *,
    expected_length: int,
    key: str,
    flatten_feature: bool,
    float_dtype: str,
) -> np.ndarray:
    if array.dtype == object:
        raise TypeError(f"Key {key!r} has object dtype and cannot be stored in zarr.")
    if array.ndim == 0:
        raise ValueError(f"Key {key!r} is scalar; expected a time-major array.")
    if array.shape[0] != expected_length:
        raise ValueError(
            f"Key {key!r} has length {array.shape[0]}, expected {expected_length}."
        )

    if flatten_feature:
        if array.ndim == 1:
            array = array[:, None]
        elif array.ndim > 2:
            array = array.reshape(array.shape[0], -1)

    if np.issubdtype(array.dtype, np.floating) and float_dtype != "preserve":
        array = array.astype(float_dtype, copy=False)

    return np.ascontiguousarray(array)


def chunks_for_shape(shape: tuple[int, ...], chunk_length: int) -> tuple[int, ...]:
    time_chunk = min(max(1, chunk_length), max(1, shape[0]))
    return (time_chunk, *shape[1:])


_ZARR_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def create_empty_dataset(
    group,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunk_length: int,
):
    chunks = chunks_for_shape(shape, chunk_length)

    try:
        return group.create_array(
            name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compressors=(_ZARR_COMPRESSOR,),
            overwrite=True,
        )
    except (AttributeError, TypeError):
        pass

    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=_ZARR_COMPRESSOR,
        overwrite=True,
    )


def episode_source_arrays(
    steps: Iterable[dict[str, Any]], *, step_dt: float
) -> dict[str, np.ndarray]:
    arrays: dict[str, list[Any]] = defaultdict(list)

    for t, step in enumerate(steps):
        arrays["action"].append(step["action"])
        arrays["done"].append(step["done"])
        arrays["success"].append(step["success"])
        arrays["time_out"].append(step["time_out"])
        arrays["timestamp"].append(t * step_dt)

        for key, value in step["state"].items():
            arrays[key].append(value)

    return {key: stack_frames(values) for key, values in arrays.items()}


def episode_env_origin(
    source: dict[str, np.ndarray], *, start_idx: int
) -> np.ndarray | None:
    if "env_origin" not in source:
        return None
    origin = np.asarray(source["env_origin"], dtype=np.float64)
    if origin.ndim >= 2:
        origin = origin[min(start_idx, origin.shape[0] - 1)]
    origin = origin.reshape(-1)
    if origin.shape[0] < 3:
        return None
    return origin[:3]


def data_arrays_from_source(
    source: dict[str, np.ndarray],
    *,
    task_id: str,
    start_idx: int,
    total_length: int,
    cfg: CollectConfig,
) -> tuple[dict[str, np.ndarray], str]:
    selected_state_key = select_state_key(source, cfg.state_key)
    expected_length = total_length - start_idx

    action = normalize_time_major(
        np.asarray(source["action"])[start_idx:],
        expected_length=expected_length,
        key="action",
        flatten_feature=True,
        float_dtype=cfg.float_dtype,
    )
    action = environment_action_to_policy_action(action, task_id)

    state = normalize_time_major(
        np.asarray(source[selected_state_key])[start_idx:],
        expected_length=expected_length,
        key=selected_state_key,
        flatten_feature=True,
        float_dtype=cfg.float_dtype,
    )

    arrays = {
        "action": action,
        "state": state,
    }
    if "timestamp" in source:
        arrays["timestamp"] = normalize_time_major(
            np.asarray(source["timestamp"])[start_idx:],
            expected_length=expected_length,
            key="timestamp",
            flatten_feature=False,
            float_dtype=cfg.float_dtype,
        )
    return arrays, selected_state_key


def replay_arrays_from_source(
    source: dict[str, np.ndarray],
    *,
    start_idx: int,
    total_length: int,
) -> dict[str, np.ndarray]:
    expected_length = total_length - start_idx
    arrays: dict[str, np.ndarray] = {}
    for key, value in sorted(source.items()):
        arrays[key] = normalize_time_major(
            np.asarray(value)[start_idx:],
            expected_length=expected_length,
            key=key,
            flatten_feature=False,
            float_dtype="preserve",
        )
    return arrays


class StreamingZarrWriter:
    """Append successful episodes into one task-level zarr replay buffer."""

    def __init__(
        self, task_id: str, path: Path, cfg: CollectConfig, *, step_dt: float
    ) -> None:
        try:
            import zarr
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: zarr. Install it first, for example `uv pip install zarr`."
            ) from exc

        self.task_id = task_id
        self.path = path
        self.cfg = cfg
        self.step_dt = step_dt

        if path.exists() and cfg.overwrite:
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.root = zarr.open_group(str(path), mode="a", zarr_format=2)
        self.data_group = self.root.require_group("data")
        self.replay_group = self.root.require_group("replay")
        self.meta_group = self.root.require_group("meta")

        self.num_steps = self._existing_num_steps()
        self.num_episodes = self._existing_num_episodes()
        self.trimmed_leading_frames = list(
            self.root.attrs.get("trimmed_leading_frames", [])
        )
        self.source_env_ids = list(self.root.attrs.get("source_env_ids", []))
        self.success_flags = list(self.root.attrs.get("success_flags", []))
        self.state_source_key = str(self.root.attrs.get("state_source_key", ""))
        self.goal_names = task_goal_names(task_id)
        self.goal_balance_attrs: dict[str, Any] = {}
        self._validate_existing_contract()
        self._ensure_goal_names()

    def _validate_existing_contract(self) -> None:
        if self.num_episodes == 0:
            return
        attrs = self.root.attrs
        if attrs.get("created_by") != "scripts/collect.py":
            raise ValueError(
                f"Cannot append to {self.path}: created_by={attrs.get('created_by')!r}."
            )
        if attrs.get("task_id") != self.task_id:
            raise ValueError(
                f"Cannot append task {self.task_id!r} to dataset for "
                f"{attrs.get('task_id')!r}."
            )
        existing_step_dt = float(attrs.get("step_dt", float("nan")))
        if not np.isclose(existing_step_dt, self.step_dt, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Cannot append with step_dt={self.step_dt}; existing dataset uses "
                f"step_dt={existing_step_dt}."
            )
        validate_action_layout_attrs(self.root.attrs, self.task_id)
        if "extra" in self.data_group or "extra" in self.replay_group:
            raise ValueError(
                f"Cannot append to legacy goal layout in {self.path}. "
                "Recollect with --overwrite to use episode-level meta goals."
            )
        if "episode_prompts" not in self.meta_group:
            raise ValueError(
                f"Cannot append to {self.path}: meta/episode_prompts is missing. "
                "Recollect with --overwrite."
            )
        if int(self.meta_group["episode_prompts"].shape[0]) != self.num_episodes:
            raise ValueError(
                "meta/episode_prompts must contain one prompt per episode."
            )
        if self.goal_names is None:
            unexpected = {"goal_names", "episode_goals"} & set(
                self.meta_group.array_keys()
            )
            if unexpected:
                raise ValueError(
                    f"Fixed-goal task {self.task_id!r} has unexpected goal arrays: {sorted(unexpected)}."
                )
            return
        for key in ("goal_names", "episode_goals"):
            if key not in self.meta_group:
                raise ValueError(
                    f"Cannot append to {self.path}: meta/{key} is missing."
                )
        existing_names = tuple(str(value) for value in self.meta_group["goal_names"][:])
        if existing_names != self.goal_names:
            raise ValueError(
                f"Goal fields changed for {self.task_id!r}: {existing_names} != {self.goal_names}."
            )
        if int(self.meta_group["episode_goals"].shape[0]) != self.num_episodes:
            raise ValueError("meta/episode_goals must contain one goal per episode.")

    def _ensure_goal_names(self) -> None:
        if self.goal_names is None or "goal_names" in self.meta_group:
            return
        values = np.asarray(self.goal_names, dtype="<U64")
        dataset = create_empty_dataset(
            self.meta_group,
            "goal_names",
            shape=values.shape,
            dtype=values.dtype,
            chunk_length=self.cfg.chunk_length,
        )
        dataset[:] = values

    def _existing_num_steps(self) -> int:
        if (
            "episode_ends" in self.meta_group
            and self.meta_group["episode_ends"].shape[0]
        ):
            return int(self.meta_group["episode_ends"][-1])
        return int(self.root.attrs.get("num_steps", 0))

    def _existing_num_episodes(self) -> int:
        if "episode_ends" in self.meta_group:
            return int(self.meta_group["episode_ends"].shape[0])
        return int(self.root.attrs.get("num_episodes", 0))

    def existing_episode_goals(self) -> np.ndarray:
        if self.goal_names is None or "episode_goals" not in self.meta_group:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(self.meta_group["episode_goals"][:])

    def configure_goal_balance(self, scheduler: BalancedGoalScheduler | None) -> None:
        if self.goal_names is None:
            if scheduler is not None:
                raise ValueError(
                    f"Fixed-goal task {self.task_id!r} cannot use goal balancing."
                )
            return

        if scheduler is None:
            if self.num_episodes and bool(self.root.attrs.get("goal_balanced", False)):
                raise ValueError(
                    "Cannot append unbalanced episodes to a balanced collection. "
                    "Use balanced collection or --overwrite."
                )
            self.goal_balance_attrs = {"goal_balanced": False}
        else:
            expected = scheduler.metadata()
            if self.num_episodes:
                for key, value in expected.items():
                    existing = self.root.attrs.get(key)
                    if existing != value:
                        raise ValueError(
                            f"Cannot resume {self.path}: {key}={existing!r}, expected {value!r}. "
                            "Recollect with --overwrite."
                        )
            self.goal_balance_attrs = expected

        for key, value in self.goal_balance_attrs.items():
            self.root.attrs[key] = value

    def _ensure_dataset(self, group: Any, key: str, array: np.ndarray):
        if array.dtype == object:
            raise TypeError(
                f"Key {key!r} has object dtype and cannot be stored in zarr."
            )
        if key in group:
            dataset = group[key]
            if tuple(dataset.shape[1:]) != tuple(array.shape[1:]):
                raise ValueError(
                    f"Shape mismatch for {key!r}: existing feature shape "
                    f"{tuple(dataset.shape[1:])} != {tuple(array.shape[1:])}"
                )
            if np.dtype(dataset.dtype) != np.dtype(array.dtype):
                raise ValueError(
                    f"Dtype mismatch for {key!r}: existing {dataset.dtype} != {array.dtype}"
                )
            return dataset

        return create_empty_dataset(
            group,
            key,
            shape=(0, *array.shape[1:]),
            dtype=array.dtype,
            chunk_length=self.cfg.chunk_length,
        )

    def _validate_group_keys(
        self, group: Any, expected_keys: set[str], group_name: str
    ) -> None:
        if self.num_steps == 0:
            return
        existing_keys = set(group.array_keys())
        if existing_keys != expected_keys:
            missing = sorted(existing_keys - expected_keys)
            extra = sorted(expected_keys - existing_keys)
            raise ValueError(
                f"Cannot append to {self.path}: {group_name} keys changed; "
                f"missing_in_new={missing}, new_extra={extra}"
            )

    def _append_arrays(
        self, group: Any, arrays: dict[str, np.ndarray], start: int, end: int
    ) -> None:
        for key, array in arrays.items():
            dataset = self._ensure_dataset(group, key, array)
            dataset.resize((end, *dataset.shape[1:]))
            dataset[start:end] = array

    def _append_meta_array(self, key: str, values: np.ndarray) -> None:
        dataset = self._ensure_dataset(self.meta_group, key, np.asarray(values))
        old_len = int(dataset.shape[0])
        new_len = old_len + int(values.shape[0])
        dataset.resize((new_len, *dataset.shape[1:]))
        dataset[old_len:new_len] = values

    def write_episode(
        self,
        steps: Iterable[dict[str, Any]],
        *,
        env_id: int,
        success: bool,
    ) -> tuple[int, int]:
        steps = list(steps)
        episode_goal = self._episode_goal(steps)
        prompt = format_task_prompt(
            self.task_id,
            None if episode_goal is None else episode_goal.tolist(),
        )
        source = episode_source_arrays(steps, step_dt=self.step_dt)
        raw_action = np.asarray(source["action"])
        if raw_action.ndim == 0:
            raise ValueError("Collected action is scalar; expected a time-major array.")

        total_length = int(raw_action.shape[0])
        start_idx = (
            find_first_non_static_frame(raw_action)
            if self.cfg.skip_static_frames
            else 0
        )
        if (
            self.cfg.skip_static_frames
            and start_idx < total_length
            and np.all(raw_action[start_idx] == 0)
        ):
            start_idx += 1
        if start_idx >= total_length:
            raise ValueError(
                "Episode has no frames after trimming leading static frames."
            )

        env_origin = episode_env_origin(source, start_idx=start_idx)
        data_arrays, selected_state_key = data_arrays_from_source(
            source,
            task_id=self.task_id,
            start_idx=start_idx,
            total_length=total_length,
            cfg=self.cfg,
        )
        replay_arrays = replay_arrays_from_source(
            source,
            start_idx=start_idx,
            total_length=total_length,
        )

        if self.state_source_key and self.state_source_key != selected_state_key:
            raise ValueError(
                f"Cannot append to {self.path}: state source changed from "
                f"{self.state_source_key!r} to {selected_state_key!r}."
            )
        self.state_source_key = selected_state_key

        self._validate_group_keys(self.data_group, set(data_arrays), "data")
        self._validate_group_keys(self.replay_group, set(replay_arrays), "replay")

        start = self.num_steps
        length = next(iter(data_arrays.values())).shape[0]
        end = start + length
        self._append_arrays(self.data_group, data_arrays, start, end)
        self._append_arrays(self.replay_group, replay_arrays, start, end)

        self._append_meta_array("episode_ends", np.asarray([end], dtype=np.int64))
        origin = np.zeros((1, 3), dtype=np.float32)
        if env_origin is not None:
            origin[0] = np.asarray(env_origin[:3], dtype=np.float32)
        self._append_meta_array("episode_env_origins", origin)
        self._append_meta_array("episode_prompts", np.asarray([prompt], dtype="<U512"))
        if episode_goal is not None:
            self._append_meta_array("episode_goals", episode_goal[None])

        self.num_steps = end
        self.num_episodes += 1
        self.trimmed_leading_frames.append(int(start_idx))
        self.source_env_ids.append(int(env_id))
        self.success_flags.append(bool(success))
        self._write_attrs(data_arrays, replay_arrays)
        return length, start_idx

    def _episode_goal(self, steps: list[dict[str, Any]]) -> np.ndarray | None:
        goals = [step.get("goal") for step in steps]
        if self.goal_names is None:
            if any(goal is not None for goal in goals):
                raise ValueError(
                    f"Fixed-goal task {self.task_id!r} unexpectedly produced goal values."
                )
            return None
        if not goals or any(goal is None for goal in goals):
            raise ValueError(
                f"Variable-goal task {self.task_id!r} has missing per-step goal values."
            )

        first = np.asarray(goals[0]).reshape(-1)
        if first.shape != (len(self.goal_names),):
            raise ValueError(
                f"Episode goal for {self.task_id!r} has shape {first.shape}, "
                f"expected {(len(self.goal_names),)}."
            )
        for index, goal in enumerate(goals[1:], start=1):
            value = np.asarray(goal).reshape(-1)
            equal = (
                np.allclose(value, first, rtol=0.0, atol=1e-6)
                if np.issubdtype(first.dtype, np.floating)
                else np.array_equal(value, first)
            )
            if not equal:
                raise ValueError(
                    f"Episode goal changed at step {index}: {value.tolist()} != {first.tolist()}."
                )
        if np.issubdtype(first.dtype, np.integer):
            return np.ascontiguousarray(first, dtype=np.int64)
        return np.ascontiguousarray(first, dtype=np.float32)

    def _write_attrs(
        self, data_arrays: dict[str, np.ndarray], replay_arrays: dict[str, np.ndarray]
    ) -> None:
        datasets = [
            *(self.data_group[key] for key in self.data_group.array_keys()),
            *(self.replay_group[key] for key in self.replay_group.array_keys()),
            *(self.meta_group[key] for key in self.meta_group.array_keys()),
        ]
        fully_compressed = all(any(dataset.compressors) for dataset in datasets)
        self.root.attrs["format"] = "dexbiolab_replay_buffer"
        self.root.attrs["created_by"] = "scripts/collect.py"
        self.root.attrs["uncompressed"] = not fully_compressed
        if fully_compressed:
            self.root.attrs["compression"] = "blosc-zstd"
        else:
            self.root.attrs.pop("compression", None)
        self.root.attrs["streaming_write"] = True
        self.root.attrs["task_id"] = self.task_id
        self.root.attrs["num_episodes"] = self.num_episodes
        self.root.attrs["num_steps"] = self.num_steps
        self.root.attrs["step_dt"] = float(self.step_dt)
        self.root.attrs["keys"] = sorted(data_arrays)
        self.root.attrs["replay_keys"] = sorted(replay_arrays)
        self.root.attrs["source_env_ids"] = self.source_env_ids
        self.root.attrs["success_flags"] = self.success_flags
        self.root.attrs["trimmed_leading_frames"] = self.trimmed_leading_frames
        self.root.attrs["state_source_key"] = self.state_source_key
        state_entity_names = DUAL_ARM_STATE_ENTITIES.get(self.task_id)
        if state_entity_names is not None:
            self.root.attrs["state_layout"] = "entity_joint_pos_concat"
            self.root.attrs["state_entity_names"] = list(state_entity_names)
        self.root.attrs["action_source_key"] = "action"
        self.root.attrs["action_transform"] = action_transform_name(self.task_id)
        self.root.attrs["action_position_slices"] = [
            list(value) for value in action_position_slices(self.task_id)
        ]
        self.root.attrs["action_quaternion_slices"] = [
            list(value) for value in action_quaternion_slices(self.task_id)
        ]
        self.root.attrs["action_env_origin_normalization"] = "none"
        if self.task_id == _XHAND_TUBE_INSERT_TASK_ID:
            self.root.attrs["action_layout"] = (
                "real_ur_tcp_target_xyz_wxyz_7+xhand_joint_target_12"
            )
            self.root.attrs["action_coordinate_frame"] = "real_ur_base"
            self.root.attrs["action_tcp_frame"] = "ur_controller_tcp"
            self.root.attrs["scene_coordinate_frame"] = "real_ur_base"
        self.root.attrs["has_replay_group"] = True
        self.root.attrs["replay_layout"] = "time_major_raw_collect_arrays"
        for key, value in self.goal_balance_attrs.items():
            self.root.attrs[key] = value
        body_names = getattr(self.cfg, "_body_names", None)
        if body_names:
            self.root.attrs["body_names"] = body_names
        if self.task_id == "XHand-MixerPanel-Manipulate":
            self.root.attrs["mixer_ui_value_names"] = ["time_s", "temp_c", "freq_rpm"]
        if self.task_id == _XHAND_BUNSEN_TASK_ID:
            self.root.attrs["bunsen_fire_state_names"] = [
                "enabled",
                "base_color_mix",
            ]
        if self.task_id == _WUJI_HEATING_PLATE_TASK_ID:
            self.root.attrs["heating_effect_state_names"] = [
                "heat_intensity",
                "stir_momentum",
                "stir_phase",
            ]

        manifest = {
            "output": str(self.path),
            "task_id": self.task_id,
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "keys": {
                f"data/{key}": {
                    "shape": list(self.data_group[key].shape),
                    "dtype": str(self.data_group[key].dtype),
                }
                for key in sorted(data_arrays)
            }
            | {
                f"replay/{key}": {
                    "shape": list(self.replay_group[key].shape),
                    "dtype": str(self.replay_group[key].dtype),
                }
                for key in sorted(replay_arrays)
            }
            | {
                f"meta/{key}": {
                    "shape": list(self.meta_group[key].shape),
                    "dtype": str(self.meta_group[key].dtype),
                }
                for key in sorted(self.meta_group.array_keys())
            },
        }
        self.root.attrs["collection_manifest"] = json.dumps(manifest, sort_keys=True)
