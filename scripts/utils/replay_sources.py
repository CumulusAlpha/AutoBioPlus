"""Episode loading from collection Zarr and final LeRobot datasets."""

from __future__ import annotations

import json
import shutil
import string
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeData:
    episode_idx: int
    start_frame: int
    qpos: np.ndarray
    action: np.ndarray | None = None
    images: dict[str, np.ndarray] | None = None
    prompt: str | None = None
    goal: dict[str, float | int] | None = None

    @property
    def num_steps(self) -> int:
        return int(self.qpos.shape[0])

    def get_qpos(self, step_idx: int) -> np.ndarray:
        return self.qpos[step_idx]

    def get_action(self, step_idx: int) -> np.ndarray | None:
        if self.action is None:
            return None
        return self.action[step_idx]

    def get_images(self, step_idx: int) -> dict[str, np.ndarray]:
        if self.images is None:
            return {}
        return {name: image[step_idx] for name, image in self.images.items()}


@dataclass
class ReplaySource:
    num_frames: int
    camera_names: list[str]
    action_transform: str
    episode_ends: np.ndarray | None = None
    episode_prompts: np.ndarray | None = None
    goal_names: tuple[str, ...] = ()
    episode_goals: np.ndarray | None = None

    @property
    def num_episodes(self) -> int:
        if self.episode_ends is None or self.episode_ends.size == 0:
            return 1
        return int(self.episode_ends.size)

    @property
    def episode_lengths(self) -> np.ndarray:
        if self.episode_ends is None or self.episode_ends.size == 0:
            return np.asarray([self.num_frames], dtype=np.int64)
        starts = np.concatenate(([0], self.episode_ends[:-1])).astype(np.int64)
        return (self.episode_ends - starts).astype(np.int64)

    @property
    def max_episode_steps(self) -> int:
        return int(np.max(self.episode_lengths))

    def episode_bounds(self, episode_idx: int) -> tuple[int, int]:
        if not 0 <= episode_idx < self.num_episodes:
            raise IndexError(
                f"Episode {episode_idx} is out of range [0, {self.num_episodes})."
            )
        if self.episode_ends is None or self.episode_ends.size == 0:
            return 0, self.num_frames
        start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1])
        end = int(self.episode_ends[episode_idx])
        return start, end

    def episode_prompt(self, episode_idx: int) -> str | None:
        if self.episode_prompts is None:
            return None
        return str(self.episode_prompts[episode_idx])

    def episode_goal(self, episode_idx: int) -> dict[str, float | int] | None:
        if self.episode_goals is None:
            return None
        values = np.asarray(self.episode_goals[episode_idx]).reshape(-1)
        return {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in zip(self.goal_names, values, strict=True)
        }

    def get_qpos(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def get_action(self, idx: int) -> np.ndarray | None:
        raise NotImplementedError

    def get_images(self, idx: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def load_episode(self, episode_idx: int) -> EpisodeData:
        raise NotImplementedError


@dataclass
class ZarrReplaySource(ReplaySource):
    data_group: Any = None
    qpos_array: Any = None
    qpos_source: str = "data/state"
    state_key: str = "state"
    action_key: str = "action"
    rgb_key: str = "rgb"
    rgb_frame_shape: tuple[int, ...] | None = None

    def get_qpos(self, idx: int) -> np.ndarray:
        return np.asarray(self.qpos_array[idx])

    def get_action(self, idx: int) -> np.ndarray | None:
        if self.action_key not in self.data_group:
            return None
        return np.asarray(self.data_group[self.action_key][idx])

    def get_images(self, idx: int) -> dict[str, np.ndarray]:
        if self.rgb_key not in self.data_group or self.rgb_frame_shape is None:
            return {}

        rgb = np.asarray(self.data_group[self.rgb_key][idx])
        if rgb.ndim == 1:
            rgb = rgb.reshape(self.rgb_frame_shape)

        if rgb.ndim == 3 and self.camera_names:
            return {self.camera_names[0]: rgb}
        if rgb.ndim == 4:
            return {
                name: rgb[cam_idx] for cam_idx, name in enumerate(self.camera_names)
            }

        raise ValueError(f"Unsupported rgb frame shape: {rgb.shape}")

    def load_episode(self, episode_idx: int) -> EpisodeData:
        start, end = self.episode_bounds(episode_idx)
        action = None
        if self.action_key in self.data_group:
            action = np.asarray(self.data_group[self.action_key][start:end])

        images: dict[str, np.ndarray] = {}
        if self.rgb_key in self.data_group and self.rgb_frame_shape is not None:
            rgb = np.asarray(self.data_group[self.rgb_key][start:end])
            if rgb.ndim == 2 and len(self.rgb_frame_shape) > 1:
                rgb = rgb.reshape((end - start, *self.rgb_frame_shape))

            if rgb.ndim == 4 and self.camera_names:
                images[self.camera_names[0]] = rgb
            elif rgb.ndim == 5:
                images = {
                    name: rgb[:, cam_idx]
                    for cam_idx, name in enumerate(self.camera_names)
                }
            else:
                raise ValueError(f"Unsupported episode rgb shape: {rgb.shape}")

        return EpisodeData(
            episode_idx=episode_idx,
            start_frame=start,
            qpos=np.asarray(self.qpos_array[start:end]),
            action=action,
            images=images,
            prompt=self.episode_prompt(episode_idx),
            goal=self.episode_goal(episode_idx),
        )


@dataclass
class LeRobotReplaySource(ReplaySource):
    dataset_path: Path | None = None
    info: dict[str, Any] | None = None
    state: np.ndarray | None = None
    replay_qpos: np.ndarray | None = None
    action: np.ndarray | None = None
    episode_rows: tuple[dict[str, Any], ...] = ()

    def get_qpos(self, idx: int) -> np.ndarray:
        assert self.state is not None
        source = self.replay_qpos if self.replay_qpos is not None else self.state
        return source[idx]

    def get_action(self, idx: int) -> np.ndarray | None:
        if self.action is None:
            return None
        return self.action[idx]

    def get_images(self, idx: int) -> dict[str, np.ndarray]:
        if self.episode_ends is None:
            episode_idx = 0
        else:
            episode_idx = int(np.searchsorted(self.episode_ends, idx, side="right"))
        episode = self.load_episode(episode_idx)
        return episode.get_images(idx - episode.start_frame)

    def _video_path(self, episode_idx: int, video_key: str) -> tuple[Path, int]:
        assert self.dataset_path is not None
        assert self.info is not None
        template = str(self.info.get("video_path", ""))
        if not template:
            raise KeyError("LeRobot meta/info.json has no video_path template.")
        row = self.episode_rows[episode_idx] if self.episode_rows else {}
        chunks_size = int(self.info.get("chunks_size", 1000))
        prefix = f"videos/{video_key}"
        context: dict[str, Any] = {
            "video_key": video_key,
            "episode_index": episode_idx,
            "episode_chunk": episode_idx // chunks_size,
        }
        for name in ("chunk_index", "file_index"):
            value = row_value(row, f"{prefix}/{name}")
            if value is None:
                value = row_value(row, name)
            if value is not None:
                context[name] = int(value)
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        }
        missing = fields - set(context)
        if missing:
            raise KeyError(
                f"Episode {episode_idx} metadata cannot fill video_path fields {sorted(missing)}."
            )
        path = self.dataset_path / template.format(**context)
        start_seconds = row_value(row, f"{prefix}/from_timestamp")
        if start_seconds is None:
            start_seconds = row_value(row, f"{prefix}/from")
        fps = int(self.info.get("fps", 0))
        start_frame = 0 if start_seconds is None else round(float(start_seconds) * fps)
        return path, start_frame

    def _load_video_episode(
        self,
        episode_idx: int,
        camera_name: str,
        length: int,
    ) -> np.ndarray:
        assert self.info is not None
        video_key = f"observation.images.{camera_name}"
        path, start_frame = self._video_path(episode_idx, video_key)
        if not path.is_file():
            raise FileNotFoundError(path)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required to decode LeRobot videos.")

        feature = self.info.get("features", {}).get(video_key, {})
        shape = tuple(int(value) for value in feature.get("shape", ()))
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"Invalid LeRobot video shape for {video_key}: {shape}.")
        height, width, _ = shape
        fps = float(self.info.get("fps", 0))
        if fps <= 0:
            raise ValueError(f"Invalid LeRobot video FPS: {fps}.")

        command = [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{start_frame / fps:.9f}",
            "-i",
            str(path),
            "-frames:v",
            str(length),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        expected_bytes = length * height * width * 3
        frames = np.empty((length, height, width, 3), dtype=np.uint8)
        raw_frames = memoryview(frames).cast("B")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdout is not None
        assert process.stderr is not None
        decoded_bytes = 0
        while decoded_bytes < expected_bytes:
            count = process.stdout.readinto(raw_frames[decoded_bytes:])
            if not count:
                break
            decoded_bytes += count
        process.stdout.close()
        error = process.stderr.read().decode(errors="replace").strip()
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"FFmpeg failed to decode {path}: {error}")
        if decoded_bytes != expected_bytes:
            decoded_frames = decoded_bytes // (height * width * 3)
            raise EOFError(
                f"Video {path} produced {decoded_frames}/{length} episode frames."
            )
        return frames

    def load_episode(self, episode_idx: int) -> EpisodeData:
        assert self.state is not None
        start, end = self.episode_bounds(episode_idx)
        qpos_source = self.replay_qpos if self.replay_qpos is not None else self.state
        images = {
            name: self._load_video_episode(episode_idx, name, end - start)
            for name in self.camera_names
        }
        action = None if self.action is None else self.action[start:end]
        return EpisodeData(
            episode_idx=episode_idx,
            start_frame=start,
            qpos=qpos_source[start:end],
            action=action,
            images=images,
            prompt=self.episode_prompt(episode_idx),
        )


def row_value(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    value: Any = row
    for part in key.split("/"):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def parquet_records(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        return []
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "LeRobot visualization requires pyarrow. Run `uv sync` after updating the project."
        ) from exc
    tables = [pq.read_table(path) for path in paths]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return table.to_pylist()


def metadata_records(meta_dir: Path, stem: str) -> list[dict[str, Any]]:
    jsonl = meta_dir / f"{stem}.jsonl"
    if jsonl.is_file():
        return [
            json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()
        ]
    parquet = meta_dir / f"{stem}.parquet"
    paths = (
        [parquet] if parquet.is_file() else sorted((meta_dir / stem).rglob("*.parquet"))
    )
    return parquet_records(paths)


def matrix_from_records(records: list[dict[str, Any]], key: str) -> np.ndarray | None:
    if not records or key not in records[0]:
        return None
    matrix = np.asarray([row[key] for row in records], dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(
            f"LeRobot {key!r} must be a finite matrix, got {matrix.shape}."
        )
    return np.ascontiguousarray(matrix)


def load_lerobot_source(path: Path) -> LeRobotReplaySource:
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    data_paths = sorted((path / "data").rglob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No parquet files found under {path / 'data'}.")
    records = parquet_records(data_paths)
    if not records:
        raise ValueError("LeRobot data parquet files contain no frames.")

    sort_key = "index" if "index" in records[0] else None
    if sort_key is not None:
        records.sort(key=lambda row: int(row[sort_key]))
    elif {"episode_index", "frame_index"} <= set(records[0]):
        records.sort(
            key=lambda row: (int(row["episode_index"]), int(row["frame_index"]))
        )
    else:
        raise KeyError("LeRobot data needs index or episode_index/frame_index columns.")

    state = matrix_from_records(records, "observation.state")
    if state is None:
        raise KeyError("LeRobot data has no observation.state column.")
    action = matrix_from_records(records, "action")
    episode_ids = np.asarray([row["episode_index"] for row in records], dtype=np.int64)
    if np.any(np.diff(episode_ids) < 0):
        raise ValueError("LeRobot episode indices are not contiguous after sorting.")
    unique_ids, counts = np.unique(episode_ids, return_counts=True)
    if not np.array_equal(unique_ids, np.arange(unique_ids.size)):
        raise ValueError(
            f"LeRobot episode indices must start at zero, got {unique_ids}."
        )
    episode_ends = np.cumsum(counts, dtype=np.int64)
    if int(info.get("total_frames", -1)) != state.shape[0]:
        raise ValueError("LeRobot info total_frames does not match parquet rows.")
    if int(info.get("total_episodes", -1)) != episode_ends.size:
        raise ValueError(
            "LeRobot info total_episodes does not match parquet episode indices."
        )

    task_rows = metadata_records(path / "meta", "tasks")
    task_by_index = {
        int(row["task_index"]): str(row.get("task", row.get("__index_level_0__", "")))
        for row in task_rows
    }
    episode_rows = metadata_records(path / "meta", "episodes")
    episode_rows.sort(key=lambda row: int(row.get("episode_index", 0)))
    if episode_rows and len(episode_rows) != episode_ends.size:
        raise ValueError(
            "LeRobot episode metadata count does not match parquet episodes."
        )

    prompts: list[str] = []
    starts = np.concatenate(([0], episode_ends[:-1])).astype(np.int64)
    for episode_idx, start in enumerate(starts):
        row = episode_rows[episode_idx] if episode_rows else {}
        tasks = row.get("tasks")
        if isinstance(tasks, (list, tuple)) and tasks:
            prompts.append(str(tasks[0]))
            continue
        task_index = records[int(start)].get("task_index")
        prompts.append(
            task_by_index.get(int(task_index), "") if task_index is not None else ""
        )

    features = info.get("features", {})
    camera_names = [
        key.removeprefix("observation.images.")
        for key, feature in features.items()
        if key.startswith("observation.images.") and feature.get("dtype") == "video"
    ]
    if not camera_names:
        raise ValueError("LeRobot info has no video camera features.")
    provenance = info.get("dexbiolab", {})
    replay_qpos = None
    replay_qpos_metadata = provenance.get("replay_qpos")
    if replay_qpos_metadata is not None:
        if not isinstance(replay_qpos_metadata, dict):
            raise ValueError(
                "LeRobot dexbiolab.replay_qpos metadata must be an object."
            )
        relative_path = Path(str(replay_qpos_metadata.get("path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Invalid replay qpos path: {relative_path}.")
        replay_qpos_path = path / relative_path
        if not replay_qpos_path.is_file():
            raise FileNotFoundError(replay_qpos_path)
        replay_qpos = np.load(replay_qpos_path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(
            int(value) for value in replay_qpos_metadata.get("shape", ())
        )
        if replay_qpos.ndim != 2 or replay_qpos.shape[0] != state.shape[0]:
            raise ValueError(
                f"Replay qpos has shape {replay_qpos.shape}, expected ({state.shape[0]}, nq)."
            )
        if expected_shape and replay_qpos.shape != expected_shape:
            raise ValueError(
                f"Replay qpos has shape {replay_qpos.shape}, metadata declares {expected_shape}."
            )
        qpos_source = str(relative_path)
    else:
        qpos_source = "observation.state (incomplete legacy fallback)"
        print(
            "Warning: dataset has no DexBioLab replay qpos sidecar; "
            "the reconstructed 3D scene may omit free objects."
        )
    print(
        f"Loaded LeRobot {info.get('codebase_version', 'unknown')}: {path}\n"
        f"episodes={episode_ends.size}, frames={state.shape[0]}, cameras={camera_names}\n"
        f"qpos source: {qpos_source}"
    )
    return LeRobotReplaySource(
        num_frames=state.shape[0],
        camera_names=camera_names,
        action_transform=str(provenance.get("action_transform", "none")),
        episode_ends=episode_ends,
        episode_prompts=np.asarray(prompts),
        dataset_path=path,
        info=info,
        state=state,
        replay_qpos=replay_qpos,
        action=action,
        episode_rows=tuple(episode_rows),
    )


def load_zarr_source(
    path: Path,
    *,
    state_key: str,
    use_replay_qpos: bool,
    action_key: str,
    rgb_key: str,
    scene_camera_names: list[str],
) -> ZarrReplaySource:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: zarr. Run this script with `uv run python ...`."
        ) from exc

    try:
        root = zarr.open_group(str(path), mode="r", zarr_format=2)
    except Exception:
        root = zarr.open_group(str(path), mode="r", zarr_format=3)
    data_group = root["data"] if "data" in root else root
    attrs = dict(root.attrs)
    has_replay_qpos = "replay" in root and "qpos" in root["replay"]
    has_state_qpos = state_key in data_group
    rendered_zarr = attrs.get("created_by") == "scripts/render_zarr.py"

    if use_replay_qpos or (rendered_zarr and has_replay_qpos):
        if not has_replay_qpos:
            raise KeyError("Zarr does not contain replay/qpos.")
        qpos_array = root["replay"]["qpos"]
        qpos_source_path = "replay/qpos"
    else:
        if not has_state_qpos:
            raise KeyError(f"Zarr data group does not contain state key {state_key!r}.")
        qpos_array = data_group[state_key]
        qpos_source_path = f"data/{state_key}"

    rgb_frame_shape = _rgb_frame_shape(data_group, rgb_key, attrs)

    camera_names = list(attrs.get("rgb_camera_names") or scene_camera_names)
    if rgb_frame_shape is not None and len(rgb_frame_shape) == 4:
        camera_names = camera_names_for_count(camera_names, rgb_frame_shape[0])
    elif rgb_frame_shape is not None and len(rgb_frame_shape) == 3:
        camera_names = camera_names_for_count(camera_names, 1)

    if "meta" not in root:
        raise KeyError("Zarr does not contain the current meta group.")
    meta = root["meta"]
    for key in ("episode_ends", "episode_env_origins", "episode_prompts"):
        if key not in meta:
            raise KeyError(f"Zarr does not contain required meta/{key}.")

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    if (
        episode_ends.ndim != 1
        or episode_ends.size == 0
        or np.any(np.diff(episode_ends) <= 0)
        or int(episode_ends[-1]) != int(qpos_array.shape[0])
    ):
        raise ValueError(
            "meta/episode_ends is inconsistent with the replay frame count."
        )
    episode_prompts = np.asarray(meta["episode_prompts"][:])
    if episode_prompts.shape != episode_ends.shape:
        raise ValueError(
            "meta/episode_prompts must contain exactly one prompt per episode."
        )

    goal_names: tuple[str, ...] = ()
    episode_goals = None
    has_goal_names = "goal_names" in meta
    has_episode_goals = "episode_goals" in meta
    if has_goal_names != has_episode_goals:
        raise ValueError(
            "meta/goal_names and meta/episode_goals must either both exist or both be absent."
        )
    if has_goal_names:
        goal_names = tuple(str(value) for value in meta["goal_names"][:])
        episode_goals = np.asarray(meta["episode_goals"][:])
        if episode_goals.shape != (episode_ends.shape[0], len(goal_names)):
            raise ValueError(
                f"meta/episode_goals has shape {episode_goals.shape}, expected "
                f"{(episode_ends.shape[0], len(goal_names))}."
            )

    print(f"Loaded zarr: {path}")
    for key in data_group.array_keys():
        arr = data_group[key]
        print(f"data/{key}: {arr.shape} {arr.dtype}")
    if "replay" in root:
        for key in root["replay"].array_keys():
            arr = root["replay"][key]
            print(f"replay/{key}: {arr.shape} {arr.dtype}")
    print(f"qpos source: {qpos_source_path}")
    if episode_prompts is not None:
        print(f"meta/episode_prompts: {episode_prompts.shape} {episode_prompts.dtype}")
    if episode_goals is not None:
        print(f"meta/goal_names: {goal_names}")
        print(f"meta/episode_goals: {episode_goals.shape} {episode_goals.dtype}")

    return ZarrReplaySource(
        num_frames=int(qpos_array.shape[0]),
        camera_names=camera_names if rgb_key in data_group else [],
        action_transform=str(attrs.get("action_transform", "none")),
        episode_ends=episode_ends,
        episode_prompts=episode_prompts,
        goal_names=goal_names,
        episode_goals=episode_goals,
        data_group=data_group,
        qpos_array=qpos_array,
        qpos_source=qpos_source_path,
        state_key=state_key,
        action_key=action_key,
        rgb_key=rgb_key,
        rgb_frame_shape=rgb_frame_shape,
    )


def _rgb_frame_shape(
    data_group: Any, rgb_key: str, attrs: dict[str, Any]
) -> tuple[int, ...] | None:
    if rgb_key not in data_group:
        return None

    shape = tuple(int(v) for v in data_group[rgb_key].shape)
    attr_shape = attrs.get("rgb_frame_shape")
    if attr_shape is not None and (
        attrs.get("rgb_layout") == "flattened_nhwc_per_timestep" or len(shape) == 2
    ):
        return tuple(int(v) for v in attr_shape)
    if len(shape) >= 2:
        return shape[1:]
    return None


def camera_names_for_count(camera_names: list[str], count: int) -> list[str]:
    names = camera_names[:count]
    names.extend(f"camera_{idx}" for idx in range(len(names), count))
    return names
