#!/usr/bin/env python3
"""Balanced episode replay-state recording for policy evaluation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .replay_state import extract_replay_state, extract_task_replay_state

_REPLAY_KEYS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "env_origin",
    "mixer_ui_state",
    "bunsen_fire_state",
    "heating_effect_state",
)


@dataclass(frozen=True)
class EpisodeReplay:
    path: Path
    success: bool
    episode: int
    env_id: int
    frames: int


def select_episode_replays(
    successes: Sequence[EpisodeReplay],
    failures: Sequence[EpisodeReplay],
    count: int,
) -> list[tuple[str, EpisodeReplay]]:
    front = [("success", replay) for replay in successes[:count]]
    front.extend(
        ("success_fill", replay)
        for replay in failures[count : count + (count - len(front))]
    )

    back = [("failure", replay) for replay in failures[:count]]
    back.extend(
        ("failure_fill", replay)
        for replay in successes[count : count + (count - len(back))]
    )
    return front + back


class EpisodeReplayRecorder:
    """Keep balanced evaluation episodes as compressed MuJoCo replay states."""

    def __init__(self, cfg: Any, env: Any) -> None:
        if cfg.replay_output_dir is None:
            raise ValueError("replay_output_dir is required for replay recording.")
        self.output_dir = cfg.replay_output_dir.expanduser().resolve()
        self.candidate_dir = self.output_dir / ".candidates"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.raw_env = env.unwrapped
        self.task_id = cfg.task_id
        self.seed = cfg.seed
        self.num_envs = int(self.raw_env.num_envs)
        self.keep_per_outcome = int(cfg.replays_per_outcome) * 2
        self.buffers: list[dict[str, list[np.ndarray]]] = [
            {} for _ in range(self.num_envs)
        ]
        self.successes: list[EpisodeReplay] = []
        self.failures: list[EpisodeReplay] = []

    def write(self, *, env_ids: Sequence[int] | None = None) -> None:
        state = extract_replay_state(self.raw_env, self.task_id)
        state.update(extract_task_replay_state(self.raw_env, self.task_id))
        selected = {key: state[key] for key in _REPLAY_KEYS if key in state}
        if "qpos" not in selected:
            raise KeyError("MuJoCo replay state has no qpos field.")

        if env_ids is None:
            env_ids = range(self.num_envs)
        env_ids = [int(env_id) for env_id in env_ids]
        for key, value in selected.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != self.num_envs:
                raise ValueError(
                    f"Replay field {key!r} has shape {array.shape}; expected first "
                    f"dimension {self.num_envs}."
                )
            for env_id in env_ids:
                self.buffers[env_id].setdefault(key, []).append(array[env_id].copy())

    def finish(self, env_id: int, *, success: bool, episode: int) -> None:
        buffer = self.buffers[env_id]
        frames = len(buffer.get("qpos", ()))
        if frames == 0:
            raise RuntimeError(
                f"Episode {episode} in env {env_id} has no replay frames."
            )

        collection = self.successes if success else self.failures
        if len(collection) < self.keep_per_outcome:
            path = self.candidate_dir / (f"env-{env_id:03d}-episode-{episode:04d}.npz")
            arrays = {
                key: np.stack(values, axis=0) for key, values in sorted(buffer.items())
            }
            metadata = {
                "format": "dexbiolab_episode_replay_v1",
                "task_id": self.task_id,
                "seed": self.seed,
                "episode": episode,
                "env_id": env_id,
                "success": success,
                "frames": frames,
                "step_dt": float(self.raw_env.step_dt),
                "replay_keys": sorted(arrays),
            }
            np.savez_compressed(
                path,
                **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            collection.append(EpisodeReplay(path, success, episode, env_id, frames))

        self.buffers[env_id] = {}

    def close(self, *, select: bool) -> None:
        self.buffers = []
        if select:
            selected = select_episode_replays(
                self.successes,
                self.failures,
                self.keep_per_outcome // 2,
            )
            selected_paths = set()
            manifest_lines = [
                "slot\trequested\tactual\tepisode\tenv_id\tframes\treplay"
            ]
            for slot, (requested, replay) in enumerate(selected, start=1):
                actual = "success" if replay.success else "failure"
                destination = self.output_dir / (
                    f"{slot:02d}_{requested}_{actual}_episode-{replay.episode:04d}"
                    f"_env-{replay.env_id:03d}.npz"
                )
                shutil.move(replay.path, destination)
                selected_paths.add(replay.path)
                manifest_lines.append(
                    f"{slot}\t{requested}\t{actual}\t{replay.episode}\t"
                    f"{replay.env_id}\t{replay.frames}\t{destination.name}"
                )
            (self.output_dir / "manifest.tsv").write_text(
                "\n".join(manifest_lines) + "\n",
                encoding="utf-8",
            )
            print(f"Episode replay states: {self.output_dir}")
            for replay in self.successes + self.failures:
                if replay.path not in selected_paths:
                    replay.path.unlink(missing_ok=True)
        else:
            for replay in self.successes + self.failures:
                replay.path.unlink(missing_ok=True)

        self.candidate_dir.rmdir()
