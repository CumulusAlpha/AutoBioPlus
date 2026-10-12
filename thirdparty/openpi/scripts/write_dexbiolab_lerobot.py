#!/usr/bin/env python3
"""Consume rendered RGB from stdin and write an OpenPI LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

OPENPI_ROOT = Path(__file__).resolve().parents[1]
DEXBIOLAB_ROOT = OPENPI_ROOT.parents[1]
for path in (OPENPI_ROOT / "src", DEXBIOLAB_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.utils.lerobot_stream import iter_rgb_batches  # noqa: E402
from scripts.utils.lerobot_stream import load_collection  # noqa: E402

from src.task_goals import TASK_PROMPT_TEMPLATES  # noqa: E402
from src.task_goals import format_task_prompt as format_prompt  # noqa: E402
from src.task_goals import goal_balance_provenance  # noqa: E402
from src.task_goals import task_goal_names  # noqa: E402
from openpi.dexbiolab_schema import DEXBIOLAB_TASK_SCHEMA_BY_NAME  # noqa: E402


def _validate_prompts_and_goals(collection: Any, schema: Any) -> None:
    meta = collection.root["meta"]
    canonical_names = task_goal_names(collection.task_id) or ()
    if schema.goal_names != canonical_names:
        raise ValueError(
            f"OpenPI goal names are {schema.goal_names}, expected {canonical_names}."
        )
    canonical_template = TASK_PROMPT_TEMPLATES[collection.task_id]
    if schema.prompt_template != canonical_template:
        raise ValueError(
            f"OpenPI prompt template {schema.prompt_template!r} does not match "
            f"{canonical_template!r}."
        )
    present = {"goal_names", "episode_goals"} & set(meta.array_keys())
    if not schema.goal_names:
        if present:
            raise ValueError(f"Fixed-goal task has unexpected metadata: {sorted(present)}.")
        expected = [format_prompt(collection.task_id)] * len(collection.bounds)
    else:
        if present != {"goal_names", "episode_goals"}:
            raise KeyError(f"Variable-goal task is missing goal metadata: {sorted(present)}.")
        names = tuple(str(value) for value in meta["goal_names"][:])
        if names != schema.goal_names:
            raise ValueError(f"Goal names are {names}, expected {schema.goal_names}.")
        goals = np.asarray(meta["episode_goals"][: len(collection.bounds)])
        if goals.shape != (len(collection.bounds), len(schema.goal_names)):
            raise ValueError(f"Invalid episode_goals shape: {goals.shape}.")
        expected = [
            format_prompt(collection.task_id, row.tolist())
            for row in goals
        ]
    for index, (actual, wanted) in enumerate(zip(collection.prompts, expected, strict=True)):
        if actual != wanted:
            raise ValueError(f"Episode {index} prompt {actual!r} does not match {wanted!r}.")


def _features(
    camera_aliases: tuple[str, ...],
    *,
    state_dim: int,
    action_dim: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,)},
        "action": {"dtype": "float32", "shape": (action_dim,)},
    }
    for alias in camera_aliases:
        features[f"observation.images.{alias}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _create_dataset(output: Path, *, repo_id: str, fps: int, features: dict[str, Any]):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("LeRobot is missing; run uv sync in the DexBioLab root.") from exc
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_queue_maxsize=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--camera-names-json", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--repo-id", default="local_repo")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    schema = DEXBIOLAB_TASK_SCHEMA_BY_NAME.get(args.task_id)
    if schema is None:
        raise ValueError(f"OpenPI has no DexBioLab schema for {args.task_id!r}.")
    camera_names = tuple(str(value) for value in json.loads(args.camera_names_json))
    expected_sources = tuple(source for _, source in schema.camera_sources)
    if len(set(camera_names)) != len(camera_names) or set(camera_names) != set(expected_sources):
        raise ValueError(f"Rendered cameras are {camera_names}, expected {expected_sources}.")
    camera_aliases = tuple(alias for alias, _ in schema.camera_sources)
    camera_indices = tuple(camera_names.index(source) for source in expected_sources)

    collection = load_collection(
        args.collection,
        task_id=args.task_id,
        state_dim=schema.state_dim,
        action_dim=schema.raw_action_dim,
        position_slices=schema.position_slices,
        quaternion_slices=schema.quaternion_slices,
        max_frames=args.max_frames,
        fps=args.fps,
    )
    if schema.state_entity_names:
        attrs = collection.root.attrs
        actual_entities = tuple(str(value) for value in attrs.get("state_entity_names", ()))
        if attrs.get("state_layout") != "entity_joint_pos_concat":
            raise ValueError(f"{args.task_id} requires entity_joint_pos_concat state layout.")
        if actual_entities != schema.state_entity_names:
            raise ValueError(
                f"{args.task_id} state entities are {actual_entities}, "
                f"expected {schema.state_entity_names}."
            )
    _validate_prompts_and_goals(collection, schema)

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists. Pass --overwrite to replace it.")
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"OpenPI writer: task={args.task_id}, episodes={len(collection.bounds)}, "
        f"frames={collection.num_frames}, fps={collection.fps}, output={output}",
        flush=True,
    )
    dataset = None
    try:
        dataset = _create_dataset(
            staging,
            repo_id=args.repo_id,
            fps=collection.fps,
            features=_features(
                camera_aliases,
                state_dim=schema.state_dim,
                action_dim=schema.raw_action_dim,
                height=args.height,
                width=args.width,
            ),
        )
        episode_index = 0
        frame_count = 0
        episode_end = collection.bounds[0][1]
        for start, rgb in iter_rgb_batches(
            sys.stdin.buffer,
            num_cameras=len(camera_names),
            height=args.height,
            width=args.width,
        ):
            for offset in range(rgb.shape[0]):
                frame_index = start + offset
                frame: dict[str, Any] = {
                    "task": collection.prompts[episode_index],
                    "observation.state": collection.state[frame_index],
                    "action": collection.action[frame_index],
                }
                for camera_index, alias in zip(camera_indices, camera_aliases, strict=True):
                    frame[f"observation.images.{alias}"] = rgb[offset, camera_index]
                dataset.add_frame(frame)
                frame_count += 1
                if frame_count == episode_end:
                    dataset.save_episode()
                    episode_index += 1
                    print(
                        f"OpenPI writer: episode {episode_index}/{len(collection.bounds)} complete",
                        flush=True,
                    )
                    if episode_index < len(collection.bounds):
                        episode_end = collection.bounds[episode_index][1]
        if frame_count != collection.num_frames or episode_index != len(collection.bounds):
            raise ValueError(
                f"RGB stream ended at {frame_count} frames/{episode_index} episodes; expected "
                f"{collection.num_frames}/{len(collection.bounds)}."
            )
        dataset.finalize()
        info_path = staging / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError(
                "Expected LeRobot v3.0 dataset format, got "
                f"{info.get('codebase_version')!r}. Use lerobot==0.4.4."
            )
        info["dexbiolab"] = {
            "created_by": "scripts/render_lerobot.py",
            "source_collection": str(collection.path),
            "task_id": args.task_id,
            "camera_sources": list(camera_names),
            "action_transform": "none",
            "action_env_origin_normalization": "episode_origin_subtracted",
            "state_env_origin_normalization": "none",
        }
        balance = goal_balance_provenance(collection.root.attrs)
        if balance is not None:
            info["dexbiolab"]["goal_balance"] = balance
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"Wrote OpenPI LeRobot dataset: {output}", flush=True)


if __name__ == "__main__":
    main()
