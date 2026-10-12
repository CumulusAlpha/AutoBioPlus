"""Asynchronous OpenPI inference and timestamped action execution."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from mjlab.rl import RslRlVecEnvWrapper

from scripts.utils.openpi_client import load_openpi_client
from scripts.utils.openpi_config import EvalOpenPIConfig, policy_camera_sources
from src.tasks.policy_action import (
    action_quaternion_slices,
    policy_action_to_environment_action,
    shift_action_positions,
)
from src.tasks.policy_state import policy_joint_positions
from src.tasks.task_goals import task_prompts_for_envs
from src.utils.chunked_evaluation import (
    TimestampedActionBuffer,
    merge_timestamped_action_chunk,
)


def to_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def resize_uint8_nhwc(image: Any, *, height: int, width: int) -> np.ndarray:
    tensor = torch.as_tensor(image)
    if tensor.ndim == 5:
        tensor = tensor[:, -1]
    if tensor.ndim == 2:
        pixels = height * width
        if tensor.shape[1] not in (pixels * 3, pixels * 4):
            raise ValueError(
                f"Flattened camera image has width {tensor.shape[1]}, expected "
                f"{pixels * 3} (RGB) or {pixels * 4} (RGBA)."
            )
        tensor = tensor.reshape(tensor.shape[0], height, width, -1)
    if tensor.ndim != 4:
        raise ValueError(f"Expected image shape [B,H,W,C], got {tuple(tensor.shape)}.")
    if tensor.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB or RGBA images, got {tuple(tensor.shape)}.")
    tensor = tensor[..., :3]

    if tensor.is_floating_point():
        if not torch.isfinite(tensor).all():
            raise ValueError("Camera image contains non-finite values.")
        max_value = float(tensor.max().item()) if tensor.numel() else 0.0
        if max_value <= 1.0 + 1e-6:
            tensor = tensor * 255.0
        tensor = tensor.clamp(0, 255)
    elif tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 255)

    if tuple(tensor.shape[1:3]) != (height, width):
        tensor = F.interpolate(
            tensor.permute(0, 3, 1, 2).to(dtype=torch.float32),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

    return to_numpy(tensor.to(dtype=torch.uint8), dtype=np.uint8)


def normalize_action_quaternions(
    action: np.ndarray,
    quaternion_slices: tuple[tuple[int, int], ...],
) -> np.ndarray:
    action = np.array(action, dtype=np.float32, copy=True)
    for start, stop in quaternion_slices:
        quaternion = action[:, start:stop]
        norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        quaternion[valid] /= norms[valid]
        quaternion[~valid] = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    return action


def load_quantile_bounds(
    path: Path | None,
    *,
    state_dim: int,
    action_dim: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    stats = payload.get("norm_stats", payload)
    try:
        state_q01 = np.asarray(stats["state"]["q01"], dtype=np.float32)
        state_q99 = np.asarray(stats["state"]["q99"], dtype=np.float32)
        action_q01 = np.asarray(stats["actions"]["q01"], dtype=np.float32)
        action_q99 = np.asarray(stats["actions"]["q99"], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(
            f"Normalization stats in {path} have no required quantiles."
        ) from exc
    if state_q01.shape != (state_dim,) or state_q99.shape != (state_dim,):
        raise ValueError(
            f"State quantiles in {path} have shapes {state_q01.shape}/{state_q99.shape}, "
            f"expected ({state_dim},)."
        )
    if action_q01.shape != (action_dim,) or action_q99.shape != (action_dim,):
        raise ValueError(
            f"Action quantiles in {path} have shapes {action_q01.shape}/{action_q99.shape}, "
            f"expected ({action_dim},)."
        )
    if not all(
        np.isfinite(value).all()
        for value in (state_q01, state_q99, action_q01, action_q99)
    ):
        raise ValueError(
            f"Normalization quantiles in {path} contain non-finite values."
        )
    if np.any(state_q01 > state_q99) or np.any(action_q01 > action_q99):
        raise ValueError(f"Normalization quantiles in {path} are not ordered.")
    action_range = action_q99 - action_q01
    return (
        state_q01,
        state_q99,
        action_q01 - margin * action_range,
        action_q99 + margin * action_range,
    )


@dataclass(frozen=True)
class _InferenceRequest:
    request_id: int
    env_id: int
    generation: int
    timestamp: int
    state: np.ndarray
    images: dict[str, np.ndarray]
    prompt: str


@dataclass(frozen=True)
class _InferenceResult:
    request: _InferenceRequest
    actions: np.ndarray | None
    infer_ms: float
    error: BaseException | None = None


class OpenPIChunkPolicy:
    def __init__(self, cfg: EvalOpenPIConfig, env: RslRlVecEnvWrapper) -> None:
        if not 0.0 < cfg.replan_ratio <= 1.0:
            raise ValueError("replan_ratio must be in (0, 1].")

        self.cfg = cfg
        self.env = env
        self.num_envs = env.unwrapped.num_envs
        self.prompt_override = cfg.prompt
        self.prompts = [cfg.prompt or ""] * self.num_envs
        self.client = load_openpi_client().WebsocketClientPolicy(
            host=cfg.host, port=cfg.port
        )
        client_socket = getattr(getattr(self.client, "_ws", None), "socket", None)
        if client_socket is None:
            raise RuntimeError(
                "OpenPI websocket client does not expose its transport socket."
            )
        client_socket.settimeout(cfg.request_timeout_s)
        self.inference_batch_size = min(cfg.inference_batch_size, self.num_envs)
        if self.inference_batch_size > 1 and not self.client.supports_batch_inference():
            raise RuntimeError(
                "The OpenPI policy server does not advertise batched inference. "
                "Sync the matching OpenPI server or pass --inference-batch-size 1."
            )
        print(
            "OpenPI inference transport: "
            f"batch_size={self.inference_batch_size}, "
            "fixed-size padding enabled."
        )

        self.camera_sources = policy_camera_sources(cfg)
        self.action_buffers: list[TimestampedActionBuffer | None] = [
            None
        ] * self.num_envs
        self.timestamps = np.zeros(self.num_envs, dtype=np.int64)
        self.generations = np.zeros(self.num_envs, dtype=np.int64)
        self.request_pending = np.zeros(self.num_envs, dtype=np.bool_)
        self.active = np.ones(self.num_envs, dtype=np.bool_)
        self.action_horizon: int | None = None
        self._request_queue: Queue[tuple[_InferenceRequest, ...] | None] = Queue()
        self._result_queue: Queue[_InferenceResult] = Queue()
        self._worker_stop = threading.Event()
        self._fatal_error: BaseException | None = None
        self._closed = False
        self.quaternion_slices = action_quaternion_slices(cfg.task_id)
        state_dim = int(policy_joint_positions(env.unwrapped, cfg.task_id).shape[-1])
        action_dim = int(env.unwrapped.action_space.shape[-1])
        self.quantile_bounds = load_quantile_bounds(
            cfg.norm_stats_path,
            state_dim=state_dim,
            action_dim=action_dim,
            margin=cfg.action_clip_margin,
        )
        self.episode_requests = np.zeros(env.unwrapped.num_envs, dtype=np.int64)
        self.episode_infer_ms = np.zeros(env.unwrapped.num_envs, dtype=np.float64)
        self.episode_state_values = np.zeros(env.unwrapped.num_envs, dtype=np.int64)
        self.episode_state_ood = np.zeros(env.unwrapped.num_envs, dtype=np.int64)
        self.episode_state_ood_by_dim = np.zeros(
            (env.unwrapped.num_envs, state_dim),
            dtype=np.int64,
        )
        self.episode_action_values = np.zeros(env.unwrapped.num_envs, dtype=np.int64)
        self.episode_action_outside = np.zeros(env.unwrapped.num_envs, dtype=np.int64)
        self.episode_action_outside_by_dim = np.zeros(
            (env.unwrapped.num_envs, action_dim),
            dtype=np.int64,
        )
        self.episode_max_action_jump = np.zeros(
            env.unwrapped.num_envs, dtype=np.float64
        )
        self.episode_hold_steps = np.zeros(env.unwrapped.num_envs, dtype=np.int64)
        self.last_actions: list[np.ndarray | None] = [None] * env.unwrapped.num_envs
        self.total_requests = 0
        self.total_infer_ms = 0.0
        self.total_inference_batches = 0
        self.total_padded_inference_items = 0
        self.total_state_values = 0
        self.total_state_ood = 0
        self.total_state_ood_by_dim = np.zeros(state_dim, dtype=np.int64)
        self.total_action_values = 0
        self.total_action_outside = 0
        self.total_action_outside_by_dim = np.zeros(action_dim, dtype=np.int64)
        self.total_hold_steps = 0
        self._reported_action_horizon = False
        self._reported_camera_stats = False
        self._reported_buffer_underrun = False
        self._refresh_prompts()
        if self.quantile_bounds is None:
            print(
                "Warning: no normalization stats were supplied; state/action "
                "training-range diagnostics are disabled."
            )
        else:
            mode = (
                "enabled (non-canonical override)"
                if cfg.clip_actions_to_training_quantiles
                else "disabled; diagnostics only"
            )
            print(
                f"Loaded training quantiles from {cfg.norm_stats_path}; "
                f"empirical action clipping is {mode}, "
                f"margin={cfg.action_clip_margin:g}."
            )
        self._worker = threading.Thread(
            target=self._inference_worker,
            name="openpi-inference",
            daemon=True,
        )
        self._worker.start()

    def reset(self) -> None:
        self.action_buffers = [None] * self.num_envs
        self.timestamps.fill(0)
        self.generations += 1
        self.request_pending.fill(False)
        self.active.fill(True)
        self.last_actions = [None] * self.num_envs
        self._reset_episode_metrics(range(self.env.unwrapped.num_envs))
        self._refresh_prompts()

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        ids = to_numpy(env_ids, dtype=np.int64).reshape(-1)
        for env_id in ids:
            env_id = int(env_id)
            self.action_buffers[env_id] = None
            self.timestamps[env_id] = 0
            self.generations[env_id] += 1
            self.request_pending[env_id] = False
            self.active[env_id] = True
            self.last_actions[env_id] = None
        self._refresh_prompts(ids)

    def retire_envs(self, env_ids: torch.Tensor) -> None:
        ids = to_numpy(env_ids, dtype=np.int64).reshape(-1)
        for raw_env_id in ids:
            env_id = int(raw_env_id)
            self.active[env_id] = False
            self.action_buffers[env_id] = None
            self.generations[env_id] += 1
            self.request_pending[env_id] = False

    def _reset_episode_metrics(self, env_ids: Any) -> None:
        ids = np.asarray(list(env_ids), dtype=np.int64)
        for values in (
            self.episode_requests,
            self.episode_infer_ms,
            self.episode_state_values,
            self.episode_state_ood,
            self.episode_action_values,
            self.episode_action_outside,
            self.episode_max_action_jump,
            self.episode_hold_steps,
        ):
            values[ids] = 0
        self.episode_state_ood_by_dim[ids] = 0
        self.episode_action_outside_by_dim[ids] = 0

    def pop_episode_diagnostics(self, env_ids: Any) -> dict[int, dict[str, Any]]:
        ids = [int(value) for value in to_numpy(env_ids, dtype=np.int64).reshape(-1)]
        result = {}
        for env_id in ids:
            requests = int(self.episode_requests[env_id])
            state_values = int(self.episode_state_values[env_id])
            action_values = int(self.episode_action_values[env_id])
            result[env_id] = {
                "policy_requests": requests,
                "mean_infer_ms": (
                    float(self.episode_infer_ms[env_id] / requests) if requests else 0.0
                ),
                "state_ood_values": int(self.episode_state_ood[env_id]),
                "state_values": state_values,
                "state_ood_fraction": (
                    float(self.episode_state_ood[env_id] / state_values)
                    if state_values
                    else 0.0
                ),
                "state_ood_by_dim": self.episode_state_ood_by_dim[env_id].tolist(),
                "action_outside_training_range_values": int(
                    self.episode_action_outside[env_id]
                ),
                "action_values": action_values,
                "action_outside_training_range_fraction": (
                    float(self.episode_action_outside[env_id] / action_values)
                    if action_values
                    else 0.0
                ),
                "action_outside_training_range_by_dim": self.episode_action_outside_by_dim[
                    env_id
                ].tolist(),
                "empirical_action_clipping_enabled": (
                    self.cfg.clip_actions_to_training_quantiles
                ),
                "max_executed_action_jump": float(self.episode_max_action_jump[env_id]),
                "action_buffer_hold_steps": int(self.episode_hold_steps[env_id]),
            }
        self._reset_episode_metrics(ids)
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "policy_requests": self.total_requests,
            "inference_batches": self.total_inference_batches,
            "mean_model_batch_size": (
                (self.total_requests + self.total_padded_inference_items)
                / self.total_inference_batches
                if self.total_inference_batches
                else 0.0
            ),
            "padded_inference_items": self.total_padded_inference_items,
            "mean_infer_ms": (
                self.total_infer_ms / self.total_requests
                if self.total_requests
                else 0.0
            ),
            "state_ood_values": self.total_state_ood,
            "state_values": self.total_state_values,
            "state_ood_fraction": (
                self.total_state_ood / self.total_state_values
                if self.total_state_values
                else 0.0
            ),
            "state_ood_by_dim": self.total_state_ood_by_dim.tolist(),
            "action_outside_training_range_values": self.total_action_outside,
            "action_values": self.total_action_values,
            "action_outside_training_range_fraction": (
                self.total_action_outside / self.total_action_values
                if self.total_action_values
                else 0.0
            ),
            "action_outside_training_range_by_dim": (
                self.total_action_outside_by_dim.tolist()
            ),
            "action_buffer_hold_steps": self.total_hold_steps,
        }

    def needs_replan(self) -> bool:
        self._drain_ready_results()
        return any(self._should_request(env_id) for env_id in range(self.num_envs))

    def _refresh_prompts(self, env_ids: Any = None) -> None:
        ids = (
            list(range(self.env.unwrapped.num_envs))
            if env_ids is None
            else [int(value) for value in np.asarray(env_ids).reshape(-1)]
        )
        prompts = task_prompts_for_envs(
            self.env.unwrapped,
            self.cfg.task_id,
            ids,
            override=self.prompt_override,
        )
        for env_id, prompt in zip(ids, prompts, strict=True):
            self.prompts[env_id] = prompt

    def _inference_worker(self) -> None:
        while not self._worker_stop.is_set():
            requests = self._request_queue.get()
            if requests is None:
                return
            try:
                payloads = [
                    {
                        "state": request.state,
                        "prompt": request.prompt,
                        **request.images,
                    }
                    for request in requests
                ]
                padded_items = self.inference_batch_size - len(payloads)
                if padded_items < 0:
                    raise RuntimeError(
                        f"Inference request batch has {len(payloads)} items, "
                        f"exceeding configured size {self.inference_batch_size}."
                    )
                if padded_items:
                    payloads.extend([payloads[-1]] * padded_items)
                start_time = time.perf_counter()
                if self.inference_batch_size == 1:
                    responses = [self.client.infer(payloads[0])]
                else:
                    responses = self.client.infer_batch(payloads)
                infer_ms = (time.perf_counter() - start_time) * 1000.0
                self.total_inference_batches += 1
                self.total_padded_inference_items += padded_items
                if len(responses) != self.inference_batch_size:
                    raise ValueError(
                        f"OpenPI returned {len(responses)} batched result(s), "
                        f"expected {self.inference_batch_size}."
                    )
                result_actions = []
                for request, response in zip(requests, responses, strict=False):
                    if "actions" not in response:
                        raise KeyError(
                            f"OpenPI response has no 'actions' key: {sorted(response)}"
                        )
                    actions = np.asarray(response["actions"], dtype=np.float32)
                    result_actions.append((request, np.ascontiguousarray(actions)))
                for request, actions in result_actions:
                    self._result_queue.put(
                        _InferenceResult(
                            request=request,
                            actions=actions,
                            infer_ms=infer_ms,
                        )
                    )
            except Exception as exc:
                for request in requests:
                    self._result_queue.put(
                        _InferenceResult(
                            request=request,
                            actions=None,
                            infer_ms=0.0,
                            error=exc,
                        )
                    )
                return

    def _trim_buffer(self, env_id: int) -> TimestampedActionBuffer | None:
        buffer = self.action_buffers[env_id]
        if buffer is not None:
            buffer = buffer.discard_before(int(self.timestamps[env_id]))
            self.action_buffers[env_id] = buffer
        return buffer

    def _should_request(self, env_id: int) -> bool:
        if not self.active[env_id]:
            return False
        if self.request_pending[env_id]:
            return False
        buffer = self._trim_buffer(env_id)
        if buffer is None or self.action_horizon is None:
            return True
        return buffer.actions.shape[0] < self.cfg.replan_ratio * self.action_horizon

    def _record_state_diagnostics(
        self,
        env_id: int,
        state: np.ndarray,
    ) -> None:
        if self.quantile_bounds is None:
            return
        state_q01, state_q99, _, _ = self.quantile_bounds
        state_ood = (state < state_q01) | (state > state_q99)
        ood_count = int(state_ood.sum())
        value_count = int(state_ood.size)
        self.episode_state_ood[env_id] += ood_count
        self.episode_state_values[env_id] += value_count
        self.episode_state_ood_by_dim[env_id] += state_ood.astype(np.int64)
        self.total_state_ood += ood_count
        self.total_state_values += value_count
        self.total_state_ood_by_dim += state_ood.astype(np.int64)

    def _submit_requests(
        self,
        env_ids: list[int],
        *,
        state: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> None:
        if state.shape[0] != len(env_ids) or any(
            image.shape[0] != len(env_ids) for image in images.values()
        ):
            raise ValueError("OpenPI replan batch dimensions do not match.")
        if not np.isfinite(state).all():
            locations = np.argwhere(~np.isfinite(state))
            raise ValueError(
                "DexBioLab qpos contains non-finite values at replan batch/env "
                f"indices {locations[:16].tolist()} (showing at most 16)."
            )

        requests = []
        for batch_id, env_id in enumerate(env_ids):
            if self.request_pending[env_id]:
                raise RuntimeError(
                    f"Environment {env_id} already has an OpenPI request pending."
                )
            current_state = np.ascontiguousarray(state[batch_id]).copy()
            self._record_state_diagnostics(env_id, current_state)
            request_id = self.total_requests + 1
            request = _InferenceRequest(
                request_id=request_id,
                env_id=env_id,
                generation=int(self.generations[env_id]),
                timestamp=int(self.timestamps[env_id]),
                state=current_state,
                images={
                    key: np.ascontiguousarray(image[batch_id]).copy()
                    for key, image in images.items()
                },
                prompt=self.prompts[env_id],
            )
            self.total_requests += 1
            self.episode_requests[env_id] += 1
            self.request_pending[env_id] = True
            requests.append(request)
            if (
                request_id == 1
                or request_id % self.cfg.diagnostic_interval_replans == 0
            ):
                print(
                    f"policy request: requests={request_id}, env={env_id}, "
                    f"timestamp={request.timestamp}, "
                    f"timeout_s={self.cfg.request_timeout_s:g}",
                    flush=True,
                )

        for start in range(0, len(requests), self.inference_batch_size):
            self._request_queue.put(
                tuple(requests[start : start + self.inference_batch_size])
            )

        if not self._reported_camera_stats:
            summary = ", ".join(
                f"{name}:shape={value.shape[1:]},range="
                f"[{int(value.min())},{int(value.max())}],mean={float(value.mean()):.1f}"
                for name, value in images.items()
            )
            print(f"First policy camera batch: {summary}")
            self._reported_camera_stats = True

    def _accept_result(self, result: _InferenceResult) -> None:
        request = result.request
        env_id = request.env_id
        if request.generation != self.generations[env_id]:
            return
        if result.error is not None:
            self._fatal_error = result.error
            if request.generation == self.generations[env_id]:
                self.request_pending[env_id] = False
            raise RuntimeError(
                f"OpenPI inference request {request.request_id} failed."
            ) from result.error
        self.request_pending[env_id] = False
        self.episode_infer_ms[env_id] += result.infer_ms
        self.total_infer_ms += result.infer_ms
        chunk = result.actions
        assert chunk is not None
        expected_action_dim = int(self.env.unwrapped.action_space.shape[-1])
        if chunk.ndim != 2 or chunk.shape[1] != expected_action_dim:
            raise ValueError(
                f"OpenPI action chunk shape {chunk.shape} does not match "
                f"(*, {expected_action_dim})."
            )
        if chunk.shape[0] == 0:
            raise ValueError("OpenPI returned an empty action chunk.")
        if self.action_horizon is None:
            self.action_horizon = int(chunk.shape[0])
        elif chunk.shape[0] != self.action_horizon:
            raise ValueError(
                f"OpenPI action horizon changed from {self.action_horizon} to "
                f"{chunk.shape[0]}."
            )
        if not np.isfinite(chunk).all():
            raise ValueError("OpenPI returned non-finite actions.")

        outside_count = 0
        outside_by_dim = np.zeros(expected_action_dim, dtype=np.int64)
        if self.quantile_bounds is not None:
            _, _, action_low, action_high = self.quantile_bounds
            outside = (chunk < action_low) | (chunk > action_high)
            outside_count = int(outside.sum())
            value_count = int(outside.size)
            self.episode_action_outside[env_id] += outside_count
            self.episode_action_values[env_id] += value_count
            outside_by_dim = outside.sum(axis=0, dtype=np.int64)
            self.episode_action_outside_by_dim[env_id] += outside_by_dim
            self.total_action_outside += outside_count
            self.total_action_values += value_count
            self.total_action_outside_by_dim += outside_by_dim
            if self.cfg.clip_actions_to_training_quantiles:
                chunk = np.clip(chunk, action_low, action_high)

        chunk = normalize_action_quaternions(chunk, self.quaternion_slices)
        device_chunk = torch.as_tensor(
            np.ascontiguousarray(chunk),
            device=self.env.unwrapped.device,
            dtype=torch.float32,
        )
        origins = self.env.unwrapped.scene.env_origins[env_id].expand(
            device_chunk.shape[0],
            -1,
        )
        shifted_chunk = shift_action_positions(
            device_chunk,
            origins,
            self.cfg.task_id,
            scale=1.0,
        )
        self.action_buffers[env_id] = merge_timestamped_action_chunk(
            self.action_buffers[env_id],
            shifted_chunk,
            new_timestamp=request.timestamp,
            now_timestamp=int(self.timestamps[env_id]),
            quaternion_slices=self.quaternion_slices,
        )

        if not self._reported_action_horizon:
            print(
                f"OpenPI returned {self.action_horizon}-step action chunks; "
                "inference runs asynchronously, replanning starts when the "
                f"remaining buffer is below {self.cfg.replan_ratio:.0%}, and "
                "time-aligned overlap is blended from the old plan to the new."
            )
            self._reported_action_horizon = True

        if (
            request.request_id == 1
            or request.request_id % self.cfg.diagnostic_interval_replans == 0
        ):
            outside_dimensions = np.where(outside_by_dim > 0)[0].tolist()
            buffered = self.action_buffers[env_id]
            print(
                "policy diagnostic: "
                f"requests={request.request_id}, infer_ms={result.infer_ms:.1f}, "
                f"env={env_id}, request_ts={request.timestamp}, "
                f"receive_ts={self.timestamps[env_id]}, "
                f"state=[{request.state.min():.4g},{request.state.max():.4g}], "
                f"action=[{chunk.min():.4g},{chunk.max():.4g}], "
                f"first_pos={np.round(chunk[0, :3], 4).tolist()}, "
                f"last_pos={np.round(chunk[-1, :3], 4).tolist()}, "
                f"buffer={buffered.actions.shape[0] if buffered is not None else 0}, "
                f"outside_training_range={outside_count}/{chunk.size}, "
                f"outside_dims={outside_dimensions}",
                flush=True,
            )

    def _drain_ready_results(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(
                "OpenPI inference worker has failed."
            ) from self._fatal_error
        while True:
            try:
                result = self._result_queue.get_nowait()
            except Empty:
                return
            self._accept_result(result)

    def _wait_for_initial_action_buffers(self) -> None:
        missing = [
            env_id
            for env_id in range(self.num_envs)
            if self.active[env_id]
            and self._trim_buffer(env_id) is None
            and self.last_actions[env_id] is None
        ]
        if not missing:
            return
        if any(not self.request_pending[env_id] for env_id in missing):
            raise RuntimeError(
                f"No action or pending request for environments {missing}."
            )
        deadline = time.monotonic() + self.cfg.request_timeout_s + 5.0 * self.num_envs
        while missing:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for asynchronous OpenPI actions for "
                    f"environments {missing}."
                )
            try:
                result = self._result_queue.get(timeout=remaining)
            except Empty as exc:
                raise TimeoutError(
                    "Timed out waiting for asynchronous OpenPI inference."
                ) from exc
            self._accept_result(result)
            missing = [
                env_id
                for env_id in range(self.num_envs)
                if self.active[env_id]
                and self._trim_buffer(env_id) is None
                and self.last_actions[env_id] is None
            ]

    def __call__(self, obs: Any) -> torch.Tensor:
        self._drain_ready_results()
        query_ids = [
            env_id for env_id in range(self.num_envs) if self._should_request(env_id)
        ]
        if query_ids:
            device_ids = torch.as_tensor(
                query_ids,
                device=self.env.unwrapped.device,
                dtype=torch.long,
            )
            state = to_numpy(
                policy_joint_positions(self.env.unwrapped, self.cfg.task_id)[
                    device_ids
                ],
                dtype=np.float32,
            )
            images = {}
            for alias, source in self.camera_sources.items():
                if source not in obs:
                    raise KeyError(f"Camera observation {source!r} is missing.")
                images[alias] = resize_uint8_nhwc(
                    obs[source][device_ids],
                    height=self.cfg.camera_height,
                    width=self.cfg.camera_width,
                )
            self._submit_requests(
                query_ids,
                state=state,
                images=images,
            )

        # The first action must come from the policy. Later buffer underruns
        # hold the previous absolute target while inference continues, which
        # matches DexJoCo's asynchronous stay behavior without freezing the
        # simulator or viewer.
        self._wait_for_initial_action_buffers()
        self._drain_ready_results()
        actions = []
        for env_id in range(self.num_envs):
            if not self.active[env_id]:
                previous = self.last_actions[env_id]
                if previous is None:
                    raise RuntimeError(
                        f"Retired environment {env_id} has no action to hold."
                    )
                actions.append(
                    torch.as_tensor(
                        previous,
                        device=self.env.unwrapped.device,
                        dtype=torch.float32,
                    )
                )
                continue
            timestamp = int(self.timestamps[env_id])
            buffer = self._trim_buffer(env_id)
            if buffer is None:
                previous = self.last_actions[env_id]
                if previous is None:
                    raise RuntimeError(
                        f"OpenPI has no action to hold for environment {env_id}."
                    )
                action = torch.as_tensor(
                    previous,
                    device=self.env.unwrapped.device,
                    dtype=torch.float32,
                )
                remaining = None
                self.episode_hold_steps[env_id] += 1
                self.total_hold_steps += 1
                if not self._reported_buffer_underrun:
                    print(
                        "OpenPI action buffer underrun; holding the previous "
                        "absolute target while asynchronous inference finishes.",
                        flush=True,
                    )
                    self._reported_buffer_underrun = True
            else:
                action, remaining = buffer.pop(timestamp)
            actions.append(action)
            self.action_buffers[env_id] = remaining
            self.timestamps[env_id] += 1

        action = torch.stack(actions)
        expected_shape = tuple(self.env.unwrapped.action_space.shape)
        if tuple(action.shape) != expected_shape:
            raise ValueError(
                f"OpenPI action shape {tuple(action.shape)} != env action shape {expected_shape}."
            )
        action_numpy = to_numpy(action, dtype=np.float32)
        for env_id, current in enumerate(action_numpy):
            if not self.active[env_id]:
                continue
            previous = self.last_actions[env_id]
            if previous is not None:
                self.episode_max_action_jump[env_id] = max(
                    self.episode_max_action_jump[env_id],
                    float(np.max(np.abs(current - previous))),
                )
            self.last_actions[env_id] = current.copy()
        return policy_action_to_environment_action(action, self.cfg.task_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker_stop.set()
        self._request_queue.put(None)
        websocket = getattr(self.client, "_ws", None)
        if websocket is not None:
            websocket.close()
        self._worker.join(timeout=2.0)
