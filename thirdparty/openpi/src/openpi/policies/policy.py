from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        return self.infer_batch([obs], noise=noise)[0]

    @override
    def infer_batch(
        self,
        observations: Sequence[dict],
        *,
        noise: np.ndarray | None = None,
    ) -> list[dict]:
        if not observations:
            raise ValueError("Batch inference requires at least one observation.")

        # Transform each observation independently, preserving the canonical
        # single-observation adapters before stacking model inputs.
        transformed = []
        for obs in observations:
            inputs = jax.tree.map(lambda x: x, obs)
            transformed.append(self._input_transform(inputs))
        inputs = jax.tree.map(
            lambda *values: np.stack([np.asarray(value) for value in values]),
            *transformed,
        )
        batch_size = len(observations)

        if not self._is_pytorch_model:
            inputs = jax.tree.map(jnp.asarray, inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.asarray(x)).to(self._pytorch_device),
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            if noise.ndim != 3 or noise.shape[0] != batch_size:
                raise ValueError(f"Noise shape {noise.shape} does not match inference batch size {batch_size}.")
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(np.asarray, outputs)

        results = []
        for batch_index in range(batch_size):
            item = jax.tree.map(
                lambda x, index=batch_index: np.asarray(x[index, ...]),
                outputs,
            )
            item = self._output_transform(item)
            item["policy_timing"] = {
                "infer_ms": model_time * 1000,
                "batch_size": batch_size,
            }
            results.append(item)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        self._record(obs, results)
        return results

    @override
    def infer_batch(self, observations: Sequence[dict]) -> list[dict]:
        results = self._policy.infer_batch(observations)
        for obs, result in zip(observations, results, strict=True):
            self._record(obs, result)
        return results

    def _record(self, obs: dict, results: dict) -> None:
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
