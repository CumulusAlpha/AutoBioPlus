"""Helpers for evaluating exactly one episode per vectorized environment."""

from __future__ import annotations

from typing import Any

import torch


class FirstEpisodeStateFreezer:
    """Keep completed MJWarp worlds at their terminal state without resetting."""

    _STATE_FIELDS = (
        "time",
        "qpos",
        "qvel",
        "act",
        "qacc_warmstart",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "eq_active",
        "mocap_pos",
        "mocap_quat",
    )

    def __init__(self, env: Any) -> None:
        self.raw_env = env.unwrapped
        self._original_reset_idx = self.raw_env._reset_idx
        self._frozen = torch.zeros(
            self.raw_env.num_envs,
            device=self.raw_env.device,
            dtype=torch.bool,
        )
        data = self.raw_env.sim.data
        self._state = {
            name: getattr(data, name).clone()
            for name in self._STATE_FIELDS
        }
        self._closed = False
        self.raw_env._reset_idx = self._skip_automatic_reset

    @property
    def frozen(self) -> torch.Tensor:
        return self._frozen

    def _skip_automatic_reset(self, env_ids: torch.Tensor) -> None:
        del env_ids

    def freeze(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(
            env_ids,
            device=self.raw_env.device,
            dtype=torch.long,
        ).reshape(-1)
        if env_ids.numel() == 0:
            return
        data = self.raw_env.sim.data
        for name, frozen_value in self._state.items():
            frozen_value[env_ids] = getattr(data, name)[env_ids]
        self._frozen[env_ids] = True

    def restore(self) -> None:
        env_ids = torch.where(self._frozen)[0]
        if env_ids.numel() == 0:
            return
        data = self.raw_env.sim.data
        for name, frozen_value in self._state.items():
            getattr(data, name)[env_ids] = frozen_value[env_ids]
        for name in ("reset_buf", "reset_terminated", "reset_time_outs"):
            value = getattr(self.raw_env, name, None)
            if value is not None:
                value[env_ids] = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.raw_env._reset_idx = self._original_reset_idx

    def __enter__(self) -> FirstEpisodeStateFreezer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()
