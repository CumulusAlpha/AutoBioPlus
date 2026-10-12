"""Monkey patch MJLab envs with a kinematic observation refresh path.

This is useful when state has already been written into ``sim.data`` and we want
fresh observations, including camera observations, without advancing physics.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mujoco_warp as mjwarp
import torch
import warp as wp
from tensordict import TensorDict


def apply_patch() -> None:
    """Install kinematic observation refresh helpers on MJLab env wrappers."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    _patch_manager_based_env(ManagerBasedRlEnv)
    _patch_rsl_wrapper(RslRlVecEnvWrapper)


def _patch_manager_based_env(env_cls: type) -> None:
    current = getattr(env_cls, "refresh_observations_from_sim", None)
    if getattr(current, "_kinematic_refresh", False):
        return

    _refresh_observations_from_sim._kinematic_refresh = True  # type: ignore[attr-defined]
    env_cls.refresh_observations_from_sim = _refresh_observations_from_sim


def _patch_rsl_wrapper(wrapper_cls: type) -> None:
    current = getattr(wrapper_cls, "refresh_observations_from_sim", None)
    if not getattr(current, "_kinematic_refresh", False):
        _rsl_refresh_observations_from_sim._kinematic_refresh = True  # type: ignore[attr-defined]
        wrapper_cls.refresh_observations_from_sim = _rsl_refresh_observations_from_sim

    current_get = getattr(wrapper_cls, "get_observations", None)
    if getattr(current_get, "_kinematic_refresh", False):
        return

    if not hasattr(wrapper_cls, "_original_get_observations"):
        wrapper_cls._original_get_observations = current_get
    _rsl_get_observations._kinematic_refresh = True  # type: ignore[attr-defined]
    wrapper_cls.get_observations = _rsl_get_observations


def _refresh_observations_from_sim(
    self,
    *,
    update_history: bool = False,
    write_scene_to_sim: bool = False,
    update_scene: bool = True,
    apply_events: bool = False,
    event_modes: Iterable[str] = ("step",),
    dt: float | None = None,
    resense_after_events: bool = True,
    include_com_pos: bool = False,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Refresh observations from the current sim state without ``mjwarp.step``.

    Args:
        update_history: Whether observation history buffers should receive the
            refreshed value.
        write_scene_to_sim: If True, first write scene/entity cached state into
            ``sim.data``. Leave False when external code already wrote sim state.
        update_scene: Invalidate scene sensor caches after kinematic refresh.
        apply_events: Whether to apply configured events before computing obs.
        event_modes: Event modes to apply when ``apply_events`` is True. Useful
            for per-step UI events.
        dt: Event dt. Defaults to the environment step dt.
        resense_after_events: Run kinematics/camera/sense again after events.
        include_com_pos: Also update ``subtree_com`` before ``camlight``. This is
            still kinematic-only and avoids ``forward()``.
    """
    if write_scene_to_sim:
        self.scene.write_data_to_sim()

    _refresh_kinematic_camera_state(self, include_com_pos=include_com_pos)
    if update_scene:
        self.scene.update(dt=0.0)
    self.sim.sense()

    if apply_events:
        _apply_refresh_events(self, event_modes=event_modes, dt=dt)
        if resense_after_events:
            _refresh_kinematic_camera_state(self, include_com_pos=include_com_pos)
            if update_scene:
                self.scene.update(dt=0.0)
            self.sim.sense()

    # Force recompute even when ObservationManager has a cached obs buffer.
    self.observation_manager._obs_buffer = None
    self.obs_buf = self.observation_manager.compute(update_history=update_history)
    return self.obs_buf


def _refresh_kinematic_camera_state(env: Any, *, include_com_pos: bool) -> None:
    sim = env.unwrapped.sim
    with wp.ScopedDevice(sim.wp_device):
        mjwarp.kinematics(sim.wp_model, sim.wp_data)
        if include_com_pos:
            mjwarp.com_pos(sim.wp_model, sim.wp_data)
        mjwarp.camlight(sim.wp_model, sim.wp_data)


def _apply_refresh_events(
    env: Any,
    *,
    event_modes: Iterable[str],
    dt: float | None,
) -> None:
    event_manager = env.event_manager
    event_dt = env.step_dt if dt is None else float(dt)
    available = set(event_manager.available_modes)
    for mode in event_modes:
        if mode not in available:
            continue
        if mode == "step":
            event_manager.apply(mode="step", dt=event_dt)
        elif mode == "interval":
            event_manager.apply(mode="interval", dt=event_dt)
        else:
            raise ValueError(
                "Kinematic observation refresh only supports step/interval events; "
                f"got {mode!r}."
            )


def _rsl_refresh_observations_from_sim(self, **kwargs) -> TensorDict:
    obs_dict = self.unwrapped.refresh_observations_from_sim(**kwargs)
    return TensorDict(obs_dict, batch_size=[self.num_envs])


def _rsl_get_observations(self) -> TensorDict:
    return self.refresh_observations_from_sim(update_history=False)
