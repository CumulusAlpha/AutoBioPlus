"""Simulation and task state snapshots shared by collection and evaluation."""

from __future__ import annotations

from typing import Any

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv

from src.tasks.policy_state import policy_joint_positions
from src.utils.arrays import snapshot_to_numpy

_XHAND_BUNSEN_TASK_ID = "XHand-Bunsen-Manipulate"
_WUJI_HEATING_PLATE_TASK_ID = "Wuji-HeatingPlate-Rotate"


def mujoco_names(model: Any, obj_type: Any, count: int) -> list[str]:
    names = []
    for idx in range(count):
        name = mujoco.mj_id2name(model, obj_type, idx)
        names.append(name if name is not None else str(idx))
    return names


def extract_body_names(raw_env: ManagerBasedRlEnv) -> list[str]:
    model = raw_env.sim.mj_model
    return mujoco_names(model, mujoco.mjtObj.mjOBJ_BODY, int(model.nbody))


def extract_replay_state(raw_env: ManagerBasedRlEnv, task_id: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for out_key, field_name in (
        ("time", "time"),
        ("qpos", "qpos"),
        ("qvel", "qvel"),
        ("act", "act"),
        ("ctrl", "ctrl"),
        ("qacc_warmstart", "qacc_warmstart"),
        ("eq_active", "eq_active"),
        ("mocap_pos", "mocap_pos"),
        ("mocap_quat", "mocap_quat"),
        ("body_pos_w", "xpos"),
        ("body_quat_w", "xquat"),
    ):
        value = getattr(raw_env.sim.data, field_name, None)
        if value is not None:
            state[out_key] = snapshot_to_numpy(value, name=out_key)
    state["current_qpos"] = snapshot_to_numpy(
        policy_joint_positions(raw_env, task_id), name="robot_current_qpos"
    )
    state["env_origin"] = snapshot_to_numpy(
        raw_env.scene.env_origins[: raw_env.num_envs], name="env_origin"
    )
    return state


def extract_xhand_mixerpanel_replay_state(
    raw_env: ManagerBasedRlEnv,
) -> dict[str, Any]:
    """Extract task-local mixer UI state needed for deterministic replay."""
    from src.tasks.mdp.events import get_mixer_ui_state

    state: dict[str, Any] = {}
    ui_state = get_mixer_ui_state(raw_env)
    if ui_state is not None:
        state["mixer_ui_state"] = snapshot_to_numpy(
            torch.stack(
                (
                    ui_state["time_s"][: raw_env.num_envs],
                    ui_state["temp_c"][: raw_env.num_envs],
                    ui_state["freq_rpm"][: raw_env.num_envs],
                ),
                dim=-1,
            ),
            name="mixer_ui_state",
        )

    return state


def extract_xhand_bunsen_manipulate_replay_state(
    raw_env: ManagerBasedRlEnv,
) -> dict[str, Any]:
    from src.tasks.config.xhand_bunsen_manipulate.env_cfgs import (
        BUNSEN_FIRE_BASE_COLOR_MIX_ATTR,
        BUNSEN_FIRE_ENABLED_ATTR,
    )

    enabled = getattr(raw_env, BUNSEN_FIRE_ENABLED_ATTR, None)
    base_color_mix = getattr(raw_env, BUNSEN_FIRE_BASE_COLOR_MIX_ATTR, None)
    if enabled is None or base_color_mix is None:
        raise RuntimeError("Bunsen fire state was not initialized on the environment.")

    fire_state = torch.stack(
        (
            torch.as_tensor(enabled, device=raw_env.device, dtype=torch.float32)[
                : raw_env.num_envs
            ],
            torch.as_tensor(base_color_mix, device=raw_env.device, dtype=torch.float32)[
                : raw_env.num_envs
            ],
        ),
        dim=-1,
    )
    return {
        "bunsen_fire_state": snapshot_to_numpy(fire_state, name="bunsen_fire_state")
    }


def extract_wuji_heating_plate_replay_state(
    raw_env: ManagerBasedRlEnv,
) -> dict[str, Any]:
    from src.tasks.config.wuji_heating_plate_rotate.env_cfgs import (
        HEATING_EFFECT_INTENSITY_ATTR,
        HEATING_EFFECT_STIR_MOMENTUM_ATTR,
        HEATING_EFFECT_STIR_PHASE_ATTR,
    )

    values = [
        getattr(raw_env, attr, None)
        for attr in (
            HEATING_EFFECT_INTENSITY_ATTR,
            HEATING_EFFECT_STIR_MOMENTUM_ATTR,
            HEATING_EFFECT_STIR_PHASE_ATTR,
        )
    ]
    if any(value is None for value in values):
        raise RuntimeError(
            "Heating effect state was not initialized on the environment."
        )

    heating_effect_state = torch.stack(
        tuple(
            torch.as_tensor(value, device=raw_env.device, dtype=torch.float32)[
                : raw_env.num_envs
            ]
            for value in values
        ),
        dim=-1,
    )
    return {
        "heating_effect_state": snapshot_to_numpy(
            heating_effect_state,
            name="heating_effect_state",
        )
    }


def extract_task_replay_state(
    raw_env: ManagerBasedRlEnv, task_id: str
) -> dict[str, Any]:
    if task_id == "XHand-MixerPanel-Manipulate":
        return extract_xhand_mixerpanel_replay_state(raw_env)
    if task_id == _XHAND_BUNSEN_TASK_ID:
        return extract_xhand_bunsen_manipulate_replay_state(raw_env)
    if task_id == _WUJI_HEATING_PLATE_TASK_ID:
        return extract_wuji_heating_plate_replay_state(raw_env)
    return {}
