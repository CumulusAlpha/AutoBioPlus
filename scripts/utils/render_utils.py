"""Shared replay helpers for direct LeRobot rendering."""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul

from src.tasks.mdp.events import get_mixer_ui_state, update_mixer_display_texture
from src.tasks.task_goals import task_goal_names

_MIXER_PANEL_TASK_ID = "XHand-MixerPanel-Manipulate"
_MIXER_UI_VALUE_KEYS = ("time_s", "temp_c", "freq_rpm")
_XHAND_BUNSEN_TASK_ID = "XHand-Bunsen-Manipulate"
_WUJI_HEATING_PLATE_TASK_ID = "Wuji-HeatingPlate-Rotate"


def is_camera_sensor_cfg(sensor_cfg: object) -> bool:
    from mugs_mjlab.sensors import GaussianSensorMjlabCfg

    if isinstance(sensor_cfg, GaussianSensorMjlabCfg):
        return True
    cls = sensor_cfg.__class__
    return cls.__name__ == "CameraSensorCfg" or (
        cls.__module__.startswith("mugs_mjlab.") and "Gaussian" in cls.__name__
    )


def camera_sensor_cfgs(env_cfg) -> list[object]:
    return [
        sensor_cfg
        for sensor_cfg in getattr(env_cfg.scene, "sensors", ())
        if is_camera_sensor_cfg(sensor_cfg)
    ]


def configured_camera_resolution(sensor_cfg: object) -> tuple[int, int]:
    return int(getattr(sensor_cfg, "width")), int(getattr(sensor_cfg, "height"))


def configure_camera_resolutions(
    env_cfg,
    *,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    camera_cfgs = camera_sensor_cfgs(env_cfg)
    if not camera_cfgs:
        raise RuntimeError("No camera sensors were found in env.cfg.scene.sensors.")

    widths = [int(getattr(sensor_cfg, "width")) for sensor_cfg in camera_cfgs]
    heights = [int(getattr(sensor_cfg, "height")) for sensor_cfg in camera_cfgs]
    target_width = int(width) if width is not None else max(widths)
    target_height = int(height) if height is not None else max(heights)

    for sensor_cfg in camera_cfgs:
        old_width, old_height = configured_camera_resolution(sensor_cfg)
        if (old_width, old_height) == (target_width, target_height):
            continue
        sensor_cfg.width = target_width
        sensor_cfg.height = target_height
        name = getattr(sensor_cfg, "name", sensor_cfg.__class__.__name__)
        print(
            f"Camera {name!r}: configured resolution "
            f"{old_width}x{old_height} -> {target_width}x{target_height}",
            flush=True,
        )

    return target_width, target_height


def configured_camera_sensor_names(env: ManagerBasedRlEnv) -> list[str]:
    names: list[str] = []
    for sensor_cfg in getattr(env.cfg.scene, "sensors", ()):
        if not is_camera_sensor_cfg(sensor_cfg):
            continue
        name = getattr(sensor_cfg, "name", None)
        if name is not None and hasattr(env.scene[name], "camera_idx"):
            names.append(name)
    return names


def sample_episode_camera_quaternions(
    env: ManagerBasedRlEnv,
    sensor_names: list[str],
    *,
    num_episodes: int,
    rotation_range_deg: tuple[float, float],
    seed: int | None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Sample one fixed local-frame camera rotation per episode and sensor."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    min_deg, max_deg = rotation_range_deg
    if min_deg < 0.0 or max_deg < min_deg:
        raise ValueError(f"Invalid camera rotation range {rotation_range_deg} degrees.")

    if "cam_quat" not in env.sim.expanded_fields:
        env.sim.expand_model_fields(("cam_quat",))

    generator = torch.Generator(device=env.device)
    actual_seed = int(generator.seed() if seed is None else seed)
    if seed is not None:
        generator.manual_seed(actual_seed)

    default_camera_quats = env.sim.get_default_field("cam_quat")
    sampled: dict[str, torch.Tensor] = {}
    for sensor_name in sensor_names:
        camera_id = env.scene[sensor_name].camera_idx
        base_quat = default_camera_quats[camera_id].expand(num_episodes, 4)
        axis = torch.randn(
            (num_episodes, 3),
            dtype=base_quat.dtype,
            device=env.device,
            generator=generator,
        )
        axis /= torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
        angle = torch.empty(
            num_episodes,
            dtype=base_quat.dtype,
            device=env.device,
        ).uniform_(
            math.radians(min_deg),
            math.radians(max_deg),
            generator=generator,
        )
        sampled[sensor_name] = quat_mul(base_quat, quat_from_angle_axis(angle, axis))
    return sampled, actual_seed


def apply_episode_camera_quaternions(
    env: ManagerBasedRlEnv,
    sampled: dict[str, torch.Tensor],
    *,
    frame_start: int,
    active: int,
    episode_ends: np.ndarray,
) -> None:
    """Apply episode-stable camera rotations to the active render worlds."""
    frame_ids = np.arange(frame_start, frame_start + active, dtype=np.int64)
    episode_ids = np.searchsorted(episode_ends, frame_ids, side="right")
    if episode_ids.size and int(episode_ids[-1]) >= len(episode_ends):
        raise ValueError(
            "Frame index exceeds meta/episode_ends while randomizing cameras."
        )
    indices = torch.as_tensor(episode_ids, dtype=torch.long, device=env.device)
    for sensor_name, episode_quats in sampled.items():
        camera_id = env.scene[sensor_name].camera_idx
        env.sim.model.cam_quat[:active, camera_id] = episode_quats[indices]


def refresh_camera_state(
    env: ManagerBasedRlEnv,
    *,
    post_kinematics: Callable[[], None] | None = None,
) -> None:
    sim = env.sim
    with wp.ScopedDevice(sim.wp_device):
        mjwarp.kinematics(sim.wp_model, sim.wp_data)
        if post_kinematics is not None:
            post_kinematics()
        mjwarp.camlight(sim.wp_model, sim.wp_data)
    sim.sense()
    for sensor in env.scene.sensors.values():
        invalidate_cache = getattr(sensor, "_invalidate_cache", None)
        if invalidate_cache is not None:
            invalidate_cache()


def validate_rgb_shape(
    rgb: np.ndarray, frame_shape: tuple[int, int, int]
) -> np.ndarray:
    target_height, target_width, target_channels = frame_shape
    if rgb.ndim != 4:
        raise ValueError(f"Expected camera RGB shape (N, H, W, C), got {rgb.shape}.")
    _, height, width, channels = rgb.shape
    if (height, width, channels) == frame_shape:
        return rgb

    raise ValueError(
        f"Camera RGB shape {(height, width, channels)} does not match target frame shape "
        f"{(target_height, target_width, target_channels)}. Configure the camera sensor resolution "
        "before rendering instead of padding rendered images."
    )


def camera_rgb_from_sensor_context(
    env: ManagerBasedRlEnv,
    sensor_names: list[str],
    active: int,
    frame_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    rendered = []
    native_shapes = []
    for name in sensor_names:
        sensor = env.scene[name]
        rgb = sensor.data.rgb
        if rgb is None:
            raise RuntimeError(f"Camera sensor {name!r} did not produce RGB data.")
        rgb_np = rgb[:active].contiguous().cpu().numpy()
        if rgb_np.ndim != 4:
            raise ValueError(
                f"Camera sensor {name!r} produced RGB shape {rgb_np.shape}; "
                "expected (N, H, W, C)."
            )
        if rgb_np.shape[0] != active:
            raise ValueError(
                f"Camera sensor {name!r} produced {rgb_np.shape[0]} active frame(s), "
                f"expected {active}."
            )
        native_shapes.append(tuple(int(dim) for dim in rgb_np.shape[1:]))
        rendered.append(rgb_np)

    if frame_shape is None:
        shapes = set(native_shapes)
        if len(shapes) != 1:
            shape_by_name = ", ".join(
                f"{name}={shape}" for name, shape in zip(sensor_names, native_shapes)
            )
            raise ValueError(
                f"Camera sensors produced different RGB shapes: {shape_by_name}."
            )
        frame_shape = native_shapes[0]

    rendered = [validate_rgb_shape(rgb, frame_shape) for rgb in rendered]
    return np.stack(rendered, axis=1).astype(np.uint8, copy=False)


def episode_origin_arrays(
    source_root,
    *,
    task_id: str,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "meta" not in source_root:
        raise KeyError("Collected Zarr has no meta group.")
    meta = source_root["meta"]
    required = {"episode_ends", "episode_env_origins", "episode_prompts"}
    missing = required - set(meta.array_keys())
    if missing:
        raise KeyError(
            f"Collected Zarr is missing current episode metadata: {sorted(missing)}. "
            "Recollect it with scripts/collect.py."
        )

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    episode_env_origins = np.asarray(meta["episode_env_origins"][:], dtype=np.float32)
    if (
        episode_ends.ndim != 1
        or episode_env_origins.ndim != 2
        or episode_env_origins.shape[1] < 3
    ):
        raise ValueError(
            "Expected meta/episode_ends shape (E,) and "
            f"meta/episode_env_origins shape (E, >=3), got "
            f"{episode_ends.shape} and {episode_env_origins.shape}."
        )
    if episode_ends.shape[0] != episode_env_origins.shape[0]:
        raise ValueError(
            "meta/episode_ends and meta/episode_env_origins must have the same length, "
            f"got {episode_ends.shape[0]} and {episode_env_origins.shape[0]}."
        )
    if episode_ends.shape[0] == 0:
        raise ValueError("Collected Zarr contains no episodes.")
    if np.any(np.diff(episode_ends) <= 0) or int(episode_ends[-1]) != num_frames:
        raise ValueError(
            "meta/episode_ends must be strictly increasing and end at the frame count; "
            f"got final end {int(episode_ends[-1])} for {num_frames} frames."
        )
    if int(meta["episode_prompts"].shape[0]) != episode_ends.shape[0]:
        raise ValueError("meta/episode_prompts must contain one prompt per episode.")

    expected_goal_names = task_goal_names(task_id)
    if expected_goal_names is None:
        unexpected = {"goal_names", "episode_goals"} & set(meta.array_keys())
        if unexpected:
            raise ValueError(
                f"Fixed-goal task {task_id!r} has unexpected goal arrays: {sorted(unexpected)}."
            )
    else:
        for key in ("goal_names", "episode_goals"):
            if key not in meta:
                raise KeyError(f"Variable-goal task {task_id!r} is missing meta/{key}.")
        actual_goal_names = tuple(str(value) for value in meta["goal_names"][:])
        if actual_goal_names != expected_goal_names:
            raise ValueError(
                f"Goal fields for {task_id!r} are {actual_goal_names}, "
                f"expected {expected_goal_names}."
            )
        episode_goals = meta["episode_goals"]
        if tuple(episode_goals.shape) != (
            episode_ends.shape[0],
            len(expected_goal_names),
        ):
            raise ValueError(
                f"meta/episode_goals has shape {episode_goals.shape}, expected "
                f"{(episode_ends.shape[0], len(expected_goal_names))}."
            )
    return episode_ends, episode_env_origins[:, :3]


def _freejoint_qpos_addresses(mj_model) -> list[int]:
    return [
        int(mj_model.jnt_qposadr[joint_id])
        for joint_id in range(mj_model.njnt)
        if mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]


def apply_episode_origins_to_qpos(
    qpos: np.ndarray,
    *,
    frame_start: int,
    episode_ends: np.ndarray | None,
    episode_env_origins: np.ndarray | None,
    freejoint_qpos_addresses: list[int],
) -> np.ndarray:
    if (
        episode_ends is None
        or episode_env_origins is None
        or not freejoint_qpos_addresses
        or not np.any(episode_env_origins)
    ):
        return qpos

    frame_ids = np.arange(frame_start, frame_start + qpos.shape[0], dtype=np.int64)
    episode_ids = np.searchsorted(episode_ends, frame_ids, side="right")
    if np.any(episode_ids >= episode_env_origins.shape[0]):
        raise ValueError(
            "Frame index exceeds meta/episode_ends while applying episode origins."
        )

    origins = episode_env_origins[episode_ids].astype(qpos.dtype, copy=False)
    adjusted = np.array(qpos, copy=True)
    for qpos_adr in freejoint_qpos_addresses:
        adjusted[:, qpos_adr : qpos_adr + 3] -= origins
    return adjusted


def episode_origins_for_frames(
    *,
    frame_start: int,
    frame_count: int,
    episode_ends: np.ndarray | None,
    episode_env_origins: np.ndarray | None,
) -> np.ndarray | None:
    if (
        episode_ends is None
        or episode_env_origins is None
        or frame_count <= 0
        or not np.any(episode_env_origins)
    ):
        return None

    frame_ids = np.arange(frame_start, frame_start + frame_count, dtype=np.int64)
    episode_ids = np.searchsorted(episode_ends, frame_ids, side="right")
    if np.any(episode_ids >= episode_env_origins.shape[0]):
        raise ValueError(
            "Frame index exceeds meta/episode_ends while applying episode origins."
        )
    return episode_env_origins[episode_ids]


def apply_episode_origins_to_mocap_pos(
    mocap_pos: np.ndarray,
    *,
    frame_start: int,
    episode_ends: np.ndarray,
    episode_env_origins: np.ndarray,
) -> np.ndarray:
    origins = episode_origins_for_frames(
        frame_start=frame_start,
        frame_count=mocap_pos.shape[0],
        episode_ends=episode_ends,
        episode_env_origins=episode_env_origins,
    )
    if origins is None:
        return mocap_pos
    return np.asarray(mocap_pos) - origins[:, None, :]


def mixer_display_asset_cfg(env: ManagerBasedRlEnv) -> SceneEntityCfg:
    asset_cfg = SceneEntityCfg(
        "mixer",
        material_names="body-display",
        geom_names="body-display",
    )
    asset_cfg.resolve(env.scene)
    return asset_cfg


def mixer_ui_replay_source(source_root, task_id: str):
    if task_id != _MIXER_PANEL_TASK_ID:
        return None, None
    if "replay" in source_root and "mixer_ui_state" in source_root["replay"]:
        return source_root["replay"]["mixer_ui_state"], "replay/mixer_ui_state"
    raise KeyError(
        f"{_MIXER_PANEL_TASK_ID} rendering requires replay/mixer_ui_state "
        "to restore the mixer display texture."
    )


def apply_mixer_ui_values_for_render(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    values: np.ndarray,
) -> None:
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(
            "Mixer UI replay values must have shape (N, >=3) in "
            "[time_s, temp_c, freq_rpm] order, got "
            f"{values.shape}."
        )

    active = int(values.shape[0])
    env_ids = torch.arange(active, device=env.device, dtype=torch.long)
    ui_state = get_mixer_ui_state(env)
    if ui_state is None:
        update_mixer_display_texture(env, env_ids, asset_cfg)
        ui_state = get_mixer_ui_state(env)
    if ui_state is None:
        raise RuntimeError("Mixer UI state was not initialized for render replay.")

    tensor_values = torch.as_tensor(
        values[:, :3], device=env.device, dtype=torch.float32
    )
    for col, key in enumerate(_MIXER_UI_VALUE_KEYS):
        ui_state[key][env_ids] = tensor_values[:, col]

    # Direct replay sets qpos without stepping, so invalidate the cached display.
    ctx = update_mixer_display_texture.__globals__.get("_MIXER_UI_CACHE", {}).get(
        id(env)
    )
    if ctx is not None:
        ctx["last_tick"] = None
    update_mixer_display_texture(env, env_ids, asset_cfg)


def bunsen_fire_replay_source(source_root, task_id: str):
    if task_id != _XHAND_BUNSEN_TASK_ID:
        return None, None
    if "replay" in source_root and "bunsen_fire_state" in source_root["replay"]:
        return source_root["replay"]["bunsen_fire_state"], "replay/bunsen_fire_state"
    raise KeyError(
        f"{_XHAND_BUNSEN_TASK_ID} rendering requires replay/bunsen_fire_state "
        "to restore the render-only flame state."
    )


def apply_bunsen_fire_state_for_render(
    env: ManagerBasedRlEnv,
    values: np.ndarray,
) -> None:
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(
            "Bunsen fire replay values must have shape (N, >=2) in "
            "[enabled, base_color_mix] order, got "
            f"{values.shape}."
        )

    from src.tasks.config.xhand_bunsen_manipulate.env_cfgs import (
        BUNSEN_FIRE_BASE_COLOR_MIX_ATTR,
        BUNSEN_FIRE_ENABLED_ATTR,
    )

    active = int(values.shape[0])
    enabled = torch.as_tensor(values[:, 0], device=env.device, dtype=torch.float32)
    base_color_mix = torch.as_tensor(
        values[:, 1], device=env.device, dtype=torch.float32
    )

    fire_enabled = getattr(env, BUNSEN_FIRE_ENABLED_ATTR, None)
    if fire_enabled is None or fire_enabled.shape[0] != env.num_envs:
        fire_enabled = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
        setattr(env, BUNSEN_FIRE_ENABLED_ATTR, fire_enabled)
    fire_enabled[:active] = enabled

    fire_base_color_mix = getattr(env, BUNSEN_FIRE_BASE_COLOR_MIX_ATTR, None)
    if fire_base_color_mix is None or fire_base_color_mix.shape[0] != env.num_envs:
        fire_base_color_mix = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
        setattr(env, BUNSEN_FIRE_BASE_COLOR_MIX_ATTR, fire_base_color_mix)
    fire_base_color_mix[:active] = base_color_mix


def heating_effect_replay_source(source_root, task_id: str):
    if task_id != _WUJI_HEATING_PLATE_TASK_ID:
        return None, None
    if "replay" in source_root and "heating_effect_state" in source_root["replay"]:
        return source_root["replay"][
            "heating_effect_state"
        ], "replay/heating_effect_state"
    raise KeyError(
        f"{_WUJI_HEATING_PLATE_TASK_ID} rendering requires replay/heating_effect_state "
        "to restore the bubbles and rotor rotation."
    )


def apply_heating_effect_state_for_render(
    env: ManagerBasedRlEnv,
    values: np.ndarray,
) -> None:
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(
            "Heating effect replay values must have shape (N, >=3) in "
            "[heat_intensity, stir_momentum, stir_phase] order, got "
            f"{values.shape}."
        )

    from src.tasks.config.wuji_heating_plate_rotate.env_cfgs import (
        HEATING_EFFECT_INTENSITY_ATTR,
        HEATING_EFFECT_STIR_MOMENTUM_ATTR,
        HEATING_EFFECT_STIR_PHASE_ATTR,
    )

    active = int(values.shape[0])
    tensor_values = torch.as_tensor(
        values[:, :3], device=env.device, dtype=torch.float32
    )
    for col, attr in enumerate(
        (
            HEATING_EFFECT_INTENSITY_ATTR,
            HEATING_EFFECT_STIR_MOMENTUM_ATTR,
            HEATING_EFFECT_STIR_PHASE_ATTR,
        )
    ):
        state = getattr(env, attr, None)
        if state is None or state.shape[0] != env.num_envs:
            state = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
            setattr(env, attr, state)
        state[:active] = tensor_values[:, col]


def time_values_for_render(values: np.ndarray, active: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1)[:, 0]
    if values.shape[0] != active:
        raise ValueError(
            f"Expected {active} replay time value(s), got shape {values.shape}."
        )
    return values
