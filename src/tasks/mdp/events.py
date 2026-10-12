import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpl_image
import matplotlib.font_manager as mpl_font_manager
import matplotlib.colors as mpl_colors
import mujoco_warp as mjw
import warnings
from PIL import Image
from pathlib import Path
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity

from mjlab.envs import ManagerBasedRlEnv
from src import MESH_ROOT
from src.tasks.task_goals import requested_task_goal_values
from src.utils.quat_util import compose_pose_wxyz, quat_to_mat

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def add_noise_to_joints(
    env,
    env_ids,
    lows,
    highs,
    asset_cfg=_DEFAULT_ASSET_CFG,
    joint_action_terms=(),
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    
    # --- 1. name → joint_ids ---
    joint_ids = asset_cfg.joint_ids

    # --- 2. default state ---
    default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
    default_joint_vel = asset.data.default_joint_vel[env_ids].clone()

    joint_pos = default_joint_pos[:, joint_ids].clone()

    # --- 3. noise ---
    lows = torch.tensor(lows, device=env.device)
    highs = torch.tensor(highs, device=env.device)

    noise = torch.rand_like(joint_pos) * (highs - lows) + lows
    joint_pos = joint_pos + noise

    # --- 4. clamp ---
    joint_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])

    joint_vel = default_joint_vel[:, joint_ids]

    # --- 5. write back ---
    joint_ids_tensor = torch.as_tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
        joint_ids=joint_ids_tensor,
    )

    # --- 6. sync controller --
    if hasattr(asset, "set_joint_position_target"):
        # NOTE: env_ids and joint_ids are both tensors; advanced indexing requires broadcastable shapes.
        # We want to set all selected joints for all selected envs: (E, J).
        env_ids_grid = env_ids[:, None].expand(-1, joint_ids_tensor.numel())
        joint_ids_grid = joint_ids_tensor[None, :].expand(env_ids.shape[0], -1)
        asset.set_joint_position_target(joint_pos, env_ids=env_ids_grid, joint_ids=joint_ids_grid)

    joint_names = [asset.joint_names[joint_id] for joint_id in joint_ids]
    for term_name in joint_action_terms:
        try:
            term = env.action_manager.get_term(term_name)
        except KeyError:
            continue
        term_joint_ids = [joint_names.index(name) for name in term.target_names]
        _sync_action_term_target(
            env,
            env_ids,
            term_name,
            joint_pos[:, term_joint_ids],
        )


def _sync_action_term_target(env, env_ids, term_name: str, target: torch.Tensor) -> None:
    if not hasattr(env, "action_manager"):
        return
    try:
        term = env.action_manager.get_term(term_name)
    except KeyError:
        return

    term._processed_actions[env_ids] = target
    if hasattr(term, "_raw_actions"):
        scale = term.scale
        offset = term.offset
        if isinstance(scale, torch.Tensor):
            scale = scale[env_ids]
        if isinstance(offset, torch.Tensor):
            offset = offset[env_ids]
        term._raw_actions[env_ids] = (target - offset) / scale


def reset_joints_and_tendons_by_range(
    env,
    env_ids,
    joint_lows,
    joint_highs,
    joint_asset_cfg=_DEFAULT_ASSET_CFG,
    joint_action_terms=(),
    tendon_lows=None,
    tendon_highs=None,
    tendon_asset_cfg=_DEFAULT_ASSET_CFG,
    tendon_action_terms=(),
    forward_after_write: bool = False,
):
    """Sample absolute joint positions and tendon length targets on reset."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[joint_asset_cfg.name]
    joint_ids = joint_asset_cfg.joint_ids
    default_joint_vel = asset.data.default_joint_vel[env_ids].clone()

    joint_lows = torch.tensor(joint_lows, device=env.device)
    joint_highs = torch.tensor(joint_highs, device=env.device)
    joint_pos = torch.rand(
        (env_ids.numel(), len(joint_ids)),
        device=env.device,
        dtype=asset.data.default_joint_pos.dtype,
    ) * (joint_highs - joint_lows) + joint_lows

    joint_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
    joint_vel = default_joint_vel[:, joint_ids]
    joint_ids_tensor = torch.as_tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
        joint_ids=joint_ids_tensor,
    )

    if hasattr(asset, "set_joint_position_target"):
        env_ids_grid = env_ids[:, None].expand(-1, joint_ids_tensor.numel())
        joint_ids_grid = joint_ids_tensor[None, :].expand(env_ids.shape[0], -1)
        asset.set_joint_position_target(
            joint_pos,
            env_ids=env_ids_grid,
            joint_ids=joint_ids_grid,
        )

    joint_names = [asset.joint_names[joint_id] for joint_id in joint_ids]
    for term_name in joint_action_terms:
        try:
            term = env.action_manager.get_term(term_name)
        except KeyError:
            continue
        term_joint_ids = [joint_names.index(name) for name in term.target_names]
        _sync_action_term_target(
            env,
            env_ids,
            term_name,
            joint_pos[:, term_joint_ids],
        )

    if tendon_lows is None or tendon_highs is None:
        if forward_after_write:
            env.sim.forward()
        return

    tendon_asset: Entity = env.scene[tendon_asset_cfg.name]
    tendon_ids = torch.as_tensor(tendon_asset_cfg.tendon_ids, device=env.device)
    tendon_lows = torch.tensor(tendon_lows, device=env.device)
    tendon_highs = torch.tensor(tendon_highs, device=env.device)
    tendon_target = torch.rand(
        (env_ids.numel(), tendon_ids.numel()),
        device=env.device,
        dtype=asset.data.default_joint_pos.dtype,
    ) * (tendon_highs - tendon_lows) + tendon_lows
    env_ids_grid = env_ids[:, None].expand(-1, tendon_ids.numel())
    tendon_ids_grid = tendon_ids[None, :].expand(env_ids.shape[0], -1)
    tendon_asset.set_tendon_len_target(
        tendon_target,
        tendon_ids=tendon_ids_grid,
        env_ids=env_ids_grid,
    )
    for term_name in tendon_action_terms:
        _sync_action_term_target(env, env_ids, term_name, tendon_target)
    if forward_after_write:
        env.sim.forward()

def change_texture(env, env_ids, asset_cfg: SceneEntityCfg):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    model = env.unwrapped.sim.model
    device = model.mat_texid.device

    geom_ids = asset_cfg.geom_ids

    material_ids = model.geom_matid[0, geom_ids]
    tex_ids = model.mat_texid[0]
    valid_tex_ids = tex_ids[tex_ids >= 0].unique()
    if len(valid_tex_ids) == 0:
        print("No valid textures found")
        return
    
    for mid in material_ids:
        slots = model.mat_texid[0, mid]
        used_slots = (slots >= 0).nonzero(as_tuple=True)[0]

        if len(used_slots) == 0:
            continue

        slot = used_slots[0].item()

        # random texture
        rand_idx = torch.randint(0, len(valid_tex_ids), (1,), device=device)
        new_tex = valid_tex_ids[rand_idx].item()

        model.mat_texid[0, mid, slot] = new_tex


_MIXER_UI_CACHE: dict[int, dict] = {}
_MIXER_UI_WIDTH = 239
_MIXER_UI_HEIGHT = 65
_MAX_MIXER_DISPLAY_ENVS = 128
_MIXER_DISPLAY_TEXTURE_NAMES = ("body-display",) + tuple(
    f"body-display-env-{env_id}" for env_id in range(1, _MAX_MIXER_DISPLAY_ENVS)
)
_THERMAL_UI_ASSET_ROOT = MESH_ROOT
_THERMAL_UI_FONT_ROOT = _THERMAL_UI_ASSET_ROOT / "common" / "font"
_THERMAL_UI_ICON_ROOT = _THERMAL_UI_ASSET_ROOT / "instrument" / "thermal_mixer_eppendorf_c" / "icon"
_THERMAL_UI_ICONS: dict[str, np.ndarray] = {}
_MIXER_TEXTURE_UPDATE_WARNED = False
_THERMAL_UI_ICON_CMAP = mpl_colors.ListedColormap(
    [
        (0, 0, 0, 1),
        (1, 1, 1, 0),
    ],
    name="binary",
    N=2,
)
_THERMAL_UI_ICONS_LOADED = False
_MIXER_BUTTON_SENSOR_NAMES = (
    "button-speed-up",
    "button-speed-down",
    "button-temp-up",
    "button-temp-down",
    "button-time-up",
    "button-time-down",
)


def reset_mixer_button_target(
    env,
    env_ids,
    button_names: tuple[str, ...] = _MIXER_BUTTON_SENSOR_NAMES,
) -> None:
    """Sample one mixer button target per env for scripted button-touch policies."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    if not hasattr(env, "mixer_button_target_ids"):
        env.mixer_button_target_ids = torch.zeros(
            (env.num_envs,),
            device=env.device,
            dtype=torch.long,
        )
    env.mixer_button_target_names = tuple(button_names)
    env.mixer_button_site_names = tuple(_MIXER_BUTTON_SENSOR_NAMES)
    sampled = torch.randint(
        low=0,
        high=len(button_names),
        size=(env_ids.numel(),),
        device=env.device,
        dtype=torch.long,
    )
    env.mixer_button_target_ids[env_ids] = sampled


def reset_mixer_ui_target(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset mixer UI values and sample target time/temp/frequency values."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    ctx = _get_mixer_ui_context(env, asset_cfg)
    if ctx is not None:
        ctx["ui_state"]["time_s"][env_ids] = 60.0
        ctx["ui_state"]["temp_c"][env_ids] = 25.0
        ctx["ui_state"]["freq_rpm"][env_ids] = 900.0
        for timer in ctx["pressed_time"].values():
            timer[env_ids] = 0.0
        for timer in ctx["released_time"].values():
            timer[env_ids] = 0.0

    if not hasattr(env, "mixer_ui_target_state"):
        env.mixer_ui_target_state = torch.zeros(
            (env.num_envs, 3),
            device=env.device,
            dtype=torch.float32,
        )

    requested_target = requested_task_goal_values(
        env,
        "XHand-MixerPanel-Manipulate",
        env_ids,
    )
    if requested_target is not None:
        env.mixer_ui_target_state[env_ids] = requested_target
    else:
        # Match dexbiolab Thermal_mixer.generate_random_parameters():
        # [time_s, temp_c, freq_rpm], with non-zero deltas from [60, 25, 900].
        time_sign = torch.where(
            torch.randint(0, 2, (env_ids.numel(),), device=env.device, dtype=torch.long) == 0,
            torch.full((env_ids.numel(),), -1.0, device=env.device),
            torch.full((env_ids.numel(),), 1.0, device=env.device),
        )
        time_steps = torch.randint(
            1,
            11,
            (env_ids.numel(),),
            device=env.device,
            dtype=torch.float32,
        )
        temp_delta = torch.randint(
            1,
            16,
            (env_ids.numel(),),
            device=env.device,
            dtype=torch.float32,
        )
        temp_sign = torch.where(
            torch.randint(0, 2, (env_ids.numel(),), device=env.device, dtype=torch.long) == 0,
            torch.full((env_ids.numel(),), -1.0, device=env.device),
            torch.full((env_ids.numel(),), 1.0, device=env.device),
        )
        freq_delta = torch.randint(
            1,
            13,
            (env_ids.numel(),),
            device=env.device,
            dtype=torch.float32,
        )
        freq_sign = torch.where(
            torch.randint(0, 2, (env_ids.numel(),), device=env.device, dtype=torch.long) == 0,
            torch.full((env_ids.numel(),), -1.0, device=env.device),
            torch.full((env_ids.numel(),), 1.0, device=env.device),
        )

        target_time = 60.0 + time_sign * time_steps * 5.0
        target_temp = 25.0 + temp_sign * temp_delta
        target_freq = 900.0 + freq_sign * freq_delta * 50.0
        env.mixer_ui_target_state[env_ids] = torch.stack(
            (target_time, target_temp, target_freq),
            dim=-1,
        )


def get_mixer_ui_state(env) -> dict[str, torch.Tensor] | None:
    ctx = _MIXER_UI_CACHE.get(id(env))
    if ctx is None:
        return None
    return ctx["ui_state"]


def _ensure_thermal_ui_assets_loaded() -> None:
    global _THERMAL_UI_ICONS_LOADED
    if _THERMAL_UI_ICONS_LOADED:
        return

    for font_name in ("NotoSans-Bold.ttf", "NotoSansMath-Regular.ttf"):
        font_path = _THERMAL_UI_FONT_ROOT / font_name
        if font_path.exists():
            mpl_font_manager.fontManager.addfont(font_path)

    icon_paths = {
        "running": "running.png",
        "paused": "paused.png",
        "interval_mix_on": "interval_mix.png",
        "speaker_on": "speaker_on.png",
        "speaker_off": "speaker_off.png",
        "key_lock_on": "lock.png",
        "key_lock_off": "unlock.png",
        "time_control": "time_control.png",
        "temp_control": "temp_control.png",
    }
    for key, rel_path in icon_paths.items():
        icon_path = _THERMAL_UI_ICON_ROOT / rel_path
        if icon_path.exists():
            _THERMAL_UI_ICONS[key] = mpl_image.imread(icon_path)

    _THERMAL_UI_ICONS_LOADED = True


def _mujoco_texture_id(model, asset_name: str, texture_name: str) -> int:
    names = (texture_name, f"{asset_name}/{texture_name}")
    for name in names:
        try:
            return int(model.texture(name).id)
        except KeyError:
            continue
    raise RuntimeError(f"Missing mixer display texture. Tried: {names}.")


def _mujoco_sensor_info(model, asset_name: str, sensor_name: str) -> tuple[int, int]:
    names = (sensor_name, f"{asset_name}/{sensor_name}")
    for name in names:
        try:
            sensor = model.sensor(name)
            return int(sensor.adr[0]), int(sensor.dim[0])
        except KeyError:
            continue
    raise RuntimeError(f"Missing mixer sensor. Tried: {names}.")


def _to_scalar_time(value) -> float:
    """Convert MJLab time container (TorchArray/Tensor/ndarray/scalar) to float."""
    if hasattr(value, "item"):
        if getattr(value, "numel", lambda: 1)() == 1:
            return float(value.item())
        return float(value.reshape(-1)[0].item())
    if hasattr(value, "numpy"):
        arr = value.numpy()
        return float(arr.reshape(-1)[0])
    if hasattr(value, "__array__"):
        arr = np.asarray(value)
        return float(arr.reshape(-1)[0])
    return float(value)


def _update_mixer_warp_textures(sim, rc, ctx, images: list[np.ndarray]) -> None:
    global _MIXER_TEXTURE_UPDATE_WARNED

    if hasattr(mjw, "update_batched_warp_texture"):
        mjw.update_batched_warp_texture(
            sim.mj_model,
            rc,
            sim.model.mat_texid,
            images,
            mat_id=ctx["material_id"],
            tex_role=1,
        )
        return

    try:
        for tex_id, image in zip(ctx["texture_ids"], images, strict=True):
            mjw.update_warp_texture(sim.mj_model, rc, int(tex_id), image)
    except RuntimeError as exc:
        if "CUDA texture" not in str(exc):
            raise
        if not _MIXER_TEXTURE_UPDATE_WARNED:
            warnings.warn(
                "Skipping mixer display texture updates because this Warp runtime "
                "does not support CPU texture copies. Mixer UI state still updates; "
                "only the rendered display texture may remain static.",
                RuntimeWarning,
                stacklevel=2,
            )
            _MIXER_TEXTURE_UPDATE_WARNED = True


def _get_mixer_ui_context(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
    key = id(env)
    cached = _MIXER_UI_CACHE.get(key)
    if cached is not None:
        return cached

    sim = env.unwrapped.sim

    # Resolve texture target from the display geom material.
    geom_ids = asset_cfg.geom_ids
    if geom_ids is None or len(geom_ids) == 0:
        raise ValueError("asset_cfg.geom_ids is required for mixer display update.")
    geom_id = int(geom_ids[0])
    material_id = int(sim.model.geom_matid[0, geom_id].item())
    texture_ids = []
    for name in _MIXER_DISPLAY_TEXTURE_NAMES:
        texture_ids.append(_mujoco_texture_id(sim.mj_model, asset_cfg.name, name))

    if env.num_envs > len(texture_ids):
        raise RuntimeError(
            f"Mixer display has {len(texture_ids)} textures but env has {env.num_envs} envs."
        )

    texture_ids = texture_ids[: env.num_envs]
    for env_id, tex_id in enumerate(texture_ids):
        sim.model.mat_texid[env_id, material_id, 1] = tex_id

    figs = []
    axes = []
    for _ in range(env.num_envs):
        _ensure_thermal_ui_assets_loaded()
        fig = plt.figure(figsize=(_MIXER_UI_WIDTH / 72, _MIXER_UI_HEIGHT / 72), dpi=72, frameon=False)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        plt.close(fig)
        figs.append(fig)
        axes.append(ax)

    ctx = {
        "material_id": material_id,
        "texture_ids": texture_ids,
        "figs": figs,
        "axes": axes,
        "last_tick": None,
        "tex_width": int(sim.mj_model.tex_width[texture_ids[0]]),
        "tex_height": int(sim.mj_model.tex_height[texture_ids[0]]),
        "sensor_info": {
            name: _mujoco_sensor_info(sim.mj_model, asset_cfg.name, name)
            for name in _MIXER_BUTTON_SENSOR_NAMES
        },
        "ui_state": {
            "time_s": torch.full((env.num_envs,), 60.0, device=env.device),
            "temp_c": torch.full((env.num_envs,), 25.0, device=env.device),
            "freq_rpm": torch.full((env.num_envs,), 900.0, device=env.device),
        },
        "pressed_time": {
            name: torch.zeros((env.num_envs,), device=env.device)
            for name in _MIXER_BUTTON_SENSOR_NAMES
        },
        "released_time": {
            name: torch.zeros((env.num_envs,), device=env.device)
            for name in _MIXER_BUTTON_SENSOR_NAMES
        },
    }
    _MIXER_UI_CACHE[key] = ctx
    return ctx


def _time_step_size(seconds: torch.Tensor) -> torch.Tensor:
    return torch.where(
        seconds < 5 * 60,
        torch.full_like(seconds, 5.0),
        torch.where(
            seconds < 20 * 60,
            torch.full_like(seconds, 15.0),
            torch.where(
                seconds < 60 * 60,
                torch.full_like(seconds, 60.0),
                torch.where(
                    seconds < 10 * 60 * 60,
                    torch.full_like(seconds, 300.0),
                    torch.full_like(seconds, 1800.0),
                ),
            ),
        ),
    )


def _render_mixer_display(fig, ax, env_id: int, time_s: float, temp_c: float, freq_rpm: float) -> np.ndarray:
    del env_id  # Rendering style is identical across envs; only values differ.

    width, height = _MIXER_UI_WIDTH, _MIXER_UI_HEIGHT
    status_h = 15
    status_y = height - status_h
    icon_spacing = status_h + 1

    # Match dexbiolab thermal_mixer_ui: white background, black text, same layout/icons.
    ax.clear()
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Top border line and icon separators.
    ax.plot([0, width], [status_y, status_y], color="black", linewidth=1)
    cur_x = width

    def _add_icon(icon_key: str | None):
        nonlocal cur_x
        if icon_key and icon_key in _THERMAL_UI_ICONS:
            icon = _THERMAL_UI_ICONS[icon_key]
            ih, iw = icon.shape[:2]
            cy = status_y + status_h / 2
            cx = cur_x - icon_spacing / 2
            extent = (cx - iw / 2, cx + iw / 2, cy - ih / 2, cy + ih / 2)
            ax.imshow(icon, extent=extent, cmap=_THERMAL_UI_ICON_CMAP, vmin=0, vmax=1)
        cur_x -= icon_spacing
        ax.plot([cur_x, cur_x], [status_y, status_y + status_h], color="black", linewidth=1)

    # Same status icon order as dexbiolab UI.
    _add_icon("temp_control")
    _add_icon("key_lock_off")
    _add_icon("speaker_on")
    _add_icon(None)   # interval_mix off
    _add_icon(None)   # thermotop unsupported
    _add_icon(None)   # device status none

    def _draw_text(x, y, text, fontsize, ha="center", fontfamily="Noto Sans", h=None):
        if h is None:
            h = fontsize
        ax.text(
            x,
            y + h / 2,
            text,
            ha=ha,
            va="center",
            fontsize=fontsize,
            color="black",
            fontfamily=fontfamily,
            weight="bold",
            antialiased=True,
        )

    # Main parameter row (same layout constants as dexbiolab/thermal_mixer_ui.py).
    time_x, temp_x, freq_x = width * 0.15, width * 0.5, width * 0.85
    param_unit_y = 3
    param_value_y = param_unit_y + 8 + 3

    total_seconds = max(0, int(round(time_s)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        time_text = f"{hours}:{minutes:02d}"
        time_unit = "h : min"
    else:
        time_text = f"{minutes}:{seconds:02d}"
        time_unit = "min : s"
    _draw_text(time_x, param_value_y, time_text, 18)
    _draw_text(time_x, param_unit_y, time_unit, 10)

    set_temp = max(0, int(round(temp_c)))
    actual_temp = set_temp
    _draw_text(temp_x, param_value_y, f"{set_temp} / {actual_temp}", 18)
    _draw_text(temp_x, param_unit_y, "°C", 10)

    rpm = max(0, int(round(freq_rpm)))
    _draw_text(freq_x, param_value_y, f"{rpm}", 18)
    _draw_text(freq_x, param_unit_y, "rpm", 10)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    # Keep rendered RGB directly and force opaque alpha to avoid transparency artifacts.
    rgba[..., 3] = 255
    return rgba


def update_mixer_ui_state(env, env_ids, asset_cfg: SceneEntityCfg):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    ctx = _get_mixer_ui_context(env, asset_cfg)
    if ctx is None:
        return

    sim = env.unwrapped.sim
    dt = float(sim.mj_model.opt.timestep)

    time_s = ctx["ui_state"]["time_s"]
    temp_c = ctx["ui_state"]["temp_c"]
    freq_rpm = ctx["ui_state"]["freq_rpm"]
    pressed_time = ctx["pressed_time"]
    released_time = ctx["released_time"]
    sensor_info = ctx["sensor_info"]
    dropout_tolerance = 0.02

    def _sensor_values(sensor_name: str) -> torch.Tensor:
        try:
            sensor = env.scene[f"mixer/{sensor_name}"]
            values = sensor.data
            if values.ndim == 1:
                values = values.unsqueeze(-1)
            return values
        except (AssertionError, KeyError, AttributeError):
            adr, dim = sensor_info[sensor_name]
            return sim.data.sensordata[:, adr : adr + dim]

    def _apply_button(sensor_name: str, delta_getter):
        values = _sensor_values(sensor_name)
        raw_pressed = values[:, 0] > 0.5
        idx = env_ids
        raw_pressed_sel = raw_pressed[idx]
        timer_sel = pressed_time[sensor_name][idx]
        release_sel = released_time[sensor_name][idx]

        release_sel = torch.where(
            raw_pressed_sel,
            torch.zeros_like(release_sel),
            release_sel + dt,
        )
        pressed_sel = raw_pressed_sel | (
            (timer_sel > 0.0) & (release_sel <= dropout_tolerance)
        )
        timer_sel = torch.where(pressed_sel, timer_sel + dt, torch.zeros_like(timer_sel))
        fire = timer_sel >= 0.03
        delta = delta_getter(idx)
        time_s[idx] = time_s[idx] + torch.where(fire, delta["time"], torch.zeros_like(delta["time"]))
        temp_c[idx] = temp_c[idx] + torch.where(fire, delta["temp"], torch.zeros_like(delta["temp"]))
        freq_rpm[idx] = freq_rpm[idx] + torch.where(fire, delta["freq"], torch.zeros_like(delta["freq"]))
        timer_sel = torch.where(fire, torch.zeros_like(timer_sel), timer_sel)
        pressed_time[sensor_name][idx] = timer_sel
        released_time[sensor_name][idx] = torch.where(
            pressed_sel,
            release_sel,
            torch.zeros_like(release_sel),
        )

    zero = torch.zeros((env_ids.numel(),), device=env.device)

    _apply_button(
        "button-speed-up",
        lambda idx: {"time": zero, "temp": zero, "freq": torch.full_like(zero, 50.0)},
    )
    _apply_button(
        "button-speed-down",
        lambda idx: {"time": zero, "temp": zero, "freq": torch.full_like(zero, -50.0)},
    )
    _apply_button(
        "button-temp-up",
        lambda idx: {"time": zero, "temp": torch.full_like(zero, 1.0), "freq": zero},
    )
    _apply_button(
        "button-temp-down",
        lambda idx: {"time": zero, "temp": torch.full_like(zero, -1.0), "freq": zero},
    )
    _apply_button(
        "button-time-up",
        lambda idx: {
            "time": _time_step_size(time_s[idx]),
            "temp": zero,
            "freq": zero,
        },
    )
    _apply_button(
        "button-time-down",
        lambda idx: {
            "time": -_time_step_size(time_s[idx]),
            "temp": zero,
            "freq": zero,
        },
    )

    # Keep within reasonable UI bounds.
    time_s[env_ids] = torch.clamp(time_s[env_ids], min=0.0, max=360000.0)
    temp_c[env_ids] = torch.clamp(temp_c[env_ids], min=0.0, max=99.0)
    freq_rpm[env_ids] = torch.clamp(freq_rpm[env_ids], min=0.0, max=3000.0)


@requires_model_fields("mat_texid")
def update_mixer_display_texture(env, env_ids, asset_cfg: SceneEntityCfg):
    del env_ids
    sim = env.unwrapped.sim
    if sim._sensor_context is None:
        return

    ctx = _get_mixer_ui_context(env, asset_cfg)
    if ctx is None:
        return

    sim_time_s = _to_scalar_time(sim.data.time)
    tick = int(sim_time_s * 10.0)
    if ctx["last_tick"] == tick:
        return
    ctx["last_tick"] = tick

    ui_state = ctx["ui_state"]
    images = []
    for env_id, (fig, ax) in enumerate(zip(ctx["figs"], ctx["axes"], strict=True)):
        image = _render_mixer_display(
            fig,
            ax,
            env_id=env_id,
            time_s=float(ui_state["time_s"][env_id].item()),
            temp_c=float(ui_state["temp_c"][env_id].item()),
            freq_rpm=float(ui_state["freq_rpm"][env_id].item()),
        )
        if image.shape[1] != ctx["tex_width"] or image.shape[0] != ctx["tex_height"]:
            image = np.asarray(
                Image.fromarray(image).resize(
                    (ctx["tex_width"], ctx["tex_height"]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
        images.append(image)

    rc = sim._sensor_context.render_context
    _update_mixer_warp_textures(sim, rc, ctx, images)

def reset_centrifuge_on_rack(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    site_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    cap_asset_cfg: SceneEntityCfg | None = None,
    cap_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.11965),
    cap_local_rot: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
    rows = 5,
    cols = 2,
    row_gap = 0.036,
    col_gap = 0.036,
    height = 0.00,
    row_idx: int | None = None,
    col_idx: int | None = None,
) -> None:
    # -------- env_ids handling --------
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    asset: Entity = env.scene[asset_cfg.name]
    site_asset: Entity = env.scene[site_asset_cfg.name]

    env.sim.forward()

    # -------- get site id --------
    site_ids = site_asset_cfg.site_ids
    assert site_ids is not None, "Please specify site_names in SceneEntityCfg"

    # -------- rack origin (world frame) --------
    site_pose_w = site_asset.data.site_pose_w[env_ids][:, site_ids]  # (N, S, 7)
    site_pos = site_pose_w[..., :3].squeeze(1)                       # (N, 3)
    site_quat = site_pose_w[..., 3:].squeeze(1)                      # (N, 4)
    site_mat = quat_to_mat(site_quat)                               # (N, 3, 3)
    row_dir = site_mat[:, :, 1]              # (N,3)
    col_dir = site_mat[:, :, 0]              # (N,3)
    # -------- grid params --------

    # -------- sample row/col --------
    if row_idx is None:
        row_idx_tensor = torch.randint(0, rows, (len(env_ids),), device=env.device)
    else:
        row_idx_tensor = torch.full((len(env_ids),), int(row_idx), device=env.device)
    if col_idx is None:
        col_idx_tensor = torch.randint(0, cols, (len(env_ids),), device=env.device)
    else:
        col_idx_tensor = torch.full((len(env_ids),), int(col_idx), device=env.device)

    row_offset = (row_idx_tensor.to(site_pos.dtype) * row_gap).unsqueeze(-1) * row_dir
    col_offset = (col_idx_tensor.to(site_pos.dtype) * col_gap).unsqueeze(-1) * col_dir
    final_pos = site_pos + row_offset + col_offset

    # height 
    final_pos[:, 2] += height #+0.04

    current_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        device=env.device,
        dtype=asset.data.root_link_pose_w.dtype,
    ).unsqueeze(0).expand(len(env_ids), -1)

    # -------- zero velocity --------
    velocities = torch.zeros_like(asset.data.root_link_vel_w[env_ids])

    asset.write_root_link_pose_to_sim(
        torch.cat([final_pos, current_quat], dim=-1),
        env_ids=env_ids,
    )

    asset.write_root_link_velocity_to_sim(
        velocities,
        env_ids=env_ids,
    )

    if cap_asset_cfg is None:
        return

    cap_asset: Entity = env.scene[cap_asset_cfg.name]
    body_pose = torch.cat([final_pos, current_quat], dim=-1)
    cap_pose_offset = torch.tensor(
        (*cap_local_pos, *cap_local_rot),
        device=env.device,
        dtype=asset.data.root_link_pose_w.dtype,
    ).unsqueeze(0).expand(len(env_ids), -1)
    cap_pose = compose_pose_wxyz(body_pose, cap_pose_offset)
    cap_velocities = torch.zeros_like(cap_asset.data.root_link_vel_w[env_ids])

    cap_asset.write_root_link_pose_to_sim(
        cap_pose,
        env_ids=env_ids,
    )
    cap_asset.write_root_link_velocity_to_sim(
        cap_velocities,
        env_ids=env_ids,
    )


def reset_centrifuge_5430_tubes(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    centrifuge_cfg: SceneEntityCfg,
    rack_cfg: SceneEntityCfg,
    tube_cfg: SceneEntityCfg,
    tube_opposite_cfg: SceneEntityCfg,
    rotor_joint_cfg: SceneEntityCfg,
    rack_row: int = 0,
    rack_col: int = 0,
    rack_row_gap: float = 0.0132,
    rack_col_gap: float = 0.0162,
    rack_height: float = 0.021,
    rack_height_index: int = 0,
    randomize_rack_position: bool = False,
    rack_row_range: tuple[int, int] = (0, 0),
    rack_col_range: tuple[int, int] = (0, 0),
    rack_height_index_range: tuple[int, int] = (0, 0),
    rotor_noise_low: float = -0.1,
    rotor_noise_high: float = 0.1,
    target_slot_ids: tuple[int, ...] | None = None,
) -> None:
    """Reset 5430 tube placement to match dexbiolab/load_centrifuge_5430.py."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    centrifuge: Entity = env.scene[centrifuge_cfg.name]
    rack: Entity = env.scene[rack_cfg.name]
    tube: Entity = env.scene[tube_cfg.name]
    tube_opposite: Entity = env.scene[tube_opposite_cfg.name]

    rotor_joint_ids = rotor_joint_cfg.joint_ids
    assert rotor_joint_ids is not None, "Please specify rotor joint_names in SceneEntityCfg"

    default_joint_pos = centrifuge.data.default_joint_pos[env_ids].clone()
    default_joint_vel = centrifuge.data.default_joint_vel[env_ids].clone()
    rotor_noise = torch.empty(
        (env_ids.numel(), len(rotor_joint_ids)),
        device=env.device,
        dtype=default_joint_pos.dtype,
    ).uniform_(rotor_noise_low, rotor_noise_high)
    default_joint_pos[:, rotor_joint_ids] += rotor_noise
    centrifuge.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    env.sim.forward()

    rack_site_ids = rack_cfg.site_ids
    assert rack_site_ids is not None, "Please specify rack site_names in SceneEntityCfg"

    rack_pose_w = rack.data.site_pose_w[env_ids][:, rack_site_ids[0]]
    rack_pos = rack_pose_w[:, :3]
    rack_quat = rack_pose_w[:, 3:]
    rack_mat = quat_to_mat(rack_quat)
    row_dir = rack_mat[:, :, 1]
    col_dir = rack_mat[:, :, 0]
    if randomize_rack_position:
        rack_row = torch.randint(
            rack_row_range[0],
            rack_row_range[1] + 1,
            (env_ids.numel(), 1),
            device=env.device,
        ).to(dtype=rack_pos.dtype)
        rack_col = torch.randint(
            rack_col_range[0],
            rack_col_range[1] + 1,
            (env_ids.numel(), 1),
            device=env.device,
        ).to(dtype=rack_pos.dtype)
        rack_height_index = torch.randint(
            rack_height_index_range[0],
            rack_height_index_range[1] + 1,
            (env_ids.numel(), 1),
            device=env.device,
        ).to(dtype=rack_pos.dtype)

    rack_tube_pos = (
        rack_pos
        + rack_row * rack_row_gap * row_dir
        + rack_col * rack_col_gap * col_dir
    )
    rack_height_offset = rack_height * rack_height_index
    if isinstance(rack_height_offset, torch.Tensor):
        rack_height_offset = rack_height_offset.squeeze(-1)
    rack_tube_pos[:, 2] += rack_height_offset

    tube_quat = tube.data.default_root_state[env_ids, 3:7].clone()
    tube_pose = torch.cat((rack_tube_pos, tube_quat), dim=-1)
    zero_tube_vel = torch.zeros_like(tube.data.root_link_vel_w[env_ids])
    tube.write_root_link_pose_to_sim(tube_pose, env_ids=env_ids)
    tube.write_root_link_velocity_to_sim(zero_tube_vel, env_ids=env_ids)

    slot_site_ids = centrifuge_cfg.site_ids
    assert slot_site_ids is not None, "Please specify slot site_names in SceneEntityCfg"

    if target_slot_ids is None:
        allowed_slot_ids = torch.arange(len(slot_site_ids), device=env.device, dtype=torch.long)
    else:
        if not target_slot_ids:
            raise ValueError("target_slot_ids cannot be empty.")
        if len(set(target_slot_ids)) != len(target_slot_ids):
            raise ValueError(f"target_slot_ids contains duplicates: {target_slot_ids}.")
        if min(target_slot_ids) < 0 or max(target_slot_ids) >= len(slot_site_ids):
            raise ValueError(
                f"target_slot_ids must be in [0, {len(slot_site_ids) - 1}], "
                f"got {target_slot_ids}."
            )
        allowed_slot_ids = torch.tensor(target_slot_ids, device=env.device, dtype=torch.long)
    sampled_allowed_ids = torch.randint(
        0,
        allowed_slot_ids.numel(),
        (env_ids.numel(),),
        device=env.device,
    )
    slot_id = allowed_slot_ids[sampled_allowed_ids]
    opposite_slot_id = torch.remainder(slot_id + len(slot_site_ids) // 2, len(slot_site_ids))
    slot_poses = centrifuge.data.site_pose_w[env_ids][:, slot_site_ids]
    batch_ids = torch.arange(env_ids.numel(), device=env.device)
    target_slot_pose = slot_poses[batch_ids, slot_id]
    opposite_slot_pose = slot_poses[batch_ids, opposite_slot_id]

    rel_quat = torch.tensor(
        [1.0, 0.0, 0.0, -1.0],
        device=env.device,
        dtype=opposite_slot_pose.dtype,
    )
    rel_quat = rel_quat / torch.linalg.norm(rel_quat)
    rel_quat = rel_quat.unsqueeze(0).expand(env_ids.numel(), -1)

    distal_rel_pose = torch.cat(
        (
            torch.tensor(
                [0.0, 0.0, 0.005],
                device=env.device,
                dtype=target_slot_pose.dtype,
            ).unsqueeze(0).expand(env_ids.numel(), -1),
            rel_quat,
        ),
        dim=-1,
    )
    proximal_rel_pose = torch.cat(
        (
            torch.tensor(
                [0.0, 0.0, -0.03],
                device=env.device,
                dtype=opposite_slot_pose.dtype,
            ).unsqueeze(0).expand(env_ids.numel(), -1),
            rel_quat,
        ),
        dim=-1,
    )

    env.centrifuge_slot_id = getattr(
        env,
        "centrifuge_slot_id",
        torch.zeros((env.num_envs,), device=env.device, dtype=torch.long),
    )
    env.centrifuge_slot_id[env_ids] = slot_id
    env.centrifuge_target_tube_pose_w = getattr(
        env,
        "centrifuge_target_tube_pose_w",
        torch.zeros((env.num_envs, 7), device=env.device, dtype=target_slot_pose.dtype),
    )
    env.centrifuge_final_tube_pose_w = getattr(
        env,
        "centrifuge_final_tube_pose_w",
        torch.zeros((env.num_envs, 7), device=env.device, dtype=target_slot_pose.dtype),
    )
    env.centrifuge_target_tube_pose_w[env_ids] = compose_pose_wxyz(
        target_slot_pose,
        distal_rel_pose,
    )
    env.centrifuge_final_tube_pose_w[env_ids] = compose_pose_wxyz(
        target_slot_pose,
        proximal_rel_pose,
    )

    opposite_tube_pose = compose_pose_wxyz(opposite_slot_pose, proximal_rel_pose)
    zero_opposite_vel = torch.zeros_like(tube_opposite.data.root_link_vel_w[env_ids])
    tube_opposite.write_root_link_pose_to_sim(opposite_tube_pose, env_ids=env_ids)
    tube_opposite.write_root_link_velocity_to_sim(zero_opposite_vel, env_ids=env_ids)
    env.sim.forward()
