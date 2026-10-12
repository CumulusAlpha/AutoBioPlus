from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import DifferentialIKActionCfg, JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mujoco_warp._src.render_util import pack_rgba_to_uint32

from assets.mjcf.misc.big_table import get_table_cfg
from assets.mjcf.robot import (
    UR5E_JOINT_ORDER,
    XHAND_JOINT_ORDER,
    get_ur5e_xhand_cfg,
)
from assets.mjcf.robot.ur5e_xhand import get_spec as get_ur5e_xhand_spec
from src import MJCF_ROOT
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.events import add_noise_to_joints
from src.tasks.mdp.observations import body_pose_w, camera_rgb, joint_pos_abs
from src.tasks.success_diagnostics import (
    record_success_diagnostics,
    stable_success,
)

BUNSEN_XML = MJCF_ROOT / "object" / "bunsen.xml"
assert BUNSEN_XML.exists(), f"Missing MJCF: {BUNSEN_XML}"

_GAUSSIAN_SCENE = gaussian_scene_for_task("XHand-Bunsen-Manipulate")
_TABLE_GEOM_NAMES = tuple(
    f"table_geom_{idx}" for idx, _ in enumerate(get_table_cfg().spec_fn().geoms)
)
_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES = (
    "terrain",
    *(f"table/{name}" for name in _TABLE_GEOM_NAMES),
)

FIRE_TIME_SCALE = 1.0
FIRE_PARTICLE_SIZE = 0.015
FIRE_PARTICLE_ALPHA_MIN = 0.08
FIRE_PARTICLE_ALPHA_MAX = 0.95
FIRE_BACKGROUND_COLOR = pack_rgba_to_uint32(
    0.02 * 255.0,
    0.02 * 255.0,
    0.025 * 255.0,
    255.0,
)
BUNSEN_INIT_POS = (0.28, 0.0, 0.844)
BUNSEN_INIT_ROT = (0.0, 0.0, 0.0, 1.0)
BUNSEN_SPAWN_OFFSET_LOW = (-0.07, -0.055, 0.0)
BUNSEN_SPAWN_OFFSET_HIGH = (0.07, 0.055, 0.0)
BUNSEN_INITIAL_ARM_JOINT_NOISE_LOW = (-0.08, -0.06, -0.06, -0.08, -0.06, -0.08)
BUNSEN_INITIAL_ARM_JOINT_NOISE_HIGH = (0.08, 0.06, 0.06, 0.08, 0.06, 0.08)
BUNSEN_INITIAL_XHAND_JOINT_NOISE_LOW = (-0.05,) * len(XHAND_JOINT_ORDER)
BUNSEN_INITIAL_XHAND_JOINT_NOISE_HIGH = (0.05,) * len(XHAND_JOINT_ORDER)
# Both knob interactions plus the intermediate and final retreat trajectories.
BUNSEN_POLICY_DONE_STEPS = 748
BUNSEN_ENV_STEP_S = 0.05
BUNSEN_EPISODE_LENGTH_S = 2.0 * BUNSEN_POLICY_DONE_STEPS * BUNSEN_ENV_STEP_S
FIRE_PARTICLES = 320
ENABLE_BUNSEN_FIRE_RENDER = True
BUNSEN_FIRE_ENABLED_ATTR = "bunsen_fire_enabled"
BUNSEN_FIRE_BASE_COLOR_MIX_ATTR = "bunsen_fire_base_color_mix"
BUNSEN_FIRE_IGNITION_JOINT = "Cylinder.004_link_joint"
BUNSEN_FIRE_COLOR_JOINT = "Cylinder.002_link_joint"
BUNSEN_FIRE_IGNITION_ANGLE_RAD = float(np.deg2rad(10.0))
BUNSEN_FIRE_BLUE_ANGLE_RAD = float(np.deg2rad(-50.0))
BUNSEN_SUCCESS_YELLOW_ANGLE_RAD = BUNSEN_FIRE_IGNITION_ANGLE_RAD
BUNSEN_SUCCESS_SILVER_ANGLE_RAD = BUNSEN_FIRE_BLUE_ANGLE_RAD
BUNSEN_SUCCESS_STABILITY_STEPS = 3
BUNSEN_WRIST_CAM_POS = (
    0.001211877060242846,
    0.13740598060496645,
    0.09210694247939308,
)
BUNSEN_WRIST_CAM_QUAT = (
    -0.5021520247449409,
    0.004980574214687484,
    0.6488758991201758,
    -0.5716455243120788,
)
BUNSEN_WRIST_CAM_FOVY = 58.0083322
BUNSEN_FIRE_NOZZLE_RADIUS = 0.0107
BUNSEN_FIRE_BASE_SPREAD = BUNSEN_FIRE_NOZZLE_RADIUS * 0.8
BUNSEN_FIRE_SOURCE_HEIGHT = 0.0
BUNSEN_FIRE_PARTICLE_BASE_SIZE = 0.0035


@dataclass(frozen=True)
class FireParticleParams:
    x0: tuple[float, ...]
    y0: tuple[float, ...]
    z0: tuple[float, ...]
    vel_x: tuple[float, ...]
    vel_y: tuple[float, ...]
    vel_z: tuple[float, ...]
    lifetime: tuple[float, ...]
    initial_age: tuple[float, ...]
    base_size: tuple[float, ...]


@wp.kernel
def _render_only_fire_kernel(
    time: wp.array(dtype=float),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    light_xpos: wp.array2d(dtype=wp.vec3),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    geom_ids: wp.array(dtype=int),
    enabled: wp.array(dtype=float),
    base_color_mix: wp.array(dtype=float),
    x0: wp.array(dtype=float),
    y0: wp.array(dtype=float),
    z0: wp.array(dtype=float),
    vel_x: wp.array(dtype=float),
    vel_y: wp.array(dtype=float),
    vel_z: wp.array(dtype=float),
    lifetime: wp.array(dtype=float),
    initial_age: wp.array(dtype=float),
    base_size: wp.array(dtype=float),
    bunsen_root_body_id: int,
    emitter_local_top: wp.vec3,
    fire_light_id: int,
    color_gain: float,
):
    world_id, particle_id = wp.tid()
    geom_id = geom_ids[particle_id]
    emitter_center = (
        xpos[world_id, bunsen_root_body_id]
        + xmat[world_id, bunsen_root_body_id] @ emitter_local_top
    )
    if particle_id == 0 and fire_light_id >= 0:
        light_xpos[world_id, fire_light_id] = emitter_center

    intensity = enabled[world_id]
    if intensity <= 0.0:
        # Alpha-zero geoms still write foreground depth in hybrid rendering.
        # Move disabled particles outside the scene so they cannot occlude it.
        geom_xpos[world_id, geom_id] = wp.vec3(
            emitter_center[0], emitter_center[1], emitter_center[2] - 1000.0
        )
        geom_size[world_id, geom_id] = wp.vec3(0.0001, 0.0001, 0.0001)
        geom_rgba[world_id, geom_id] = wp.vec4(1.0, 1.0, 0.0, 0.0)
        return
    if intensity > 1.0:
        intensity = 1.0
    intensity = intensity * intensity

    fire_time = time[world_id] * FIRE_TIME_SCALE

    life = lifetime[particle_id]
    progress = initial_age[particle_id] + fire_time / life
    progress = progress - wp.floor(progress)

    t = progress * life
    px = emitter_center[0] + (x0[particle_id] + vel_x[particle_id] * t) * intensity
    py = emitter_center[2] + (
        y0[particle_id] + vel_y[particle_id] * t - 0.5 * 0.981 * t * t
    ) * intensity
    pz = emitter_center[1] + (z0[particle_id] + vel_z[particle_id] * t) * intensity

    geom_xpos[world_id, geom_id] = wp.vec3(px, pz, py)

    reaction_coordinate = 1.0 - t
    if reaction_coordinate < 0.0:
        reaction_coordinate = 0.0
    radius = base_size[particle_id] * (0.45 + 1.10 * reaction_coordinate) * intensity
    max_radius = base_size[particle_id] * 1.55 * intensity
    if radius > max_radius:
        radius = max_radius
    geom_size[world_id, geom_id] = wp.vec3(radius, radius, radius)

    r = 1.0 * color_gain
    g = (1.0 - progress) * color_gain
    b = 0.0
    blue_mix = base_color_mix[world_id]
    if blue_mix < 0.0:
        blue_mix = 0.0
    if blue_mix > 1.0:
        blue_mix = 1.0
    blue_height = 0.28 + 0.42 * blue_mix
    bottom_weight = 1.0 - progress / blue_height
    if bottom_weight < 0.0:
        bottom_weight = 0.0
    if bottom_weight > 1.0:
        bottom_weight = 1.0
    tint = blue_mix * bottom_weight
    blue_radius = radius * (1.0 + 0.35 * tint)
    geom_size[world_id, geom_id] = wp.vec3(blue_radius, blue_radius, blue_radius)
    base_r = (1.0 - tint) + 0.06 * tint
    base_g = (1.0 - tint) + 0.42 * tint
    base_b = tint
    r = base_r * color_gain
    g = ((1.0 - progress) * (1.0 - tint) + base_g * tint) * color_gain
    b = base_b * color_gain
    a = (1.0 - progress) * intensity
    min_alpha = FIRE_PARTICLE_ALPHA_MIN * intensity
    if a < min_alpha:
        a = min_alpha
    if a > FIRE_PARTICLE_ALPHA_MAX:
        a = FIRE_PARTICLE_ALPHA_MAX
    geom_rgba[world_id, geom_id] = wp.vec4(r, g, b, a)


@dataclass
class RenderOnlyFireInstaller:
    particle_params: FireParticleParams
    geom_names: tuple[str, ...]
    emitter_center: tuple[float, float, float]
    bunsen_asset_name: str = "bunsen"
    light_name: str | None = None
    color_gain: float = 1.0
    enabled_attr: str = BUNSEN_FIRE_ENABLED_ATTR
    base_color_mix_attr: str = BUNSEN_FIRE_BASE_COLOR_MIX_ATTR
    default_enabled: bool = False

    def __post_init__(self) -> None:
        self._installed = False
        self._cache_device: wp.context.Devicelike | None = None
        self._geom_ids: wp.array | None = None
        self._x0: wp.array | None = None
        self._y0: wp.array | None = None
        self._z0: wp.array | None = None
        self._vel_x: wp.array | None = None
        self._vel_y: wp.array | None = None
        self._vel_z: wp.array | None = None
        self._lifetime: wp.array | None = None
        self._initial_age: wp.array | None = None
        self._base_size: wp.array | None = None
        self._bunsen_root_body_id: int | None = None
        self._emitter_local_top: wp.vec3 | None = None
        self._fire_light_id: int = -1

    def __call__(self, env, env_ids=None) -> None:
        del env_ids
        if self._installed:
            return
        if env.sim._sensor_context is None:
            # State-only collection disables sensors; keep replayable flame state
            # initialized and let offline rendering install the visual pipeline.
            self._ensure_state(env)
            return

        self._ensure_cache(env)
        self._ensure_state(env)
        env.sim._sensor_context.render_context.background_color = FIRE_BACKGROUND_COLOR

        def sense_kernel(sim) -> None:
            assert sim._sensor_context is not None
            ctx = sim._sensor_context
            rc = ctx.render_context
            rc.background_color = FIRE_BACKGROUND_COLOR
            self.launch(sim, env)
            mjwarp.refit_bvh(sim.wp_model, sim.wp_data, rc)
            if ctx.has_cameras:
                mjwarp.render(sim.wp_model, sim.wp_data, rc)
                ctx.unpack_rgb()
            for sensor in ctx.raycast_sensors:
                sensor.raycast_kernel(rc=rc)

        env.sim._sense_kernel = MethodType(sense_kernel, env.sim)
        env.sim.create_graph()
        self._installed = True

    def _resolve_geom_id(self, mj_model: mujoco.MjModel, name: str) -> int:
        available = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(mj_model.ngeom)]
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            return geom_id

        matches = [i for i, candidate in enumerate(available) if candidate and candidate.endswith(f"/{name}")]
        if len(matches) != 1:
            raise RuntimeError(f"Could not resolve geom '{name}'.")
        return matches[0]

    def _resolve_geom_ids(self, mj_model: mujoco.MjModel) -> list[int]:
        return [self._resolve_geom_id(mj_model, name) for name in self.geom_names]

    def _resolve_emitter_local_top(self) -> wp.vec3:
        model = mujoco.MjModel.from_xml_path(str(BUNSEN_XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        top_center: np.ndarray | None = None
        top_height = -np.inf
        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
            geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
            world_vertices = vertices @ geom_xmat.T + data.geom_xpos[geom_id]
            geom_top_height = float(np.max(world_vertices[:, 2]))
            if geom_top_height > top_height:
                top_vertices = world_vertices[
                    np.isclose(world_vertices[:, 2], geom_top_height, rtol=0.0, atol=1.0e-6)
                ]
                top_center = np.mean(top_vertices, axis=0)
                top_height = geom_top_height

        if top_center is None:
            raise RuntimeError("Bunsen MJCF does not contain any mesh vertices.")
        return wp.vec3(float(top_center[0]), float(top_center[1]), float(top_center[2]))

    def _resolve_light_id(self, mj_model: mujoco.MjModel) -> int:
        if self.light_name is None:
            return -1
        light_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_LIGHT, self.light_name)
        if light_id >= 0:
            return light_id

        available = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_LIGHT, i) for i in range(mj_model.nlight)]
        matches = [i for i, candidate in enumerate(available) if candidate and candidate.endswith(f"/{self.light_name}")]
        if len(matches) != 1:
            return -1
        return matches[0]

    def _ensure_cache(self, env) -> None:
        device = env.sim.wp_device
        if self._cache_device == device and self._bunsen_root_body_id is not None:
            return

        geom_ids = self._resolve_geom_ids(env.sim.mj_model)
        emitter_local_top = self._resolve_emitter_local_top()
        self._geom_ids = wp.array(geom_ids, dtype=int, device=device)
        self._x0 = wp.array(self.particle_params.x0, dtype=float, device=device)
        self._y0 = wp.array(self.particle_params.y0, dtype=float, device=device)
        self._z0 = wp.array(self.particle_params.z0, dtype=float, device=device)
        self._vel_x = wp.array(self.particle_params.vel_x, dtype=float, device=device)
        self._vel_y = wp.array(self.particle_params.vel_y, dtype=float, device=device)
        self._vel_z = wp.array(self.particle_params.vel_z, dtype=float, device=device)
        self._lifetime = wp.array(self.particle_params.lifetime, dtype=float, device=device)
        self._initial_age = wp.array(self.particle_params.initial_age, dtype=float, device=device)
        self._base_size = wp.array(self.particle_params.base_size, dtype=float, device=device)
        self._bunsen_root_body_id = int(env.scene[self.bunsen_asset_name].indexing.root_body_id)
        self._emitter_local_top = emitter_local_top
        self._fire_light_id = self._resolve_light_id(env.sim.mj_model)
        self._cache_device = device

    def _ensure_state(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        enabled = getattr(env, self.enabled_attr, None)
        if enabled is None or enabled.shape[0] != env.num_envs:
            enabled = torch.full(
                (env.num_envs,),
                float(self.default_enabled),
                device=env.device,
                dtype=torch.float32,
            )
            setattr(env, self.enabled_attr, enabled)

        base_color_mix = getattr(env, self.base_color_mix_attr, None)
        if base_color_mix is None or base_color_mix.shape[0] != env.num_envs:
            base_color_mix = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
            setattr(env, self.base_color_mix_attr, base_color_mix)

        return enabled, base_color_mix

    def launch(self, sim, env) -> None:
        assert self._geom_ids is not None
        assert self._x0 is not None
        assert self._y0 is not None
        assert self._z0 is not None
        assert self._vel_x is not None
        assert self._vel_y is not None
        assert self._vel_z is not None
        assert self._lifetime is not None
        assert self._initial_age is not None
        assert self._base_size is not None
        assert self._bunsen_root_body_id is not None
        assert self._emitter_local_top is not None

        enabled, base_color_mix = self._ensure_state(env)
        wp.launch(
            _render_only_fire_kernel,
            dim=(sim.num_envs, len(self.geom_names)),
            inputs=[
                sim.wp_data.time,
                sim.wp_data.xpos,
                sim.wp_data.xmat,
                sim.wp_data.geom_xpos,
                sim.wp_data.light_xpos,
                sim.wp_model.geom_size,
                sim.wp_model.geom_rgba,
                self._geom_ids,
                wp.from_torch(enabled, dtype=wp.float32),
                wp.from_torch(base_color_mix, dtype=wp.float32),
                self._x0,
                self._y0,
                self._z0,
                self._vel_x,
                self._vel_y,
                self._vel_z,
                self._lifetime,
                self._initial_age,
                self._base_size,
                self._bunsen_root_body_id,
                self._emitter_local_top,
                self._fire_light_id,
                self.color_gain,
            ],
            device=sim.wp_device,
        )


def quat_lookat(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    pos_v = np.asarray(pos, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)

    f = target_v - pos_v
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross(up_v, z)
    if np.linalg.norm(x) < 1e-9:
        up_v = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        x = np.cross(up_v, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.stack([x, y, z], axis=1).reshape(-1))
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def get_xhand_bunsen_robot_spec() -> mujoco.MjSpec:
    spec = get_ur5e_xhand_spec()
    wrist_cam = spec.camera("wrist_cam")
    wrist_cam.frame.pos = (0.0, 0.0, 0.0)
    wrist_cam.frame.quat = (1.0, 0.0, 0.0, 0.0)
    wrist_cam.pos = BUNSEN_WRIST_CAM_POS
    wrist_cam.quat = BUNSEN_WRIST_CAM_QUAT
    wrist_cam.fovy = BUNSEN_WRIST_CAM_FOVY
    return spec


def make_bunsen_fire_particles(particles: int, seed: int = 925) -> FireParticleParams:
    rng = np.random.default_rng(seed)
    radial = np.sqrt(rng.uniform(0.0, 1.0, particles).astype(np.float32)) * BUNSEN_FIRE_BASE_SPREAD
    theta = rng.uniform(0.0, 2.0 * np.pi, particles).astype(np.float32)
    x0 = (radial * np.cos(theta)).astype(np.float32)
    y0 = np.full(particles, BUNSEN_FIRE_SOURCE_HEIGHT, dtype=np.float32)
    z0 = (radial * np.sin(theta)).astype(np.float32)

    vel_x = ((rng.uniform(0.0, 1.0, particles).astype(np.float32) - 0.5) * 0.006).astype(np.float32)
    vel_y = (0.25 + rng.uniform(0.0, 1.0, particles).astype(np.float32) * 0.35).astype(np.float32)
    vel_z = ((rng.uniform(0.0, 1.0, particles).astype(np.float32) - 0.5) * 0.006).astype(np.float32)
    lifetime = np.maximum(0.08, rng.uniform(0.0, 1.0, particles).astype(np.float32) * 0.35)
    initial_age = rng.uniform(0.0, 1.0, particles).astype(np.float32)
    base_size = np.full(particles, BUNSEN_FIRE_PARTICLE_BASE_SIZE, dtype=np.float32)

    return FireParticleParams(
        x0=tuple(float(x) for x in x0),
        y0=tuple(float(x) for x in y0),
        z0=tuple(float(x) for x in z0),
        vel_x=tuple(float(x) for x in vel_x),
        vel_y=tuple(float(x) for x in vel_y),
        vel_z=tuple(float(x) for x in vel_z),
        lifetime=tuple(float(x) for x in lifetime),
        initial_age=tuple(float(x) for x in initial_age),
        base_size=tuple(float(x) for x in base_size),
    )


def make_fire_spec(
    base_size: tuple[float, ...],
    emitter_center: tuple[float, float, float],
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()

    for i, radius in enumerate(base_size):
        body = spec.worldbody.add_body(name=f"bunsen_fire_particle_{i:03d}", pos=emitter_center)
        body.add_geom(
            name=f"bunsen_fire_particle_{i:03d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(float(radius), float(radius), float(radius)),
            rgba=(1.0, 1.0, 0.0, 0.0),
            contype=0,
            conaffinity=0,
        )

    return spec


def get_or_create_bunsen_fire_enabled(env) -> torch.Tensor:
    enabled = getattr(env, BUNSEN_FIRE_ENABLED_ATTR, None)
    if enabled is None or enabled.shape[0] != env.num_envs:
        enabled = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        setattr(env, BUNSEN_FIRE_ENABLED_ATTR, enabled)
    return enabled


def get_or_create_bunsen_fire_base_color_mix(env) -> torch.Tensor:
    mix = getattr(env, BUNSEN_FIRE_BASE_COLOR_MIX_ATTR, None)
    if mix is None or mix.shape[0] != env.num_envs:
        mix = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        setattr(env, BUNSEN_FIRE_BASE_COLOR_MIX_ATTR, mix)
    return mix


def reset_bunsen_fire_enabled(
    env,
    env_ids,
) -> None:
    enabled = get_or_create_bunsen_fire_enabled(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    enabled[env_ids] = 0.0


def reset_bunsen_fire_base_color_mix(
    env,
    env_ids,
) -> None:
    mix = get_or_create_bunsen_fire_base_color_mix(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    mix[env_ids] = 0.0


def update_bunsen_fire_intensity_from_yellow_knob(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    target_angle_rad: float = BUNSEN_SUCCESS_YELLOW_ANGLE_RAD,
) -> None:
    enabled = get_or_create_bunsen_fire_enabled(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    bunsen = env.scene[asset_cfg.name]
    joint_id = int(asset_cfg.joint_ids[0])
    current_angle = bunsen.data.joint_pos[env_ids, joint_id]
    progress = current_angle / float(target_angle_rad)
    intensity = torch.minimum(progress, torch.ones_like(progress))
    enabled[env_ids] = torch.clamp_min(intensity, 0.0)


def update_bunsen_fire_base_color_from_silver_knob(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    blue_angle_rad: float = BUNSEN_FIRE_BLUE_ANGLE_RAD,
) -> None:
    mix = get_or_create_bunsen_fire_base_color_mix(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    bunsen = env.scene[asset_cfg.name]
    joint_id = int(asset_cfg.joint_ids[0])
    knob_angle = bunsen.data.joint_pos[env_ids, joint_id]
    progress = knob_angle / float(blue_angle_rad)
    mix[env_ids] = torch.clamp(progress, 0.0, 1.0)


def get_bunsen_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(BUNSEN_XML))
    return spec


def get_bunsen_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=BUNSEN_INIT_POS,
            rot=BUNSEN_INIT_ROT,
            joint_pos={
                "Cylinder.002_link_joint": 0.0,
                "Cylinder.004_link_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_bunsen_spec,
    )


def get_gaussian_hidden_table_cfg():
    table_cfg = get_table_cfg()
    base_spec_fn = table_cfg.spec_fn

    def table_spec_with_geom_names():
        spec = base_spec_fn()
        for geom, name in zip(spec.geoms, _TABLE_GEOM_NAMES):
            geom.name = name
        return spec

    table_cfg.spec_fn = table_spec_with_geom_names
    return table_cfg


def randomize_bunsen_spawn(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    pos_offset_low: tuple[float, float, float] = BUNSEN_SPAWN_OFFSET_LOW,
    pos_offset_high: tuple[float, float, float] = BUNSEN_SPAWN_OFFSET_HIGH,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    bunsen = env.scene[asset_cfg.name]
    pose = bunsen.data.default_root_state[env_ids, :7].clone()
    dtype = pose.dtype
    low = torch.tensor(pos_offset_low, device=env.device, dtype=dtype)
    high = torch.tensor(pos_offset_high, device=env.device, dtype=dtype)
    pos_offset = torch.rand((env_ids.numel(), 3), device=env.device, dtype=dtype) * (high - low) + low
    env_origins = env.scene.env_origins[env_ids].to(device=env.device, dtype=dtype)

    pose[:, :3] = pose[:, :3] + pos_offset + env_origins
    if not bunsen.is_mocap:
        raise RuntimeError("Bunsen spawn randomization requires the fixed-base Bunsen entity to be mocap-wrapped.")
    bunsen.write_mocap_pose_to_sim(pose, env_ids=env_ids)
    env.sim.forward()


def reset_bunsen_success_state(env, env_ids) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    complete = getattr(env, "xhand_bunsen_manipulate_policy_complete", None)
    if complete is None or complete.shape[0] != env.num_envs:
        complete = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    complete[env_ids] = False
    env.xhand_bunsen_manipulate_policy_complete = complete

    yellow_open_seen = getattr(env, "xhand_bunsen_yellow_open_seen", None)
    if yellow_open_seen is None or yellow_open_seen.shape[0] != env.num_envs:
        yellow_open_seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    yellow_open_seen[env_ids] = False
    env.xhand_bunsen_yellow_open_seen = yellow_open_seen

    silver_blue_seen = getattr(env, "xhand_bunsen_silver_blue_seen", None)
    if silver_blue_seen is None or silver_blue_seen.shape[0] != env.num_envs:
        silver_blue_seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    silver_blue_seen[env_ids] = False
    env.xhand_bunsen_silver_blue_seen = silver_blue_seen

    seen = getattr(env, "xhand_bunsen_success_seen", None)
    if seen is None or seen.shape[0] != env.num_envs:
        seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    seen[env_ids] = False
    env.xhand_bunsen_success_seen = seen


def resolve_entity_joint_id(entity, joint_name: str) -> int:
    joint_ids, resolved_joint_names = entity.find_joints([joint_name])
    ids_list = joint_ids.tolist() if hasattr(joint_ids, "tolist") else list(joint_ids)
    names_list = list(resolved_joint_names) if resolved_joint_names is not None else []
    for joint_id, resolved_name in zip(ids_list, names_list, strict=False):
        if resolved_name == joint_name or str(resolved_name).endswith(joint_name):
            return int(joint_id)
    if len(ids_list) == 1:
        return int(ids_list[0])

    joint_names = getattr(entity, "joint_names", None)
    if joint_names is not None:
        for joint_id, resolved_name in enumerate(joint_names):
            if resolved_name == joint_name or str(resolved_name).endswith(joint_name):
                return joint_id

    raise RuntimeError(f"Could not resolve joint {joint_name!r} on entity.")


def check_success(
    env,
    asset_cfg: SceneEntityCfg,
    min_yellow_angle: float,
    max_silver_angle: float,
    stability_steps: int,
    use_ring_history: bool = False,
) -> torch.Tensor:
    bunsen = env.scene[asset_cfg.name]
    joint_names = asset_cfg.joint_names
    assert joint_names is not None and len(joint_names) >= 2, "Please specify yellow and silver knob joint_names."

    joint_pos = bunsen.data.joint_pos
    yellow_joint_id = resolve_entity_joint_id(bunsen, str(joint_names[0]))
    silver_joint_id = resolve_entity_joint_id(bunsen, str(joint_names[1]))
    yellow_angle = joint_pos[:, yellow_joint_id]
    silver_angle = joint_pos[:, silver_joint_id]

    yellow_open_seen = getattr(env, "xhand_bunsen_yellow_open_seen", None)
    if yellow_open_seen is None or yellow_open_seen.shape[0] != env.num_envs:
        yellow_open_seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    yellow_open_seen |= yellow_angle >= float(min_yellow_angle)
    env.xhand_bunsen_yellow_open_seen = yellow_open_seen

    silver_blue_seen = getattr(env, "xhand_bunsen_silver_blue_seen", None)
    if silver_blue_seen is None or silver_blue_seen.shape[0] != env.num_envs:
        silver_blue_seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    silver_blue_seen |= silver_angle <= float(max_silver_angle)
    env.xhand_bunsen_silver_blue_seen = silver_blue_seen

    yellow_open = yellow_angle >= float(min_yellow_angle)
    silver_blue = silver_angle <= float(max_silver_angle)
    object_complete = (
        yellow_open_seen & silver_blue_seen
        if use_ring_history
        else yellow_open & silver_blue
    )
    require_policy_complete = bool(
        getattr(env, "xhand_bunsen_require_policy_complete", False)
    )
    policy_complete = getattr(
        env,
        "xhand_bunsen_manipulate_policy_complete",
        torch.zeros_like(object_complete),
    )
    success_condition = (
        object_complete & policy_complete
        if require_policy_complete
        else object_complete
    )
    success, stable_count = stable_success(
        env,
        success_condition,
        state_attr="xhand_bunsen_stable_steps",
        required_steps=stability_steps,
    )

    record_success_diagnostics(
        env,
        object_complete=object_complete,
        full_success=success,
        final_motion_complete=policy_complete if require_policy_complete else None,
        components={
            "yellow_open_now": yellow_open,
            "silver_blue_now": silver_blue,
            "yellow_open_seen": yellow_open_seen,
            "silver_blue_seen": silver_blue_seen,
        },
        metrics={
            "yellow_angle_rad": yellow_angle,
            "silver_angle_rad": silver_angle,
            "stable_steps": stable_count,
        },
        thresholds={
            "min_yellow_angle_rad": min_yellow_angle,
            "max_silver_angle_rad": max_silver_angle,
            "required_stability_steps": stability_steps,
        },
    )
    seen = getattr(env, "xhand_bunsen_success_seen", None)
    if seen is None or seen.shape[0] != env.num_envs:
        seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    seen |= success
    env.xhand_bunsen_success_seen = seen
    return success


def make_xhand_bunsen_manipulate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg: # the play now uses nothing
    # --------------------------
    # Scene
    # --------------------------

    robot_cfg = get_ur5e_xhand_cfg()
    robot_cfg.spec_fn = get_xhand_bunsen_robot_spec
    robot_cfg.init_state.pos = (0, 0.312, 0.824)
    robot_cfg.init_state.rot = (0.0, 0.0, 0.0, -1.0)
    robot_cfg.init_state.joint_pos.update(
        {
            "shoulder_pan": 3.1415,
            "shoulder_lift": -1.5707,
            "elbow": -1.5707,
            "wrist_1": -3.1415,
            "wrist_2": -1.5707,
            "wrist_3": -0.628,
        }
    )

    bunsen_cfg = get_bunsen_cfg()
    bunsen_cfg.init_state.pos = (0, -0.4, 0.844)
    bunsen_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)

    fire_emitter_center = (
        bunsen_cfg.init_state.pos[0],
        bunsen_cfg.init_state.pos[1],
        bunsen_cfg.init_state.pos[2],
    )
    gs_front_cam_pos = (0.0, -1.2, 1.45)
    gs_left_cam_pos = (-1.2, -0.4, 1.35)
    gs_wrist_cam, gs_front_cam, gs_left_cam = make_gaussian_camera_triplet(
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention big table",
        scene_parent_body="table/vention big table",
        wrist_fovy=BUNSEN_WRIST_CAM_FOVY,
        front_pos=gs_front_cam_pos,
        front_quat=quat_lookat(gs_front_cam_pos, fire_emitter_center),
        left_pos=gs_left_cam_pos,
        left_quat=quat_lookat(gs_left_cam_pos, fire_emitter_center),
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )

    scene_entities = {
        "robot": robot_cfg,
        "bunsen": bunsen_cfg,
        "table": get_gaussian_hidden_table_cfg(),
    }
    events = {
        "reset_scene": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "randomize_robot_start_joints": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": BUNSEN_INITIAL_ARM_JOINT_NOISE_LOW,
                "highs": BUNSEN_INITIAL_ARM_JOINT_NOISE_HIGH,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_JOINT_ORDER,
                ),
            },
        ),
        "randomize_xhand_start_joints": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": BUNSEN_INITIAL_XHAND_JOINT_NOISE_LOW,
                "highs": BUNSEN_INITIAL_XHAND_JOINT_NOISE_HIGH,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=XHAND_JOINT_ORDER,
                ),
                "joint_action_terms": ("xhand",),
            },
        ),
        "randomize_bunsen_spawn": EventTermCfg(
            func=randomize_bunsen_spawn,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("bunsen"),
            },
        ),
        "reset_bunsen_success_state": EventTermCfg(
            func=reset_bunsen_success_state,
            mode="reset",
        ),
    }

    if ENABLE_BUNSEN_FIRE_RENDER:
        particle_params = make_bunsen_fire_particles(FIRE_PARTICLES)
        particle_geom_names = tuple(f"bunsen_fire_particle_{i:03d}" for i in range(FIRE_PARTICLES))
        scene_entities["bunsen_fire"] = EntityCfg(
            spec_fn=lambda: make_fire_spec(particle_params.base_size, fire_emitter_center),
        )
        events["install_bunsen_fire"] = EventTermCfg(
            func=RenderOnlyFireInstaller(
                particle_params=particle_params,
                geom_names=particle_geom_names,
                emitter_center=fire_emitter_center,
                color_gain=8.0,
                enabled_attr=BUNSEN_FIRE_ENABLED_ATTR,
                base_color_mix_attr=BUNSEN_FIRE_BASE_COLOR_MIX_ATTR,
                default_enabled=False,
            ),
            mode="startup",
        )
        events["reset_bunsen_fire_enabled"] = EventTermCfg(
            func=reset_bunsen_fire_enabled,
            mode="reset",
        )
        events["reset_bunsen_fire_base_color_mix"] = EventTermCfg(
            func=reset_bunsen_fire_base_color_mix,
            mode="reset",
        )
        events["update_bunsen_fire_intensity_from_yellow_knob"] = EventTermCfg(
            func=update_bunsen_fire_intensity_from_yellow_knob,
            mode="step",
            params={
                "asset_cfg": SceneEntityCfg(
                    "bunsen",
                    joint_names=[BUNSEN_FIRE_IGNITION_JOINT],
                ),
                "target_angle_rad": BUNSEN_SUCCESS_YELLOW_ANGLE_RAD,
            },
        )
        events["update_bunsen_fire_base_color_from_silver_knob"] = EventTermCfg(
            func=update_bunsen_fire_base_color_from_silver_knob,
            mode="step",
            params={
                "asset_cfg": SceneEntityCfg(
                    "bunsen",
                    joint_names=[BUNSEN_FIRE_COLOR_JOINT],
                ),
                "blue_angle_rad": BUNSEN_FIRE_BLUE_ANGLE_RAD,
            },
        )

    scene = SceneCfg(
        terrain=TerrainEntityCfg(
        terrain_type="plane",
        ),
        entities=scene_entities,
        num_envs=1,
        env_spacing=5.0,
        sensors=(gs_wrist_cam, gs_front_cam, gs_left_cam),
    )

    # --------------------------
    # Observation: only joint pos
    # --------------------------
    
    observations = {
        "current_qpos": ObservationGroupCfg(
            {
                "current_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            joint_names=UR5E_JOINT_ORDER+XHAND_JOINT_ORDER,
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "palm_body_pose_w": ObservationGroupCfg(
            {
                "palm_body_pose_w": ObservationTermCfg(
                    func=body_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            body_names=[r".*xhand:right_hand_link"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "bunsen_body_pose_w": ObservationGroupCfg(
            {
                "bunsen_body_pose_w": ObservationTermCfg(
                    func=body_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "bunsen",
                            body_names=[
                                r".*Cylinder\.002_link",
                                r".*Cylinder\.004_link",
                            ],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "gs_front_cam_rgb": ObservationGroupCfg(
            {
                "gs_front_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_front_cam_rgb"},
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "gs_wrist_cam_rgb": ObservationGroupCfg(
            {
                "gs_wrist_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_wrist_cam_rgb"},
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "gs_left_cam_rgb": ObservationGroupCfg(
            {
                "gs_left_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_left_cam_rgb"},
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
    }

    # --------------------------
    # ✅ Absolute position control
    # --------------------------
    actions = {
    "site_pose": DifferentialIKActionCfg(
        entity_name="robot",
        actuator_names=UR5E_JOINT_ORDER,
        frame_type="body",
        frame_name=r".*xhand:right_hand_link",
        use_relative_mode=False,
        position_weight=1,
        orientation_weight=0.5,
    ),
    "xhand": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=XHAND_JOINT_ORDER,
        use_default_offset=False,
        preserve_order=True,
        clip=None,
    ),
    }

    # --------------------------
    # Events
    # --------------------------
    # --------------------------
    # Termination
    # --------------------------
    terminations = {
        "time_out": TerminationTermCfg(
            func=envs_mdp.time_out,
            time_out=True,
        ),
        "success": TerminationTermCfg(
            func=check_success,
            params={
                "asset_cfg": SceneEntityCfg(
                    "bunsen",
                    joint_names=[
                        BUNSEN_FIRE_IGNITION_JOINT,
                        BUNSEN_FIRE_COLOR_JOINT,
                    ],
                ),
                "min_yellow_angle": BUNSEN_SUCCESS_YELLOW_ANGLE_RAD,
                "max_silver_angle": BUNSEN_SUCCESS_SILVER_ANGLE_RAD,
                "stability_steps": BUNSEN_SUCCESS_STABILITY_STEPS,
                "use_ring_history": False,
            },
        ),
    }

    # --------------------------
    # Simulation
    # --------------------------

    sim = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.002,
            impratio=10,
            cone="elliptic",
            ccd_iterations=50,
            multiccd=True,
        )
    )

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        events=events,
        terminations=terminations,
        sim=sim,
        decimation=25,
        episode_length_s=BUNSEN_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
