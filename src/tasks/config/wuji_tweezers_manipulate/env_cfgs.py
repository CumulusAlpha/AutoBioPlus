import warnings
from functools import partial
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
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

from assets.mjcf.misc.big_table import get_table_cfg
from assets.mjcf.robot import (
    UR5E_JOINT_ORDER,
    WUJI_JOINT_ORDER,
    get_ur5e_wuji_cfg,
)
from src import MJCF_ROOT
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.events import add_noise_to_joints, reset_centrifuge_on_rack
from src.tasks.mdp.fog import FogParticleParams, RenderOnlyFogInstaller
from src.tasks.mdp.observations import (
    body_pose_w,
    camera_rgb,
    joint_pos_abs,
    root_pose_w,
)
from src.tasks.success_diagnostics import (
    record_success_diagnostics,
    stable_success,
)
from src.utils.quat_util import (
    axis_angle_to_quat,
    compose_pose_wxyz,
    pose_inv_wxyz,
    quat_mul,
    quat_rotate_vec,
)

TWEEZERS_XML = MJCF_ROOT / "misc" / "tweezers.xml"
CRYOVIAL_XML = MJCF_ROOT / "object" / "cryovial_1-8ml.xml"
MICRO_TUBE_RACK_XML = MJCF_ROOT / "object" / "centrifuge_plate_60well.gen.xml"
BEAKER_XML = MJCF_ROOT / "object" / "glassware010.xml"
assert TWEEZERS_XML.exists(), f"Missing MJCF: {TWEEZERS_XML}"
assert CRYOVIAL_XML.exists(), f"Missing MJCF: {CRYOVIAL_XML}"
assert MICRO_TUBE_RACK_XML.exists(), f"Missing MJCF: {MICRO_TUBE_RACK_XML}"
assert BEAKER_XML.exists(), f"Missing MJCF: {BEAKER_XML}"

_GAUSSIAN_SCENE = gaussian_scene_for_task("Wuji-Tweezers-Manipulate")
_TABLE_GEOM_NAMES = tuple(
    f"table_geom_{idx}" for idx, _ in enumerate(get_table_cfg().spec_fn().geoms)
)
_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES = (
    "terrain",
    *(f"table/{name}" for name in _TABLE_GEOM_NAMES),
)

TWEEZERS_INIT_POS = (0.28, 0.0, 0.844)
TWEEZERS_INIT_ROT = (0.0, 0.0, 0.0, 1.0)
WUJI_TWEEZERS_RANDOM_POS_LOW = (-0.025, -0.025, 0.0)
WUJI_TWEEZERS_RANDOM_POS_HIGH = (0.025, 0.025, 0.0)
WUJI_TWEEZERS_RANDOM_YAW_LOW = -0.10
WUJI_TWEEZERS_RANDOM_YAW_HIGH = 0.10
TABLE_SURFACE_Z = 0.824
TUBE_RACK_HALF_HEIGHT = 0.0105
TUBE_RACK_INIT_POS = (0.0, -0.3, TABLE_SURFACE_Z + TUBE_RACK_HALF_HEIGHT)
TUBE_RACK_INIT_ROT = (0.0, 0.0, 0.0, 1.0)
CRYOVIAL_SLOT_SITE_LOCAL_POS = (-0.0873, 0.0244, -0.0105)
CRYOVIAL_SLOT_ROWS = 5
CRYOVIAL_SLOT_COLS = 12
CRYOVIAL_SLOT_ROW_GAP = 0.0132
CRYOVIAL_SLOT_COL_GAP = 0.0162
CRYOVIAL_SLOT_ROW = 4
CRYOVIAL_SLOT_COL = 8
CRYOVIAL_SLOT_HEIGHT = 0.0
BEAKER_INIT_POS = (TUBE_RACK_INIT_POS[0] - 0.23, TUBE_RACK_INIT_POS[1], 0.824)
BEAKER_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
BEAKER_CENTER_LOCAL_POS = (-0.0023, 0.0018147, 0.0491)
WUJI_INITIAL_ARM_JOINT_NOISE_LOW = (-0.12, -0.10, -0.10, -0.12, -0.10, -0.12)
WUJI_INITIAL_ARM_JOINT_NOISE_HIGH = (0.12, 0.10, 0.10, 0.12, 0.10, 0.12)
WUJI_TUBE_DENSITY = 10.0
TUBE_BODY_RACK_GEOM_PREFIX = "tube_body_rack_collision"
RACK_TUBE_BODY_GEOM_PREFIX = "rack_tube_body_collision"
TUBE_BODY_RACK_PAIR_FRICTION = (2.0, 0.02, 0.001, 0.0001, 0.0001)
CAP_TWEEZERS_GEOM_PREFIX = "cap_tweezers_collision"
CAP_TWEEZERS_PAIR_FRICTION = (3.0, 3.0, 0.01, 0.0001, 0.0001)
TUBE_TWEEZERS_PAIR_FRICTION = (40.0, 40.0, 0.02, 0.0001, 0.0001)
TUBE_TWEEZERS_PAIR_SOLREF = (0.001, 1.0)
TUBE_TWEEZERS_PAIR_SOLIMP = (0.99, 0.999, 0.00005, 0.5, 2.0)
LIQUID_NITROGEN_FOG_PARTICLES = 160
LIQUID_NITROGEN_FOG_COLOR = (1.0, 1.0, 1.0)
LIQUID_NITROGEN_FOG_ALPHA = 0.15
LIQUID_NITROGEN_FOG_RADIUS = 0.055
LIQUID_NITROGEN_FOG_HEIGHT = 0.16
LIQUID_NITROGEN_FOG_TIME_SCALE = 0.55
LIQUID_NITROGEN_FOG_SPRITE_SIGMA = 1
LIQUID_NITROGEN_FOG_SPRITE_SCALE = 1.4
ENABLE_LIQUID_NITROGEN_FOG_RENDER = True
LIQUID_NITROGEN_FOG_CENTER = (
    BEAKER_INIT_POS[0] + BEAKER_CENTER_LOCAL_POS[0],
    BEAKER_INIT_POS[1] + BEAKER_CENTER_LOCAL_POS[1],
    BEAKER_INIT_POS[2] + 0.096,
)
LIQUID_NITROGEN_FOG_CENTER_SITE_OFFSET = (
    LIQUID_NITROGEN_FOG_CENTER[0] - (BEAKER_INIT_POS[0] + BEAKER_CENTER_LOCAL_POS[0]),
    LIQUID_NITROGEN_FOG_CENTER[1] - (BEAKER_INIT_POS[1] + BEAKER_CENTER_LOCAL_POS[1]),
    LIQUID_NITROGEN_FOG_CENTER[2] - (BEAKER_INIT_POS[2] + BEAKER_CENTER_LOCAL_POS[2]),
)
# The Dexonomy pregrasp is speed-limited and varies with randomized starts.
# Fixed post-grasp phases use 349 steps; 550 covers the current pregrasp range.
WUJI_TWEEZER_POLICY_DONE_STEPS = 550
WUJI_TWEEZER_ENV_STEP_S = 0.05
WUJI_TWEEZER_EPISODE_LENGTH_S = (
    2.0 * WUJI_TWEEZER_POLICY_DONE_STEPS * WUJI_TWEEZER_ENV_STEP_S
)
# Mesh/MJCF dimensions in the beaker_center frame.  The tube freejoint origin
# is at its bottom tip and its local +Z axis runs to the top of the body.
WUJI_TUBE_AXIS_LENGTH = 0.07492
WUJI_TUBE_MAX_RADIUS = 0.0065
WUJI_BEAKER_INNER_RADIUS = 0.033
WUJI_BEAKER_INTERIOR_BOTTOM_HEIGHT = -0.0455
WUJI_BEAKER_RIM_HEIGHT = 0.0455
WUJI_TWEEZER_SUCCESS_CONTAINMENT_TOLERANCE = 0.002
WUJI_TWEEZER_SUCCESS_BOTTOM_TOLERANCE = 0.006
WUJI_TWEEZER_SUCCESS_STABILITY_STEPS = 3
WUJI_TWEEZER_SUCCESS_MAX_PALM_DIST = 0.14
WUJI_TWEEZER_MIN_HAND_CONTACT_FORCE = 0.01
WUJI_TWEEZER_DROP_NO_CONTACT_STEPS = 3
WUJI_TWEEZER_WRIST_CAM_POS = (0.047822702, -0.021633145, -0.0429238727)
WUJI_TWEEZER_WRIST_CAM_QUAT = (-0.311322426, -0.384235854, 0.854867689, -0.156978949)
WUJI_TWEEZER_WRIST_CAM_FOVY = 58.0


def cryovial_slot_init_pos() -> tuple[float, float, float]:
    return (
        TUBE_RACK_INIT_POS[0]
        + CRYOVIAL_SLOT_SITE_LOCAL_POS[0]
        + CRYOVIAL_SLOT_COL * CRYOVIAL_SLOT_COL_GAP,
        TUBE_RACK_INIT_POS[1]
        + CRYOVIAL_SLOT_SITE_LOCAL_POS[1]
        - CRYOVIAL_SLOT_ROW * CRYOVIAL_SLOT_ROW_GAP,
        TUBE_RACK_INIT_POS[2] + CRYOVIAL_SLOT_SITE_LOCAL_POS[2] + CRYOVIAL_SLOT_HEIGHT,
    )


def get_wuji_tweezers_robot_spec() -> mujoco.MjSpec:
    spec = get_ur5e_wuji_cfg().spec_fn()
    wrist_cam = spec.camera("wuji:wrist_cam")
    wrist_cam.pos = WUJI_TWEEZER_WRIST_CAM_POS
    wrist_cam.quat = WUJI_TWEEZER_WRIST_CAM_QUAT
    wrist_cam.fovy = WUJI_TWEEZER_WRIST_CAM_FOVY
    return spec


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


def make_liquid_nitrogen_fog_particles(
    particles: int,
    seed: int = 934,
) -> tuple[FogParticleParams, tuple[float, ...], tuple[float, ...]]:
    rng = np.random.default_rng(seed)
    size_class = rng.choice(3, size=particles, p=[0.56, 0.32, 0.12])

    radius_ranges = np.array(
        [
            [0.006, 0.011],
            [0.011, 0.018],
            [0.018, 0.027],
        ],
        dtype=np.float32,
    )
    radius_u = rng.uniform(0.0, 1.0, particles).astype(np.float32)
    base_radius = radius_ranges[size_class, 0] + radius_u * (
        radius_ranges[size_class, 1] - radius_ranges[size_class, 0]
    )
    alpha = np.full(particles, LIQUID_NITROGEN_FOG_ALPHA, dtype=np.float32)

    angle = rng.uniform(0.0, 2.0 * np.pi, particles).astype(np.float32)
    dist = (
        LIQUID_NITROGEN_FOG_RADIUS
        * 0.48
        * np.sqrt(rng.uniform(0.0, 1.0, particles))
    ).astype(np.float32)
    xy0 = np.column_stack((dist * np.cos(angle), dist * np.sin(angle))).astype(np.float32)

    lifetime = rng.uniform(2.8, 5.6, particles).astype(np.float32)
    initial_age = rng.uniform(0.0, lifetime).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, particles).astype(np.float32)
    drift = rng.uniform(-1.0, 1.0, (particles, 2)).astype(np.float32)

    params = FogParticleParams(
        xy0=tuple((float(x), float(y)) for x, y in xy0),
        lifetime=tuple(float(x) for x in lifetime),
        initial_age=tuple(float(x) for x in initial_age),
        phase=tuple(float(x) for x in phase),
        drift=tuple((float(x), float(y)) for x, y in drift),
    )
    return params, tuple(float(x) for x in base_radius), tuple(float(x) for x in alpha)


def make_liquid_nitrogen_fog_spec(
    base_radius: tuple[float, ...],
    alpha: tuple[float, ...],
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    for i, (radius, a) in enumerate(zip(base_radius, alpha, strict=True)):
        body = spec.worldbody.add_body(
            name=f"liquid_nitrogen_fog_particle_{i:03d}",
            pos=LIQUID_NITROGEN_FOG_CENTER,
        )
        body.add_geom(
            name=f"liquid_nitrogen_fog_particle_{i:03d}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(float(radius), 0.0, 0.0),
            rgba=(*LIQUID_NITROGEN_FOG_COLOR, float(a)),
            contype=0,
            conaffinity=0,
        )
    return spec


def get_tweezers_spec(use_spring: bool = True) -> mujoco.MjSpec:
    spec_path = TWEEZERS_XML if use_spring else DEXONOMY_TWEEZERS_XML
    spec = mujoco.MjSpec.from_file(str(spec_path))
    tweezers_body = spec.body("tweezers")
    tweezers_body.add_freejoint(name="tweezers_freejoint")
    return spec


def get_tweezers_cfg(use_spring: bool = True) -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=TWEEZERS_INIT_POS,
            rot=TWEEZERS_INIT_ROT,
            joint_pos={"tweezers_hinge": -0.1} if use_spring else {},
            joint_vel={".*": 0.0},
        ),
        spec_fn=partial(get_tweezers_spec, use_spring=use_spring),
    )


def get_beaker_spec() -> mujoco.MjSpec:
    if not BEAKER_XML.exists():
        raise FileNotFoundError(f"Missing MJCF: {BEAKER_XML}")
    spec = mujoco.MjSpec.from_file(str(BEAKER_XML))
    beaker_body = spec.body("glassware010")
    beaker_body.add_site(
        name="beaker_center",
        pos=BEAKER_CENTER_LOCAL_POS,
        quat=(1.0, 0.0, 0.0, 0.0),
        size=(0.005,),
        rgba=(0.0, 1.0, 0.0, 1.0),
        group=3,
    )
    return spec


def get_beaker_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=BEAKER_INIT_POS,
            rot=BEAKER_INIT_ROT,
        ),
        spec_fn=get_beaker_spec,
    )


def get_wuji_tube_spec() -> mujoco.MjSpec:
    cryovial_spec = mujoco.MjSpec.from_file(str(CRYOVIAL_XML))

    body_collision_idx = 0
    cap_collision_idx = 0
    for body in cryovial_spec.bodies:
        is_body = body.name == "cryovial_1-8ml_body"
        is_cap = body.name == "cryovial_1-8ml_cap"
        for geom in body.geoms:
            if geom.group != 2:
                geom.density = WUJI_TUBE_DENSITY
            if is_body and geom.group == 3:
                geom.name = f"{TUBE_BODY_RACK_GEOM_PREFIX}_{body_collision_idx}"
                body_collision_idx += 1
                continue
            if is_cap and geom.group == 3:
                geom.name = f"{CAP_TWEEZERS_GEOM_PREFIX}_{cap_collision_idx}"
                cap_collision_idx += 1

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cryovial_1-8ml")
    body.add_joint(
        name="cryovial_1-8ml_joint",
        type=mujoco.mjtJoint.mjJNT_FREE,
    )
    frame = body.add_frame()
    spec.attach(child=cryovial_spec, prefix="", frame=frame)
    return spec


def get_wuji_tube_rack_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MICRO_TUBE_RACK_XML))
    rack_collision_idx = 0
    for geom in spec.geoms:
        if geom.group != 3:
            continue
        geom.name = f"{RACK_TUBE_BODY_GEOM_PREFIX}_{rack_collision_idx}"
        rack_collision_idx += 1
    return spec


def get_wuji_tube_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=cryovial_slot_init_pos(),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_wuji_tube_spec,
    )


def get_wuji_tube_rack_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=TUBE_RACK_INIT_POS,
            rot=TUBE_RACK_INIT_ROT,
        ),
        spec_fn=get_wuji_tube_rack_spec,
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


def sample_xy_yaw_pose_delta(
    env,
    env_ids: torch.Tensor,
    pos_low: tuple[float, float, float],
    pos_high: tuple[float, float, float],
    yaw_low: float,
    yaw_high: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.tensor(pos_low, device=env.device, dtype=dtype)
    high = torch.tensor(pos_high, device=env.device, dtype=dtype)
    pos_delta = torch.rand((env_ids.numel(), 3), device=env.device, dtype=dtype) * (high - low) + low
    yaw = torch.empty((env_ids.numel(),), device=env.device, dtype=dtype).uniform_(
        float(yaw_low),
        float(yaw_high),
    )
    return pos_delta, yaw


def randomize_wuji_tweezers_pose(
    env,
    env_ids,
    tweezers_cfg: SceneEntityCfg,
    pos_low: tuple[float, float, float] = WUJI_TWEEZERS_RANDOM_POS_LOW,
    pos_high: tuple[float, float, float] = WUJI_TWEEZERS_RANDOM_POS_HIGH,
    yaw_low: float = WUJI_TWEEZERS_RANDOM_YAW_LOW,
    yaw_high: float = WUJI_TWEEZERS_RANDOM_YAW_HIGH,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    tweezers = env.scene[tweezers_cfg.name]
    default_pose = tweezers.data.default_root_state[env_ids, :7].clone()
    dtype = default_pose.dtype
    pos_delta, yaw = sample_xy_yaw_pose_delta(
        env,
        env_ids,
        pos_low,
        pos_high,
        yaw_low,
        yaw_high,
        dtype,
    )
    yaw_axis = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=dtype)
    yaw_quat = axis_angle_to_quat(yaw_axis, yaw, B=env_ids.numel())

    pose = default_pose.clone()
    pose[:, :3] = default_pose[:, :3] + pos_delta + env.scene.env_origins[env_ids].to(dtype=dtype)
    pose[:, 3:7] = quat_mul(yaw_quat, default_pose[:, 3:7])
    tweezers.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    zero_vel = torch.zeros_like(tweezers.data.root_link_vel_w[env_ids])
    tweezers.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)
    env.wuji_randomized_tweezers_pose_w = pose


def align_wuji_visible_fog_to_beaker(
    env,
    env_ids,
    fog_cfg: SceneEntityCfg,
    beaker_center_cfg: SceneEntityCfg,
    base_fog_center: tuple[float, float, float] = LIQUID_NITROGEN_FOG_CENTER,
    fog_center_site_offset: tuple[float, float, float] = LIQUID_NITROGEN_FOG_CENTER_SITE_OFFSET,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    fog = env.scene[fog_cfg.name]
    if not fog.is_mocap:
        raise RuntimeError("Wuji visible fog alignment requires the fog entity to be mocap-wrapped.")

    env.sim.forward()
    beaker = env.scene[beaker_center_cfg.name]
    site_ids = beaker_center_cfg.site_ids
    assert site_ids is not None and len(site_ids) >= 1, "Please specify beaker center site_names."

    site_pos = beaker.data.site_pose_w[env_ids, site_ids[0], :3]
    dtype = site_pos.dtype
    target_center = site_pos + torch.tensor(
        fog_center_site_offset,
        device=env.device,
        dtype=dtype,
    )
    base_center = torch.tensor(base_fog_center, device=env.device, dtype=dtype)

    pose = fog.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = target_center - base_center
    fog.write_mocap_pose_to_sim(pose, env_ids=env_ids)
    env.wuji_visible_fog_center_w = target_center

    if env_ids[0].item() == 0:
        target_center_np = target_center[0].detach().cpu().numpy()
        for body_id in range(env.sim.mj_model.nbody):
            body_name = mujoco.mj_id2name(
                env.sim.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if body_name and "liquid_nitrogen_fog_particle_" in body_name:
                env.sim.mj_model.body_pos[body_id] = target_center_np
        mujoco.mj_forward(env.sim.mj_model, env.sim.mj_data)
    env.sim.forward()


def add_scene_contact_pairs(spec: mujoco.MjSpec) -> None:
    """Inject high-friction contact pairs for the Wuji tweezer scene."""

    tweezers_geoms = (
        "tweezers/tweezers_lower_arm_collision",
        "tweezers/tweezers_upper_arm_collision",
    )
    tube_cap_geoms = tuple(
        geom.name
        for geom in spec.geoms
        if geom.name.startswith(f"tube/{CAP_TWEEZERS_GEOM_PREFIX}_")
    )
    tube_body_geoms = tuple(
        geom.name
        for geom in spec.geoms
        if geom.name.startswith(f"tube/{TUBE_BODY_RACK_GEOM_PREFIX}_")
    )
    rack_geoms = tuple(
        geom.name
        for geom in spec.geoms
        if geom.name.startswith(f"tube_rack/{RACK_TUBE_BODY_GEOM_PREFIX}_")
    )

    wuji_geoms: list[str] = []
    for geom in spec.geoms:
        if geom.name or not geom.meshname.startswith("robot/wuji:right_"):
            continue
        if geom.contype == 0 and geom.conaffinity == 0:
            continue

        geom.name = f"{geom.meshname}_collision"
        wuji_geoms.append(geom.name)

    geom_names = {g.name for g in spec.geoms}
    missing_geoms = [name for name in tweezers_geoms if name not in geom_names]
    if missing_geoms:
        warnings.warn(
            f"Skip Wuji/tweezers contact pairs: geom missing in scene: {missing_geoms}.",
            stacklevel=2,
        )
        return
    if not tube_cap_geoms:
        warnings.warn(
            "Skip tube cap/tweezers contact pairs: no named tube cap collision geoms found.",
            stacklevel=2,
        )
    if not tube_body_geoms or not rack_geoms:
        warnings.warn(
            "Skip tube body/rack contact pairs: named tube body or rack collision geoms missing.",
            stacklevel=2,
        )

    friction = (100.0, 100.0, 0.005, 0.0001, 0.0001)
    for wuji_geom in wuji_geoms:
        for tweezers_geom in tweezers_geoms:
            spec.add_pair(
                geomname1=wuji_geom,
                geomname2=tweezers_geom,
                friction=friction,
                solref=(0.001, 1.0),
                solimp=(0.85, 0.95, 0.001, 0.5, 2.0),
            )

    # Use explicit pairs for every graspable tube surface.  The body ribs and cap
    # otherwise inherit softer geom defaults, which lets the thin tweezer meshes
    # penetrate before enough tangential force is generated to lift the tube.
    for tube_geom in (*tube_cap_geoms, *tube_body_geoms):
        for tweezers_geom in tweezers_geoms:
            spec.add_pair(
                geomname1=tube_geom,
                geomname2=tweezers_geom,
                condim=4,
                friction=TUBE_TWEEZERS_PAIR_FRICTION,
                solref=TUBE_TWEEZERS_PAIR_SOLREF,
                solimp=TUBE_TWEEZERS_PAIR_SOLIMP,
            )

    for tube_body_geom in tube_body_geoms:
        for rack_geom in rack_geoms:
            spec.add_pair(
                geomname1=tube_body_geom,
                geomname2=rack_geom,
                friction=TUBE_BODY_RACK_PAIR_FRICTION,
                solref=(0.0005, 1.0),
                solimp=(0.95, 0.995, 0.0001, 0.5, 2.0),
            )


def reset_wuji_tweezer_success_state(env, env_ids) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    complete = getattr(env, "wuji_tweezers_manipulate_policy_complete", None)
    if complete is None or complete.shape[0] != env.num_envs:
        complete = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    complete[env_ids] = False
    env.wuji_tweezers_manipulate_policy_complete = complete

    seen = getattr(env, "wuji_tweezer_success_seen", None)
    if seen is None or seen.shape[0] != env.num_envs:
        seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    seen[env_ids] = False
    env.wuji_tweezer_success_seen = seen

    held_seen = getattr(env, "wuji_tweezers_held_seen", None)
    if held_seen is None or held_seen.shape[0] != env.num_envs:
        held_seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    held_seen[env_ids] = False
    env.wuji_tweezers_held_seen = held_seen

    no_contact_steps = getattr(env, "wuji_tweezers_no_contact_steps", None)
    if no_contact_steps is None or no_contact_steps.shape[0] != env.num_envs:
        no_contact_steps = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
    no_contact_steps[env_ids] = 0
    env.wuji_tweezers_no_contact_steps = no_contact_steps


def _tweezers_in_hand(
    env,
    tweezers_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    max_tweezers_palm_dist: float,
    min_hand_contact_force: float,
) -> torch.Tensor:
    tweezers = env.scene[tweezers_cfg.name]
    robot = env.scene[palm_cfg.name]
    palm_body_ids = palm_cfg.body_ids
    assert palm_body_ids is not None, "Please specify the palm body."
    assert len(palm_body_ids) == 1, "Please specify exactly one palm body."

    tweezers_pos = tweezers.data.root_link_pose_w[:, :3]
    palm_pos = robot.data.body_link_pos_w[:, palm_body_ids[0], :3]
    within_palm_distance = (
        torch.linalg.vector_norm(tweezers_pos - palm_pos, dim=-1)
        <= max_tweezers_palm_dist
    )
    hand_contact_force = tweezers_hand_contact_force(env)
    return within_palm_distance & (hand_contact_force >= min_hand_contact_force)


def tweezers_hand_contact_force(env) -> torch.Tensor:
    geom_ids = getattr(env, "wuji_tweezers_contact_geom_ids", None)
    if geom_ids is None:
        model = env.sim.mj_model
        hand_ids: list[int] = []
        tweezers_ids: list[int] = []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ""
            if name.startswith("robot/wuji:right_") and name.endswith("_collision"):
                hand_ids.append(geom_id)
            elif name in {
                "tweezers/tweezers_lower_arm_collision",
                "tweezers/tweezers_upper_arm_collision",
            }:
                tweezers_ids.append(geom_id)
        geom_ids = (
            torch.tensor(hand_ids, device=env.device, dtype=torch.long),
            torch.tensor(tweezers_ids, device=env.device, dtype=torch.long),
        )
        env.wuji_tweezers_contact_geom_ids = geom_ids

    hand_ids, tweezers_ids = geom_ids
    force = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    if hand_ids.numel() == 0 or tweezers_ids.numel() == 0:
        return force

    data = env.sim.data
    nacon = getattr(data, "nacon", None)
    if nacon is None or nacon.numel() == 0:
        return force
    contact_count = min(int(nacon[0].item()), data.contact.geom.shape[0])
    if contact_count <= 0:
        return force

    contact_geoms = data.contact.geom[:contact_count].long()
    hand_tweezers_pair = (
        torch.isin(contact_geoms[:, 0], hand_ids)
        & torch.isin(contact_geoms[:, 1], tweezers_ids)
    ) | (
        torch.isin(contact_geoms[:, 1], hand_ids)
        & torch.isin(contact_geoms[:, 0], tweezers_ids)
    )
    penetrating = data.contact.dist[:contact_count] <= 0.0
    contact_worlds = data.contact.worldid[:contact_count].long()
    contact_dims = data.contact.dim[:contact_count]
    normal_addresses = data.contact.efc_address[:contact_count, 0].long()
    valid_contacts = (
        hand_tweezers_pair
        & penetrating
        & (contact_dims > 0)
        & (normal_addresses >= 0)
        & (normal_addresses < data.efc.force.shape[1])
        & (contact_worlds >= 0)
        & (contact_worlds < env.num_envs)
    )
    valid_worlds = contact_worlds[valid_contacts]
    valid_addresses = normal_addresses[valid_contacts]
    normal_forces = data.efc.force[valid_worlds, valid_addresses].abs().float()
    force.index_add_(0, valid_worlds, normal_forces)
    return force


def check_tweezers_dropped(
    env,
    tweezers_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    max_tweezers_palm_dist: float,
    min_hand_contact_force: float,
    max_no_contact_steps: int,
) -> torch.Tensor:
    in_hand = _tweezers_in_hand(
        env,
        tweezers_cfg=tweezers_cfg,
        palm_cfg=palm_cfg,
        max_tweezers_palm_dist=max_tweezers_palm_dist,
        min_hand_contact_force=min_hand_contact_force,
    )
    held_seen = getattr(env, "wuji_tweezers_held_seen", None)
    if held_seen is None or held_seen.shape[0] != env.num_envs:
        held_seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    held_seen |= in_hand
    env.wuji_tweezers_held_seen = held_seen

    no_contact_steps = getattr(env, "wuji_tweezers_no_contact_steps", None)
    if no_contact_steps is None or no_contact_steps.shape[0] != env.num_envs:
        no_contact_steps = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
    no_contact_steps = torch.where(
        held_seen & ~in_hand,
        no_contact_steps + 1,
        torch.zeros_like(no_contact_steps),
    )
    env.wuji_tweezers_no_contact_steps = no_contact_steps
    return no_contact_steps >= max_no_contact_steps


def check_success(
    env,
    tube_cfg: SceneEntityCfg,
    beaker_center_cfg: SceneEntityCfg,
    tweezers_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    tube_axis_length: float,
    tube_radius: float,
    beaker_inner_radius: float,
    beaker_interior_bottom_height: float,
    beaker_rim_height: float,
    containment_tolerance: float,
    bottom_tolerance: float,
    stability_steps: int,
    max_tweezers_palm_dist: float,
    min_hand_contact_force: float,
) -> torch.Tensor:
    tube = env.scene[tube_cfg.name]
    beaker = env.scene[beaker_center_cfg.name]

    site_ids = beaker_center_cfg.site_ids
    assert site_ids is not None and len(site_ids) >= 1, "Please specify beaker center site_names."

    tube_pose_w = tube.data.root_link_pose_w[:, :7]
    beaker_center_pose_w = beaker.data.site_pose_w[:, site_ids[0], :7]
    tube_pose_beaker = compose_pose_wxyz(
        pose_inv_wxyz(beaker_center_pose_w),
        tube_pose_w,
    )
    tube_bottom = tube_pose_beaker[:, :3]
    tube_axis = quat_rotate_vec(
        tube_pose_beaker[:, 3:7],
        torch.tensor(
            [0.0, 0.0, 1.0],
            device=env.device,
            dtype=tube_pose_beaker.dtype,
        ).expand(env.num_envs, 3),
    )
    tube_top = tube_bottom + float(tube_axis_length) * tube_axis

    bottom_axis_radius = torch.linalg.vector_norm(tube_bottom[:, :2], dim=-1)
    top_axis_radius = torch.linalg.vector_norm(tube_top[:, :2], dim=-1)
    tube_outer_radius = (
        torch.maximum(bottom_axis_radius, top_axis_radius) + float(tube_radius)
    )
    vertical_radius = float(tube_radius) * torch.sqrt(
        torch.clamp(1.0 - tube_axis[:, 2].square(), min=0.0)
    )
    tube_lowest_height = (
        torch.minimum(tube_bottom[:, 2], tube_top[:, 2]) - vertical_radius
    )
    tube_highest_height = (
        torch.maximum(tube_bottom[:, 2], tube_top[:, 2]) + vertical_radius
    )

    tube_within_beaker_radius = (
        tube_outer_radius <= float(beaker_inner_radius + containment_tolerance)
    )
    tube_above_beaker_bottom = (
        tube_lowest_height
        >= float(beaker_interior_bottom_height - bottom_tolerance)
    )
    tube_below_beaker_rim = (
        tube_highest_height <= float(beaker_rim_height + containment_tolerance)
    )
    tweezers_in_hand = _tweezers_in_hand(
        env,
        tweezers_cfg=tweezers_cfg,
        palm_cfg=palm_cfg,
        max_tweezers_palm_dist=max_tweezers_palm_dist,
        min_hand_contact_force=min_hand_contact_force,
    )
    object_complete_now = (
        tube_within_beaker_radius
        & tube_above_beaker_bottom
        & tube_below_beaker_rim
    )
    object_complete, stable_count = stable_success(
        env,
        object_complete_now,
        state_attr="wuji_tweezer_success_stable_steps",
        required_steps=stability_steps,
    )

    record_success_diagnostics(
        env,
        object_complete=object_complete_now,
        full_success=object_complete,
        components={
            "tube_within_beaker_radius": tube_within_beaker_radius,
            "tube_above_beaker_bottom": tube_above_beaker_bottom,
            "tube_below_beaker_rim": tube_below_beaker_rim,
            "tweezers_in_hand": tweezers_in_hand,
        },
        metrics={
            "tube_outer_radius_m": tube_outer_radius,
            "tube_lowest_height_m": tube_lowest_height,
            "tube_highest_height_m": tube_highest_height,
            "stable_steps": stable_count,
        },
        thresholds={
            "beaker_inner_radius_m": beaker_inner_radius,
            "beaker_interior_bottom_height_m": beaker_interior_bottom_height,
            "beaker_rim_height_m": beaker_rim_height,
            "containment_tolerance_m": containment_tolerance,
            "bottom_tolerance_m": bottom_tolerance,
            "required_stability_steps": stability_steps,
            "max_tweezers_palm_dist_m": max_tweezers_palm_dist,
            "min_hand_contact_force_n": min_hand_contact_force,
        },
    )
    seen = getattr(env, "wuji_tweezer_success_seen", None)
    if seen is None or seen.shape[0] != env.num_envs:
        seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    seen |= object_complete
    env.wuji_tweezer_success_seen = seen
    return seen & object_complete


def make_wuji_tweezers_manipulate_env_cfg(
    play: bool = False,
    tweezers_spring: bool = True,
) -> ManagerBasedRlEnvCfg: # the play now uses nothing
    # --------------------------
    # Scene
    # --------------------------

    robot_cfg = get_ur5e_wuji_cfg()
    robot_cfg.spec_fn = get_wuji_tweezers_robot_spec
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

    tweeezers_cfg = get_tweezers_cfg(use_spring=tweezers_spring)
    tweeezers_cfg.init_state.pos = (0.4, -0.3, 0.844)
    tweeezers_cfg.init_state.rot = (-1.0, 0.0, 0.0, 0.0)

    tube_rack_cfg = get_wuji_tube_rack_cfg()

    fog_geom_names = ()
    if ENABLE_LIQUID_NITROGEN_FOG_RENDER:
        fog_particle_params, fog_base_radius, fog_alpha = make_liquid_nitrogen_fog_particles(
            LIQUID_NITROGEN_FOG_PARTICLES
        )
        fog_geom_names = tuple(
            f"liquid_nitrogen_fog_particle_{i:03d}" for i in range(LIQUID_NITROGEN_FOG_PARTICLES)
        )

    gs_wrist_cam, gs_front_cam, gs_left_cam = make_gaussian_camera_triplet(
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention big table",
        scene_parent_body="table/vention big table",
        wrist_camera_name="robot/wuji:wrist_cam",
        front_pos=(0.0, -1.2, 1.45),
        front_quat=(0.819, 0.574, 0.0, 0.0),
        left_pos=(-1.2, 0.0, 1.45),
        left_quat=(0.579, 0.406, -0.406, -0.579),
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )

    scene_entities = {
        "robot": robot_cfg,
        "tweezers": tweeezers_cfg,
        "tube_rack": tube_rack_cfg,
        "beaker": get_beaker_cfg(),
        "tube": get_wuji_tube_cfg(),
        "table": get_gaussian_hidden_table_cfg(),
    }
    if ENABLE_LIQUID_NITROGEN_FOG_RENDER:
        scene_entities["liquid_nitrogen_fog"] = EntityCfg(
            spec_fn=lambda: make_liquid_nitrogen_fog_spec(fog_base_radius, fog_alpha),
        )

    scene = SceneCfg(
        terrain=TerrainEntityCfg(
        terrain_type="plane",
        ),
        entities=scene_entities,
        spec_fn=add_scene_contact_pairs,
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
                            joint_names=UR5E_JOINT_ORDER+WUJI_JOINT_ORDER,
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
                            body_names=[r".*right_palm_link"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tweezers_body_pose_w": ObservationGroupCfg(
            {
                "tweezers_body_pose_w": ObservationTermCfg(
                    func=root_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg("tweezers"),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tube_body_pose_w": ObservationGroupCfg(
            {
                "tube_body_pose_w": ObservationTermCfg(
                    func=root_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg("tube"),
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
        frame_name=r".*right_palm_link",
        use_relative_mode=False,
        position_weight=1,
        orientation_weight=0.5,
    ),
    "wuji": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=WUJI_JOINT_ORDER,
        use_default_offset=False,
        preserve_order=True,
        clip=None,
    ),
    }

    # --------------------------
    # Events
    # --------------------------
    events = {
        "reset_scene": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "randomize_wuji_tweezers_pose": EventTermCfg(
            func=randomize_wuji_tweezers_pose,
            mode="reset",
            params={
                "tweezers_cfg": SceneEntityCfg("tweezers"),
            },
        ),
        "reset_tube_on_rack": EventTermCfg(
            func=reset_centrifuge_on_rack,
            mode="reset",
            params={
                "site_asset_cfg": SceneEntityCfg(
                    "tube_rack",
                    site_names=["centrifuge_plate-0"],
                ),
                "asset_cfg": SceneEntityCfg("tube"),
                "rows": CRYOVIAL_SLOT_COLS,
                "cols": CRYOVIAL_SLOT_ROWS,
                "row_gap": CRYOVIAL_SLOT_COL_GAP,
                "col_gap": CRYOVIAL_SLOT_ROW_GAP,
                "height": CRYOVIAL_SLOT_HEIGHT,
            },
        ),
        "randomize_initial_hand_pose": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": WUJI_INITIAL_ARM_JOINT_NOISE_LOW,
                "highs": WUJI_INITIAL_ARM_JOINT_NOISE_HIGH,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_JOINT_ORDER,
                ),
            },
        ),
        "reset_wuji_tweezer_success_state": EventTermCfg(
            func=reset_wuji_tweezer_success_state,
            mode="reset",
        ),
    }
    if ENABLE_LIQUID_NITROGEN_FOG_RENDER:
        events["align_wuji_visible_fog_to_beaker"] = EventTermCfg(
            func=align_wuji_visible_fog_to_beaker,
            mode="reset",
            params={
                "fog_cfg": SceneEntityCfg("liquid_nitrogen_fog"),
                "beaker_center_cfg": SceneEntityCfg(
                    "beaker",
                    site_names=["beaker_center"],
                ),
            },
        )
        events["install_liquid_nitrogen_fog"] = EventTermCfg(
            func=RenderOnlyFogInstaller(
                particle_params=fog_particle_params,
                base_radius=fog_base_radius,
                alpha=fog_alpha,
                geom_names=fog_geom_names,
                fog_center=LIQUID_NITROGEN_FOG_CENTER,
                fog_radius=LIQUID_NITROGEN_FOG_RADIUS,
                fog_height=LIQUID_NITROGEN_FOG_HEIGHT,
                fog_time_scale=LIQUID_NITROGEN_FOG_TIME_SCALE,
                color=LIQUID_NITROGEN_FOG_COLOR,
                gaussian_sprite_config=mjwarp.GaussianSpriteConfig(
                    particle_sigma=LIQUID_NITROGEN_FOG_SPRITE_SIGMA,
                    particle_alpha=1.0,
                    particle_scale=LIQUID_NITROGEN_FOG_SPRITE_SCALE,
                    particle_alpha_cutoff=1.0 / 255.0,
                    particle_velocity_scale=0.0,
                    particle_max_anisotropy=1.0,
                ),
            ),
            mode="startup",
        )

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
                "tube_cfg": SceneEntityCfg("tube"),
                "beaker_center_cfg": SceneEntityCfg(
                    "beaker",
                    site_names=["beaker_center"],
                ),
                "tweezers_cfg": SceneEntityCfg("tweezers"),
                "palm_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[r".*right_palm_link"],
                ),
                "tube_axis_length": WUJI_TUBE_AXIS_LENGTH,
                "tube_radius": WUJI_TUBE_MAX_RADIUS,
                "beaker_inner_radius": WUJI_BEAKER_INNER_RADIUS,
                "beaker_interior_bottom_height": WUJI_BEAKER_INTERIOR_BOTTOM_HEIGHT,
                "beaker_rim_height": WUJI_BEAKER_RIM_HEIGHT,
                "containment_tolerance": WUJI_TWEEZER_SUCCESS_CONTAINMENT_TOLERANCE,
                "bottom_tolerance": WUJI_TWEEZER_SUCCESS_BOTTOM_TOLERANCE,
                "stability_steps": WUJI_TWEEZER_SUCCESS_STABILITY_STEPS,
                "max_tweezers_palm_dist": WUJI_TWEEZER_SUCCESS_MAX_PALM_DIST,
                "min_hand_contact_force": WUJI_TWEEZER_MIN_HAND_CONTACT_FORCE,
            },
        ),
        "tweezers_dropped": TerminationTermCfg(
            func=check_tweezers_dropped,
            params={
                "tweezers_cfg": SceneEntityCfg("tweezers"),
                "palm_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[r".*right_palm_link"],
                ),
                "max_tweezers_palm_dist": WUJI_TWEEZER_SUCCESS_MAX_PALM_DIST,
                "min_hand_contact_force": WUJI_TWEEZER_MIN_HAND_CONTACT_FORCE,
                "max_no_contact_steps": WUJI_TWEEZER_DROP_NO_CONTACT_STEPS,
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
        episode_length_s=WUJI_TWEEZER_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
