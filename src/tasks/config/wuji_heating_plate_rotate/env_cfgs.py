import copy
from pathlib import Path
import warnings

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.differential_ik import DifferentialIKActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from src import MJCF_ROOT
from assets.mjcf.instrument.heating_plate import (
    get_heating_plate_cfg,
    get_heating_plate_spec,
)
from assets.mjcf.misc.simple_table import get_table_cfg
from assets.mjcf.robot.ur5e_wuji import (
    UR5E_JOINT_ORDER,
    WUJI_JOINT_ORDER,
    get_ur5e_wuji_cfg,
)
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.events import add_noise_to_joints
from src.tasks.mdp.heating_effects import RenderOnlyHeatingEffects
from src.tasks.success_diagnostics import (
    record_success_diagnostics,
)
from src.tasks.mdp.observations import (
    body_pose_w,
    camera_rgb,
    joint_pos_abs,
)


HEATING_PLATE_SUCCESS_ROTATION = 0.6
HEATING_PLATE_SUCCESS_HOME_TOLERANCE = 0.05
HEATING_PLATE_SUCCESS_HAND_LIFT = 0.05
HEATING_PLATE_POLICY_DONE_STEPS = 613
HEATING_PLATE_ENV_STEP_S = 0.002 * 25
HEATING_PLATE_EPISODE_LENGTH_S = (
    2.0 * HEATING_PLATE_POLICY_DONE_STEPS * HEATING_PLATE_ENV_STEP_S
)
HEATING_EFFECT_KNOB_THRESHOLD_RAD = 0.6
HEATING_EFFECT_MAX_KNOB_ROTATION_RAD = 1.57
HEATING_EFFECT_RAMP_UP_S = 20.0
HEATING_EFFECT_COOL_DOWN_S = 45.0
HEATING_EFFECT_STIR_RAMP_UP_S = 2.0
HEATING_EFFECT_STIR_COAST_DOWN_S = 30.0
HEATING_EFFECT_MAX_PLATE_XY_DIST = 0.075
HEATING_EFFECT_MIN_PLATE_HEIGHT = 0.105
HEATING_EFFECT_MAX_PLATE_HEIGHT = 0.190
HEATING_EFFECT_BUBBLES = 128
HEATING_EFFECT_INTENSITY_ATTR = "wuji_heating_plate_heat_intensity"
HEATING_EFFECT_STIR_MOMENTUM_ATTR = "wuji_heating_plate_stir_momentum"
HEATING_EFFECT_STIR_PHASE_ATTR = "wuji_heating_plate_stir_phase"
INITIAL_ARM_JOINT_NOISE_LOW = (-0.10, -0.10, -0.10, -0.10, -0.10, -0.10)
INITIAL_ARM_JOINT_NOISE_HIGH = (0.10, 0.10, 0.10, 0.10, 0.10, 0.10)
HEATING_PLATE_RANDOM_POS_LOW = (-0.02, 0.0, 0.0)
HEATING_PLATE_RANDOM_POS_HIGH = (0.02, 0.02, 0.0)
BEAKER_RANDOM_POS_LOW = (-0.02, -0.02, 0.0)
BEAKER_RANDOM_POS_HIGH = (0.02, 0.0, 0.0)
BEAKER_XML = MJCF_ROOT / "object" / "glassware010.xml"
BEAKER_INIT_POS = (-0.4, -0.1, 0.824)
BEAKER_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
BEAKER_CENTER_LOCAL_POS = (-0.0023, 0.0018147, 0.0491)
BEAKER_COLLISION_POS = (-0.0023, 0.0018147, 0.0473)
BEAKER_COLLISION_RADIUS = 0.0362
BEAKER_COLLISION_HALF_HEIGHT = 0.0473
BEAKER_COLLISION_MASS = 0.20
BEAKER_WATER_POS = (-0.0023, 0.0018147, 0.038)
BEAKER_WATER_RADIUS = 0.030
BEAKER_WATER_HALF_HEIGHT = 0.034
BEAKER_WATER_RGBA = (0.95, 0.72, 0.18, 0.42)
HIGH_GRIP_FRICTION = (5.0, 5, 0.005)
KNOB_WUJI_PAIR_FRICTION = (80.0, 80.0, 0.05, 0.0001, 0.0001)
KNOB_WUJI_PAIR_SOLREF = (0.001, 1.0)
KNOB_WUJI_PAIR_SOLIMP = (0.85, 0.95, 0.001, 0.5, 2.0)
WUJI_WRIST_CAM_POS = (0.025717725, -0.0295648735, -0.0305499135)
WUJI_WRIST_CAM_QUAT = (-0.263059749, -0.392408433, 0.870930529, -0.135259763)
WUJI_WRIST_CAM_FOVY = 58.0
_GAUSSIAN_SCENE = gaussian_scene_for_task("Wuji-HeatingPlate-Rotate")
_TABLE_GEOM_NAMES = tuple(
    f"table_geom_{idx}" for idx, _ in enumerate(get_table_cfg().spec_fn().geoms)
)
_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES = (
    "terrain",
    *(f"table/{name}" for name in _TABLE_GEOM_NAMES),
)
assert BEAKER_XML.exists(), f"Missing MJCF: {BEAKER_XML}"


def get_wuji_heating_plate_rotate_robot_spec() -> mujoco.MjSpec:
    spec = get_ur5e_wuji_cfg().spec_fn()
    wrist_cam = spec.camera("wuji:wrist_cam")
    wrist_cam.pos = WUJI_WRIST_CAM_POS
    wrist_cam.quat = WUJI_WRIST_CAM_QUAT
    wrist_cam.fovy = WUJI_WRIST_CAM_FOVY
    return spec


def get_gaussian_hidden_table_cfg() -> EntityCfg:
    table_cfg = get_table_cfg()
    base_spec_fn = table_cfg.spec_fn

    def table_spec_with_geom_names():
        spec = base_spec_fn()
        for geom, name in zip(spec.geoms, _TABLE_GEOM_NAMES):
            geom.name = name
        return spec

    table_cfg.spec_fn = table_spec_with_geom_names
    return table_cfg


def get_beaker_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(BEAKER_XML))
    beaker_body = spec.body("glassware010")
    beaker_body.add_freejoint(name="beaker_freejoint")
    for geom in list(beaker_body.geoms):
        if geom.group == 3:
            spec.delete(geom)
    beaker_body.add_geom(
        name="glassware010_water_visual",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=BEAKER_WATER_POS,
        size=(BEAKER_WATER_RADIUS, BEAKER_WATER_HALF_HEIGHT),
        rgba=BEAKER_WATER_RGBA,
        contype=0,
        conaffinity=0,
        density=0,
        group=2,
    )
    beaker_body.add_geom(
        name="glassware010_rotor_visual",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        pos=(BEAKER_CENTER_LOCAL_POS[0], BEAKER_CENTER_LOCAL_POS[1], 0.0081),
        quat=(0.70710678, 0.0, 0.70710678, 0.0),
        size=(0.003, 0.012, 0.0),
        rgba=(0.92, 0.92, 0.88, 1.0),
        contype=0,
        conaffinity=0,
        density=0.0,
        group=2,
    )
    beaker_body.add_geom(
        name="glassware010_cylinder_collision",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=BEAKER_COLLISION_POS,
        size=(BEAKER_COLLISION_RADIUS, BEAKER_COLLISION_HALF_HEIGHT),
        mass=BEAKER_COLLISION_MASS,
        friction=HIGH_GRIP_FRICTION,
        group=3,
    )
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


def randomize_heating_plate_and_beaker_positions(
    env,
    env_ids,
    heating_plate_cfg: SceneEntityCfg,
    beaker_cfg: SceneEntityCfg,
    heating_plate_pos_low: tuple[float, float, float] = HEATING_PLATE_RANDOM_POS_LOW,
    heating_plate_pos_high: tuple[float, float, float] = HEATING_PLATE_RANDOM_POS_HIGH,
    beaker_pos_low: tuple[float, float, float] = BEAKER_RANDOM_POS_LOW,
    beaker_pos_high: tuple[float, float, float] = BEAKER_RANDOM_POS_HIGH,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    heating_plate = env.scene[heating_plate_cfg.name]
    beaker = env.scene[beaker_cfg.name]
    if not heating_plate.is_mocap:
        raise RuntimeError("Heating plate position randomization requires a mocap-wrapped entity.")

    plate_pose = heating_plate.data.default_root_state[env_ids, :7].clone()
    beaker_pose = beaker.data.default_root_state[env_ids, :7].clone()
    env_origins = env.scene.env_origins[env_ids].to(
        device=env.device,
        dtype=plate_pose.dtype,
    )

    def sample_offset(low, high, dtype):
        low_tensor = torch.tensor(low, device=env.device, dtype=dtype)
        high_tensor = torch.tensor(high, device=env.device, dtype=dtype)
        return torch.rand((env_ids.numel(), 3), device=env.device, dtype=dtype) * (
            high_tensor - low_tensor
        ) + low_tensor

    plate_pose[:, :3] += env_origins + sample_offset(
        heating_plate_pos_low,
        heating_plate_pos_high,
        plate_pose.dtype,
    )
    beaker_pose[:, :3] += env_origins.to(dtype=beaker_pose.dtype) + sample_offset(
        beaker_pos_low,
        beaker_pos_high,
        beaker_pose.dtype,
    )

    heating_plate.write_mocap_pose_to_sim(plate_pose, env_ids=env_ids)
    beaker.write_root_link_pose_to_sim(beaker_pose, env_ids=env_ids)
    beaker.write_root_link_velocity_to_sim(
        torch.zeros_like(beaker.data.root_link_vel_w[env_ids]),
        env_ids=env_ids,
    )
    env.sim.forward()


def get_heating_effects_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    for particle_id in range(HEATING_EFFECT_BUBBLES):
        body = spec.worldbody.add_body(name=f"heating_bubble_{particle_id:03d}")
        body.add_geom(
            name=f"heating_bubble_{particle_id:03d}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.0001, 0.0, 0.0),
            rgba=(1.0, 1.0, 1.0, 0.0),
            contype=0,
            conaffinity=0,
            density=0.0,
            group=2,
        )
    return spec


def get_heating_effects_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=get_heating_effects_spec)


def reset_heating_effects(
    env,
    env_ids,
    renderer: RenderOnlyHeatingEffects,
    beaker_center_cfg: SceneEntityCfg,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    intensity = getattr(env, HEATING_EFFECT_INTENSITY_ATTR, None)
    if intensity is None or intensity.shape[0] != env.num_envs:
        intensity = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, HEATING_EFFECT_INTENSITY_ATTR, intensity)
    intensity[env_ids] = 0.0
    stir_momentum = getattr(env, HEATING_EFFECT_STIR_MOMENTUM_ATTR, None)
    if stir_momentum is None or stir_momentum.shape[0] != env.num_envs:
        stir_momentum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, HEATING_EFFECT_STIR_MOMENTUM_ATTR, stir_momentum)
    stir_momentum[env_ids] = 0.0
    stir_phase = getattr(env, HEATING_EFFECT_STIR_PHASE_ATTR, None)
    if stir_phase is None or stir_phase.shape[0] != env.num_envs:
        stir_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, HEATING_EFFECT_STIR_PHASE_ATTR, stir_phase)
    stir_phase[env_ids] = 0.0
    renderer.launch(env, beaker_center_cfg)


def update_heating_effects(
    env,
    env_ids,
    heating_plate_cfg: SceneEntityCfg,
    beaker_center_cfg: SceneEntityCfg,
    renderer: RenderOnlyHeatingEffects,
) -> None:
    del env_ids
    intensity = getattr(env, HEATING_EFFECT_INTENSITY_ATTR, None)
    if intensity is None or intensity.shape[0] != env.num_envs:
        intensity = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, HEATING_EFFECT_INTENSITY_ATTR, intensity)

    heating_plate = env.scene[heating_plate_cfg.name]
    beaker = env.scene[beaker_center_cfg.name]
    joint_ids = heating_plate_cfg.joint_ids
    site_ids = beaker_center_cfg.site_ids
    assert joint_ids is not None and len(joint_ids) == 1, "Please specify the knob joint."
    assert site_ids is not None and len(site_ids) == 1, "Please specify the beaker center site."

    plate_pos = heating_plate.data.root_link_pose_w[:, :3]
    beaker_center = beaker.data.site_pose_w[:, site_ids[0], :3]
    relative = beaker_center - plate_pos
    centered = torch.linalg.vector_norm(relative[:, :2], dim=-1) <= HEATING_EFFECT_MAX_PLATE_XY_DIST
    above_plate = (relative[:, 2] >= HEATING_EFFECT_MIN_PLATE_HEIGHT) & (
        relative[:, 2] <= HEATING_EFFECT_MAX_PLATE_HEIGHT
    )
    knob_qpos = heating_plate.data.joint_pos[:, joint_ids[0]]
    knob_on = knob_qpos >= HEATING_EFFECT_KNOB_THRESHOLD_RAD
    heating = centered & above_plate & knob_on
    stir_strength = (
        knob_qpos.clamp(0.0, HEATING_EFFECT_MAX_KNOB_ROTATION_RAD)
        / HEATING_EFFECT_MAX_KNOB_ROTATION_RAD
    ) * (centered & above_plate).to(dtype=knob_qpos.dtype)
    stir_momentum = getattr(env, HEATING_EFFECT_STIR_MOMENTUM_ATTR, None)
    if stir_momentum is None or stir_momentum.shape[0] != env.num_envs:
        stir_momentum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, HEATING_EFFECT_STIR_MOMENTUM_ATTR, stir_momentum)
    stir_ramp_delta = float(env.step_dt) / HEATING_EFFECT_STIR_RAMP_UP_S
    stir_coast_delta = float(env.step_dt) / HEATING_EFFECT_STIR_COAST_DOWN_S
    momentum_delta = (stir_strength - stir_momentum).clamp(
        min=-stir_coast_delta,
        max=stir_ramp_delta,
    )
    stir_momentum.add_(momentum_delta).clamp_(0.0, 1.0)

    heat_delta = float(env.step_dt) / HEATING_EFFECT_RAMP_UP_S
    cool_delta = float(env.step_dt) / HEATING_EFFECT_COOL_DOWN_S
    intensity.add_(torch.where(heating, heat_delta, -cool_delta)).clamp_(0.0, 1.0)

    stir_phase = getattr(env, HEATING_EFFECT_STIR_PHASE_ATTR, None)
    if stir_phase is None or stir_phase.shape[0] != env.num_envs:
        stir_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, HEATING_EFFECT_STIR_PHASE_ATTR, stir_phase)
    stir_phase.add_(float(env.step_dt) * stir_strength * 20.0)
    renderer.launch(env, beaker_center_cfg)


def get_high_grip_heating_plate_spec() -> mujoco.MjSpec:
    spec = get_heating_plate_spec()
    knob_body = spec.body("knob")
    for geom in knob_body.geoms:
        if geom.group == 3:
            geom.friction = HIGH_GRIP_FRICTION
    return spec


def get_high_grip_heating_plate_cfg() -> EntityCfg:
    cfg = get_heating_plate_cfg()
    cfg.spec_fn = get_high_grip_heating_plate_spec
    return cfg


def add_scene_contact_pairs(spec: mujoco.MjSpec) -> None:
    """Inject bidirectional high-friction Wuji/knob contact pairs."""

    knob_geoms = tuple(
        geom.name
        for geom in spec.geoms
        if geom.name
        in {
            "heating_plate/thermo_knob_collision",
        }
    )
    wuji_geoms: list[str] = []
    wuji_geom_name_index = 0
    for geom in spec.geoms:
        if geom.name:
            geom_name = geom.name
        elif geom.meshname.startswith("robot/wuji:right_"):
            geom.name = f"{geom.meshname}_collision_{wuji_geom_name_index:03d}"
            wuji_geom_name_index += 1
            geom_name = geom.name
        else:
            continue

        if geom.contype == 0 and geom.conaffinity == 0:
            continue
        if "robot/wuji:right_" in geom_name or "wuji:right_" in geom_name:
            wuji_geoms.append(geom_name)

    if not knob_geoms:
        warnings.warn(
            "Skip Wuji/knob contact pairs: no heating plate knob collision geoms found.",
            stacklevel=2,
        )
        return
    if not wuji_geoms:
        warnings.warn(
            "Skip Wuji/knob contact pairs: no named Wuji collision geoms found.",
            stacklevel=2,
        )
        return

    for knob_geom in knob_geoms:
        for wuji_geom in wuji_geoms:
            spec.add_pair(
                geomname1=knob_geom,
                geomname2=wuji_geom,
                condim=4,
                friction=KNOB_WUJI_PAIR_FRICTION,
                solref=KNOB_WUJI_PAIR_SOLREF,
                solimp=KNOB_WUJI_PAIR_SOLIMP,
            )


def reset_heating_plate_success_state(env, env_ids) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    knob_on_seen = getattr(env, "wuji_heating_plate_knob_on_seen", None)
    if knob_on_seen is None or knob_on_seen.shape[0] != env.num_envs:
        knob_on_seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    knob_on_seen[env_ids] = False
    env.wuji_heating_plate_knob_on_seen = knob_on_seen

    object_complete_seen = getattr(
        env, "wuji_heating_plate_object_complete_seen", None
    )
    if object_complete_seen is None or object_complete_seen.shape[0] != env.num_envs:
        object_complete_seen = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    object_complete_seen[env_ids] = False
    env.wuji_heating_plate_object_complete_seen = object_complete_seen

    lift_start_relative_z = getattr(
        env, "wuji_heating_plate_lift_start_relative_z", None
    )
    if (
        lift_start_relative_z is None
        or lift_start_relative_z.shape[0] != env.num_envs
    ):
        lift_start_relative_z = torch.zeros(env.num_envs, device=env.device)
    lift_start_relative_z[env_ids] = 0.0
    env.wuji_heating_plate_lift_start_relative_z = lift_start_relative_z

    complete = getattr(env, "wuji_heating_plate_rotate_policy_complete", None)
    if complete is None or complete.shape[0] != env.num_envs:
        complete = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    complete[env_ids] = False
    env.wuji_heating_plate_rotate_policy_complete = complete

def check_success(
    env,
    heating_plate_cfg: SceneEntityCfg,
    beaker_center_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    min_rotation: float,
    home_tolerance: float,
    max_plate_xy_dist: float,
    min_plate_height: float,
    max_plate_height: float,
    min_hand_lift: float,
) -> torch.Tensor:
    heating_plate = env.scene[heating_plate_cfg.name]
    joint_ids = heating_plate_cfg.joint_ids
    assert joint_ids is not None, "Please specify the knob joint."
    knob_qpos = heating_plate.data.joint_pos[:, joint_ids]
    assert knob_qpos.shape[1] == 1, "Please specify exactly one knob joint."
    knob_qpos = knob_qpos[:, 0]
    knob_body_ids = heating_plate_cfg.body_ids
    assert knob_body_ids is not None, "Please specify the knob body."
    assert len(knob_body_ids) == 1, "Please specify exactly one knob body."
    knob_z = heating_plate.data.body_link_pose_w[:, knob_body_ids[0], 2]

    beaker = env.scene[beaker_center_cfg.name]
    site_ids = beaker_center_cfg.site_ids
    assert site_ids is not None, "Please specify the beaker center site."
    assert len(site_ids) == 1, "Please specify exactly one beaker center site."
    plate_pos = heating_plate.data.root_link_pose_w[:, :3]
    beaker_center = beaker.data.site_pose_w[:, site_ids[0], :3]
    relative = beaker_center - plate_pos
    centered = torch.linalg.vector_norm(relative[:, :2], dim=-1) <= float(max_plate_xy_dist)
    above_plate = (relative[:, 2] >= float(min_plate_height)) & (
        relative[:, 2] <= float(max_plate_height)
    )
    beaker_on_plate = centered & above_plate

    knob_on_seen = getattr(env, "wuji_heating_plate_knob_on_seen", None)
    if knob_on_seen is None or knob_on_seen.shape[0] != env.num_envs:
        knob_on_seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    knob_on_seen |= knob_qpos >= float(min_rotation)
    env.wuji_heating_plate_knob_on_seen = knob_on_seen

    knob_at_home = knob_on_seen & (torch.abs(knob_qpos) <= float(home_tolerance))
    object_complete = beaker_on_plate & knob_at_home

    robot = env.scene[palm_cfg.name]
    body_ids = palm_cfg.body_ids
    assert body_ids is not None, "Please specify the palm body."
    assert len(body_ids) == 1, "Please specify exactly one palm body."
    palm_z = robot.data.body_link_pose_w[:, body_ids[0], 2]

    object_complete_seen = getattr(
        env, "wuji_heating_plate_object_complete_seen", None
    )
    if object_complete_seen is None or object_complete_seen.shape[0] != env.num_envs:
        object_complete_seen = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    lift_start_relative_z = getattr(
        env, "wuji_heating_plate_lift_start_relative_z", None
    )
    if (
        lift_start_relative_z is None
        or lift_start_relative_z.shape[0] != env.num_envs
    ):
        lift_start_relative_z = torch.zeros_like(palm_z)

    newly_complete = object_complete & ~object_complete_seen
    palm_knob_relative_z = palm_z - knob_z
    lift_start_relative_z = torch.where(
        newly_complete,
        palm_knob_relative_z,
        lift_start_relative_z,
    )
    object_complete_seen |= object_complete
    hand_lift = palm_knob_relative_z - lift_start_relative_z
    hand_lifted = object_complete_seen & (hand_lift >= float(min_hand_lift))
    success = object_complete & hand_lifted
    env.wuji_heating_plate_object_complete_seen = object_complete_seen
    env.wuji_heating_plate_lift_start_relative_z = lift_start_relative_z

    record_success_diagnostics(
        env,
        object_complete=object_complete,
        final_motion_complete=hand_lifted,
        full_success=success,
        components={
            "beaker_on_plate": beaker_on_plate,
            "knob_turned_on_seen": knob_on_seen,
            "knob_at_home": knob_at_home,
            "previous_task_complete_seen": object_complete_seen,
            "hand_lifted": hand_lifted,
        },
        metrics={
            "beaker_plate_xy_dist_m": torch.linalg.vector_norm(
                relative[:, :2],
                dim=-1,
            ),
            "beaker_plate_height_m": relative[:, 2],
            "knob_qpos_rad": knob_qpos,
            "hand_lift_m": hand_lift,
        },
        thresholds={
            "max_plate_xy_dist_m": max_plate_xy_dist,
            "min_plate_height_m": min_plate_height,
            "max_plate_height_m": max_plate_height,
            "min_rotation_rad": min_rotation,
            "home_tolerance_rad": home_tolerance,
            "min_hand_lift_m": min_hand_lift,
        },
    )
    return success


def make_heating_effects_renderer() -> RenderOnlyHeatingEffects:
    return RenderOnlyHeatingEffects(
        bubble_geom_names=tuple(
            f"heating_bubble_{particle_id:03d}"
            for particle_id in range(HEATING_EFFECT_BUBBLES)
        ),
        rotor_geom_name="glassware010_rotor_visual",
        intensity_attr=HEATING_EFFECT_INTENSITY_ATTR,
        stir_momentum_attr=HEATING_EFFECT_STIR_MOMENTUM_ATTR,
        stir_phase_attr=HEATING_EFFECT_STIR_PHASE_ATTR,
    )


def make_wuji_heating_plate_rotate_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    del play

    heating_effects_renderer = make_heating_effects_renderer()

    robot_cfg = copy.deepcopy(get_ur5e_wuji_cfg())
    robot_cfg.spec_fn = get_wuji_heating_plate_rotate_robot_spec
    robot_cfg.init_state.pos = (0.444005907, 0.149877146, 0.824000001)
    robot_cfg.init_state.rot = (0.689925909, 0.0, 0.0, 0.723879993)
    robot_cfg.init_state.joint_pos.update(
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -1.5707,
            "elbow": 1.5707,
            "wrist_1": 0.0,
            "wrist_2": 1.5707,
            "wrist_3": 0.0,
        }
    )

    gs_wrist_cam, gs_front_cam, gs_left_cam = make_gaussian_camera_triplet(
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        scene_parent_body="table/vention table",
        wrist_camera_name="robot/wuji:wrist_cam",
        front_pos=(-0.72, 0.42, 1.20),
        front_quat=(-0.32256003, -0.20591657, 0.49712658, 0.77872881),
        front_width=640,
        front_height=480,
        wrist_width=640,
        wrist_height=360,
        left_pos=(-5.55111512e-17, -0.95119257, 1.35968606),
        left_quat=(0.818902966, 0.573931993, 0.0, 0.0),
        left_width=640,
        left_height=480,
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )
    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": robot_cfg,
            "table": get_gaussian_hidden_table_cfg(),
            "heating_plate": get_high_grip_heating_plate_cfg(),
            "beaker": get_beaker_cfg(),
            "heating_effects": get_heating_effects_cfg(),
        },
        num_envs=1,
        env_spacing=3.0,
        spec_fn=add_scene_contact_pairs,
        sensors=(gs_wrist_cam, gs_front_cam, gs_left_cam),
    )

    observations = {
        "current_qpos": ObservationGroupCfg(
            {
                "current_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            joint_names=UR5E_JOINT_ORDER + WUJI_JOINT_ORDER,
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
        "heating_plate_qpos": ObservationGroupCfg(
            {
                "heating_plate_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "heating_plate",
                            joint_names=["knob_joint"],
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
    actions = {
        "site_pose": DifferentialIKActionCfg(
            entity_name="robot",
            actuator_names=UR5E_JOINT_ORDER,
            frame_type="body",
            frame_name=r".*right_palm_link",
            use_relative_mode=False,
            position_weight=1.0,
            orientation_weight=0.3,
        ),
        "wuji_joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=WUJI_JOINT_ORDER,
            use_default_offset=False,
            preserve_order=True,
            clip=None,
        ),
    }

    events = {
        "reset_scene": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "randomize_initial_hand_pose": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": INITIAL_ARM_JOINT_NOISE_LOW,
                "highs": INITIAL_ARM_JOINT_NOISE_HIGH,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_JOINT_ORDER,
                ),
            },
        ),
        "randomize_object_positions": EventTermCfg(
            func=randomize_heating_plate_and_beaker_positions,
            mode="reset",
            params={
                "heating_plate_cfg": SceneEntityCfg("heating_plate"),
                "beaker_cfg": SceneEntityCfg("beaker"),
            },
        ),
        "reset_heating_effects": EventTermCfg(
            func=reset_heating_effects,
            mode="reset",
            params={
                "renderer": heating_effects_renderer,
                "beaker_center_cfg": SceneEntityCfg(
                    "beaker",
                    site_names=["beaker_center"],
                ),
            },
        ),
        "reset_success_state": EventTermCfg(
            func=reset_heating_plate_success_state,
            mode="reset",
        ),
        "update_heating_effects": EventTermCfg(
            func=update_heating_effects,
            mode="step",
            params={
                "heating_plate_cfg": SceneEntityCfg(
                    "heating_plate",
                    joint_names=["knob_joint"],
                ),
                "beaker_center_cfg": SceneEntityCfg(
                    "beaker",
                    site_names=["beaker_center"],
                ),
                "renderer": heating_effects_renderer,
            },
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(
            func=envs_mdp.time_out,
            time_out=True,
        ),
        "success": TerminationTermCfg(
            func=check_success,
            params={
                "heating_plate_cfg": SceneEntityCfg(
                    "heating_plate",
                    joint_names=["knob_joint"],
                    body_names=["knob"],
                ),
                "beaker_center_cfg": SceneEntityCfg(
                    "beaker",
                    site_names=["beaker_center"],
                ),
                "palm_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[r".*right_palm_link"],
                ),
                "min_rotation": HEATING_PLATE_SUCCESS_ROTATION,
                "home_tolerance": HEATING_PLATE_SUCCESS_HOME_TOLERANCE,
                "max_plate_xy_dist": HEATING_EFFECT_MAX_PLATE_XY_DIST,
                "min_plate_height": HEATING_EFFECT_MIN_PLATE_HEIGHT,
                "max_plate_height": HEATING_EFFECT_MAX_PLATE_HEIGHT,
                "min_hand_lift": HEATING_PLATE_SUCCESS_HAND_LIFT,
            },
        ),
    }
    sim = SimulationCfg(
        njmax=8192,
        nconmax=2048,
        mujoco=MujocoCfg(
            timestep=0.002,
            impratio=10.0,
            cone="elliptic",
            iterations=10,
            multiccd=False,
        ),
    )

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        events=events,
        terminations=terminations,
        sim=sim,
        decimation=25,
        episode_length_s=HEATING_PLATE_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
