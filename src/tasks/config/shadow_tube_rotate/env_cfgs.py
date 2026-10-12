from __future__ import annotations

import copy
import warnings
from pathlib import Path

import mujoco
import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import DifferentialIKActionCfg, TendonLengthActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from assets.mjcf.misc.big_table import get_table_cfg
from assets.mjcf.object.centrifuge_50ml import CENTRIFUGE_50ML_XML_SCREW
from assets.mjcf.robot.ur5e_shadow import (
    UR5E_SHADOW_ARM_JOINT_ORDER,
    UR5E_SHADOW_HAND_JOINT_ORDER,
    UR5E_SHADOW_HAND_TENDON,
    get_ur5e_shadow_cfg,
)
from assets.mjcf.robot.ur5e_shadow_left import (
    UR5E_SHADOW_LEFT_HAND_JOINT_ORDER,
    UR5E_SHADOW_LEFT_HAND_TENDON,
    get_ur5e_shadow_left_cfg,
)
from src import DEXBIOLAB_SRC_ROOT
from src.tasks.config.gaussian_cameras import make_gaussian_camera
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.actions import OrderedJointPositionActionCfg
from src.tasks.mdp.events import reset_joints_and_tendons_by_range
from src.tasks.mdp.observations import body_pose_w, camera_rgb, joint_pos_abs
from src.tasks.success_diagnostics import (
    record_success_diagnostics,
    stable_success,
)

ABLAB_PLUGIN = DEXBIOLAB_SRC_ROOT / "libablab.so.3.6.0"
_GAUSSIAN_SCENE = gaussian_scene_for_task("Shadow-Tube-Rotate")
TABLE_TOP_HEIGHT = 0.824
UR5E_SHADOW_BASE_POS = (0.445396513, 0.489254981, TABLE_TOP_HEIGHT)
UR5E_SHADOW_BASE_ROT = (0.479716420, -0.0, 0.0, 0.877423584)
UR5E_SHADOW_TUBE_HOLDER_BASE_POS = (0.438317716, -0.535876930, TABLE_TOP_HEIGHT)
UR5E_SHADOW_TUBE_HOLDER_BASE_ROT = (0.466392159, -0.0, 0.0, 0.884578049)
UR5E_SHADOW_INIT_JOINT_POS = {
    "shoulder_pan": 2.57610,
    "shoulder_lift": -1.5707,
    "elbow": -1.5707,
    "wrist_1": -3.1415,
    "wrist_2": -1.5707,
    "wrist_3": 1.507,
}
UR5E_SHADOW_TUBE_HOLDER_INIT_JOINT_POS = {
    **UR5E_SHADOW_INIT_JOINT_POS,
    "wrist_3": UR5E_SHADOW_INIT_JOINT_POS["wrist_3"] + 3.141592653589793,
}
SHADOW_CAP_TUBE_WORLD_POS = (-0.10, -0.10, TABLE_TOP_HEIGHT)
SHADOW_CAP_TUBE_WORLD_ROT = (0.9238795325, 0.0, 0.0, 0.3826834324)
SHADOW_CAP_THREAD_LOW = -0.75
SHADOW_CAP_THREAD_HIGH = 0.75
SHADOW_CAP_ATTACHED_TUBE_PREFIX = "held_tube/"
SHADOW_CAP_ATTACHED_TUBE_FRAME_POS = (-0.032262, -0.027857, 0.095511)
SHADOW_CAP_ATTACHED_TUBE_FRAME_ROT = (0.70701391, 0.01146000, -0.70701391, 0.01146000)
SHADOW_CAP_ATTACHED_TUBE_BODY_POS = (0.000000, 0.000000, -0.058000)
SHADOW_CAP_ATTACHED_TUBE_BODY_ROT = (0.60876143, 0.00000000, 0.00000000, -0.79335334)
THREADED_TUBE_INIT_POS = (0.0, 0.0, 0.12)
THREADED_TUBE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
THREADED_TUBE_CAP_LOCAL_POS = (0.0, 0.0, 0.11965)
THREADED_TUBE_CAP_LOCAL_ROT = (0.0, 1.0, 0.0, 0.0)
THREADED_TUBE_CAP_INIT_POS = (0.0, 0.0, 0.23965)
THREADED_TUBE_CAP_INIT_ROT = (0.0, 1.0, 0.0, 0.0)
GRIPPER_CAP_PAIR_FRICTION = (1.0, 1.0, 0.005, 0.0001, 0.0001)
GRIPPER_CAP_PAIR_SOLREF = (0.002, 1.0)
GRIPPER_CAP_PAIR_SOLIMP = (0.9, 0.95, 0.001)
CAP_BODY_TO_DEXONOMY_OBJECT_POS = (0.0, 0.0, 0.00617)
CAP_BODY_TO_DEXONOMY_OBJECT_ROT = (1.0, 0.0, 0.0, 0.0)
SHADOW_CAP_PAIR_FRICTION = (3.0, 3.0, 0.03, 0.001, 0.001)
SHADOW_CAP_PAIR_CONDIM = 6
SHADOW_CAP_RIB_GEOM_PREFIX = "cap_rib_"
SHADOW_CAP_ENV_STEP_S = 0.05
SHADOW_CAP_POLICY_DONE_STEPS = 800
SHADOW_CAP_EPISODE_LENGTH_S = 4000 * SHADOW_CAP_ENV_STEP_S
SHADOW_CAP_DROPPED_MAX_HEIGHT = 0.85
SHADOW_CAP_SUCCESS_MIN_CAP_TUBE_DIST = 0.135
SHADOW_CAP_SUCCESS_MAX_PALM_DIST = 0.16
SHADOW_CAP_TUBE_HOLDER_GRASP_JOINT_POS = {
    "hand:lh_THJ5": 0.294000,
    "hand:lh_THJ4": 0.893000,
    "hand:lh_THJ3": 0.161000,
    "hand:lh_THJ2": 0.698000,
    "hand:lh_THJ1": 1.041000,
    "hand:lh_FFJ4": -0.022000,
    "hand:lh_FFJ3": 0.875000,
    "hand:lh_FFJ2": 1.069500,
    "hand:lh_FFJ1": 1.069500,
    "hand:lh_MFJ4": -0.001000,
    "hand:lh_MFJ3": 0.912000,
    "hand:lh_MFJ2": 1.101000,
    "hand:lh_MFJ1": 1.101000,
    "hand:lh_RFJ4": 0.028000,
    "hand:lh_RFJ3": 1.134000,
    "hand:lh_RFJ2": 1.133000,
    "hand:lh_RFJ1": 1.133000,
    "hand:lh_LFJ5": 0.000000,
    "hand:lh_LFJ4": -0.349000,
    "hand:lh_LFJ3": 0.949000,
    "hand:lh_LFJ2": 1.133000,
    "hand:lh_LFJ1": 1.133000,
}
SHADOW_CAP_ARM_RESET_JOINT_LOWS = (
    2.513276000,
    -1.570700049,
    -1.570700049,
    -3.141499996,
    -1.570700049,
    1.068142300,
)
SHADOW_CAP_ARM_RESET_JOINT_HIGHS = (
    3.078763100,
    -1.445133700,
    -1.382260000,
    -2.890267400,
    -1.193806100,
    1.506999969,
)


def quat_mul_wxyz(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def quat_conj_wxyz(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (quat[0], -quat[1], -quat[2], -quat[3])


def quat_rotate_wxyz(
    quat: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = quat_mul_wxyz(quat_mul_wxyz(quat, (0.0, *vec)), quat_conj_wxyz(quat))
    return (rotated[1], rotated[2], rotated[3])


def _compose_pose_wxyz(
    lhs_pos: tuple[float, float, float],
    lhs_quat: tuple[float, float, float, float],
    rhs_pos: tuple[float, float, float],
    rhs_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rotated_rhs_pos = quat_rotate_wxyz(lhs_quat, rhs_pos)
    pos = (
        lhs_pos[0] + rotated_rhs_pos[0],
        lhs_pos[1] + rotated_rhs_pos[1],
        lhs_pos[2] + rotated_rhs_pos[2],
    )
    quat = quat_mul_wxyz(lhs_quat, rhs_quat)
    return pos, quat


def quat_mul_torch_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_rotate_torch_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((*vec.shape[:-1], 1), device=vec.device, dtype=vec.dtype)
    vec_quat = torch.cat((zeros, vec), dim=-1)
    quat_conj = torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)
    rotated = quat_mul_torch_wxyz(quat_mul_torch_wxyz(quat, vec_quat), quat_conj)
    return rotated[..., 1:]


def compose_pose_torch_wxyz(
    lhs_pos: torch.Tensor,
    lhs_quat: torch.Tensor,
    rhs_pos: torch.Tensor,
    rhs_quat: torch.Tensor,
) -> torch.Tensor:
    rhs_pos_w = quat_rotate_torch_wxyz(lhs_quat, rhs_pos)
    quat = quat_mul_torch_wxyz(lhs_quat, rhs_quat)
    return torch.cat((lhs_pos + rhs_pos_w, quat), dim=-1)


SHADOW_CAP_TUBE_BODY_INIT_POS, SHADOW_CAP_TUBE_BODY_INIT_ROT = _compose_pose_wxyz(
    SHADOW_CAP_TUBE_WORLD_POS,
    SHADOW_CAP_TUBE_WORLD_ROT,
    THREADED_TUBE_INIT_POS,
    THREADED_TUBE_INIT_ROT,
)
SHADOW_CAP_TUBE_CAP_INIT_POS, SHADOW_CAP_TUBE_CAP_INIT_ROT = _compose_pose_wxyz(
    SHADOW_CAP_TUBE_WORLD_POS,
    SHADOW_CAP_TUBE_WORLD_ROT,
    THREADED_TUBE_CAP_INIT_POS,
    THREADED_TUBE_CAP_INIT_ROT,
)


def get_threaded_tube_body_fixed_cap_free_spec() -> mujoco.MjSpec:
    mujoco.mj_loadPluginLibrary(str(ABLAB_PLUGIN))
    spec = mujoco.MjSpec.from_file(str(CENTRIFUGE_50ML_XML_SCREW))

    body = spec.body("centrifuge_50ml_screw_body")
    if body is None:
        raise ValueError("Missing body in threaded tube spec: centrifuge_50ml_screw_body")
    body.pos = THREADED_TUBE_INIT_POS
    body.quat = THREADED_TUBE_INIT_ROT

    cap = spec.body("centrifuge_50ml_screw_cap")
    if cap is None:
        raise ValueError("Missing cap body in threaded tube spec: centrifuge_50ml_screw_cap")
    cap.pos = THREADED_TUBE_CAP_INIT_POS
    cap.quat = THREADED_TUBE_CAP_INIT_ROT
    for joint in list(cap.joints):
        spec.delete(joint)
    cap.add_joint(
        name="centrifuge_50ml_screw_cap",
        type=mujoco.mjtJoint.mjJNT_FREE,
    )
    return spec


def get_shadow_tube_rotate_tube_spec() -> mujoco.MjSpec:
    spec = get_threaded_tube_body_fixed_cap_free_spec()
    set_shadow_tube_rotate_thread_turns(spec)

    body = spec.body("centrifuge_50ml_screw_body")
    if body is None:
        raise ValueError("Missing body in threaded tube spec: centrifuge_50ml_screw_body")
    body.pos = SHADOW_CAP_TUBE_BODY_INIT_POS
    body.quat = SHADOW_CAP_TUBE_BODY_INIT_ROT

    cap = spec.body("centrifuge_50ml_screw_cap")
    if cap is None:
        raise ValueError("Missing cap body in threaded tube spec: centrifuge_50ml_screw_cap")
    cap.pos = SHADOW_CAP_TUBE_CAP_INIT_POS
    cap.quat = SHADOW_CAP_TUBE_CAP_INIT_ROT
    return spec


def set_shadow_tube_rotate_thread_turns(spec: mujoco.MjSpec) -> None:
    """Shorten the SDF thread span only for the UR5e Shadow cap environment."""
    for plugin_name in ("bolt", "nut"):
        plugin = spec.plugin(plugin_name)
        if plugin is None:
            raise ValueError(f"Missing thread plugin in threaded tube spec: {plugin_name}")
        config = dict(plugin.config)
        config["low"] = str(SHADOW_CAP_THREAD_LOW)
        config["high"] = str(SHADOW_CAP_THREAD_HIGH)
        plugin.config = config


def get_shadow_tube_rotate_tube_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_shadow_tube_rotate_tube_spec,
    )


def get_ur5e_shadow_with_attached_tube_spec() -> mujoco.MjSpec:
    spec = get_ur5e_shadow_left_cfg().spec_fn()
    tube_spec = mujoco.MjSpec.from_file(str(CENTRIFUGE_50ML_XML_SCREW))
    set_shadow_tube_rotate_thread_turns(tube_spec)

    for joint in list(tube_spec.joints):
        tube_spec.delete(joint)

    for geom in tube_spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0

    tube_body = tube_spec.body("centrifuge_50ml_screw_body")
    if tube_body is None:
        raise ValueError("Missing body in centrifuge tube spec: centrifuge_50ml_screw_body")
    tube_body.pos = SHADOW_CAP_ATTACHED_TUBE_BODY_POS
    tube_body.quat = SHADOW_CAP_ATTACHED_TUBE_BODY_ROT
    cap_body = tube_spec.body("centrifuge_50ml_screw_cap")
    if cap_body is None:
        raise ValueError("Missing cap body in centrifuge tube spec: centrifuge_50ml_screw_cap")
    tube_spec.delete(cap_body)

    palm_body = spec.body("hand:lh_palm")
    if palm_body is None:
        raise ValueError("Missing Shadow hand palm body: hand:lh_palm")
    frame = palm_body.add_frame(
        pos=SHADOW_CAP_ATTACHED_TUBE_FRAME_POS,
        quat=SHADOW_CAP_ATTACHED_TUBE_FRAME_ROT,
    )
    spec.attach(tube_spec, prefix=SHADOW_CAP_ATTACHED_TUBE_PREFIX, suffix="", frame=frame)
    return spec


def get_shadow_tube_rotate_free_cap_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(CENTRIFUGE_50ML_XML_SCREW))
    set_shadow_tube_rotate_thread_turns(spec)

    tube_body = spec.body("centrifuge_50ml_screw_body")
    if tube_body is None:
        raise ValueError("Missing body in centrifuge tube spec: centrifuge_50ml_screw_body")
    spec.delete(tube_body)

    cap_body = spec.body("centrifuge_50ml_screw_cap")
    if cap_body is None:
        raise ValueError("Missing cap body in centrifuge tube spec: centrifuge_50ml_screw_cap")
    cap_body.pos = SHADOW_CAP_TUBE_CAP_INIT_POS
    cap_body.quat = SHADOW_CAP_TUBE_CAP_INIT_ROT
    for joint in list(cap_body.joints):
        spec.delete(joint)
    cap_body.add_joint(
        name="centrifuge_50ml_screw_cap",
        type=mujoco.mjtJoint.mjJNT_FREE,
    )
    return spec


def get_shadow_tube_rotate_free_cap_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_shadow_tube_rotate_free_cap_spec,
    )


def get_ur5e_shadow_with_attached_tube_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=copy.deepcopy(get_ur5e_shadow_left_cfg().init_state),
        spec_fn=get_ur5e_shadow_with_attached_tube_spec,
        articulation=get_ur5e_shadow_left_cfg().articulation,
    )


def reset_shadow_tube_rotate_tube_cap_pose(env, env_ids) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    env.sim.forward()

    joint_name = "tube_cap/centrifuge_50ml_screw_cap"
    joint_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Missing threaded tube cap free joint in compiled model: {joint_name}")
    tube_body_name = "tube_holder_robot/held_tube/centrifuge_50ml_screw_body"
    tube_body_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, tube_body_name)
    if tube_body_id < 0:
        raise RuntimeError(f"Missing attached tube body in compiled model: {tube_body_name}")

    qadr = int(env.sim.mj_model.jnt_qposadr[joint_id])
    dadr = int(env.sim.mj_model.jnt_dofadr[joint_id])
    qpos_ids = torch.arange(qadr, qadr + 7, device=env.device, dtype=torch.long)
    qvel_ids = torch.arange(dadr, dadr + 6, device=env.device, dtype=torch.long)

    tube_pos = env.sim.data.xpos[env_ids, tube_body_id]
    tube_quat = env.sim.data.xquat[env_ids, tube_body_id]
    cap_local_pos = torch.tensor(
        THREADED_TUBE_CAP_LOCAL_POS,
        device=env.device,
        dtype=env.sim.data.qpos.dtype,
    ).unsqueeze(0).repeat(env_ids.numel(), 1)
    cap_local_quat = torch.tensor(
        THREADED_TUBE_CAP_LOCAL_ROT,
        device=env.device,
        dtype=env.sim.data.qpos.dtype,
    ).unsqueeze(0).repeat(env_ids.numel(), 1)
    cap_pose = compose_pose_torch_wxyz(tube_pos, tube_quat, cap_local_pos, cap_local_quat)
    env.sim.data.qpos[env_ids[:, None], qpos_ids[None, :]] = cap_pose
    env.sim.data.qvel[env_ids[:, None], qvel_ids[None, :]] = 0.0

    if hasattr(env.sim, "mj_data"):
        env.sim.mj_data.qpos[qadr : qadr + 7] = cap_pose[0].detach().cpu().numpy()
        env.sim.mj_data.qvel[dadr : dadr + 6] = 0.0
    env.sim.forward()


def check_success(
    env,
    cap_cfg: SceneEntityCfg,
    tube_cfg: SceneEntityCfg,
    min_cap_tube_dist: float,
    min_cap_height: float,
    stability_steps: int,
) -> torch.Tensor:
    cap = env.scene[cap_cfg.name]
    tube = env.scene[tube_cfg.name]

    cap_body_ids = cap_cfg.body_ids
    tube_body_ids = tube_cfg.body_ids
    assert cap_body_ids is not None, "Please specify cap body_names in SceneEntityCfg"
    assert tube_body_ids is not None, "Please specify tube body_names in SceneEntityCfg"

    cap_pos = cap.data.body_link_pos_w[:, cap_body_ids[0]]
    tube_pos = tube.data.body_link_pos_w[:, tube_body_ids[0]]
    cap_tube_dist = torch.linalg.norm(cap_pos - tube_pos, dim=-1)
    cap_removed = cap_tube_dist > float(min_cap_tube_dist)
    cap_lifted = cap_pos[:, 2] > float(min_cap_height)
    object_complete = cap_removed & cap_lifted
    success, stable_count = stable_success(
        env,
        object_complete,
        state_attr="shadow_tube_rotate_stable_steps",
        required_steps=stability_steps,
    )
    record_success_diagnostics(
        env,
        object_complete=object_complete,
        full_success=success,
        components={
            "cap_removed": cap_removed,
            "cap_lifted": cap_lifted,
        },
        metrics={
            "cap_tube_dist_m": cap_tube_dist,
            "cap_height_m": cap_pos[:, 2],
            "stable_steps": stable_count,
        },
        thresholds={
            "min_cap_tube_dist_m": min_cap_tube_dist,
            "min_cap_height_m": min_cap_height,
            "required_stability_steps": stability_steps,
        },
    )
    return success


def check_cap_dropped(
    env,
    cap_cfg: SceneEntityCfg,
    max_height: float,
) -> torch.Tensor:
    cap = env.scene[cap_cfg.name]
    cap_body_ids = cap_cfg.body_ids
    assert cap_body_ids is not None, "Please specify cap body_names in SceneEntityCfg"

    cap_height = cap.data.body_link_pos_w[:, cap_body_ids[0], 2]
    return cap_height <= float(max_height)


def name_shadow_tube_rotate_rib_geoms(spec: mujoco.MjSpec) -> list[str]:
    total_cap_geom = spec.geom("tube/total_cap_cylinder")
    if total_cap_geom is None:
        total_cap_geom = spec.geom("total_cap_cylinder")
    if total_cap_geom is not None:
        total_cap_geom.contype = 0
        total_cap_geom.conaffinity = 0

    cap_body = spec.body("tube/centrifuge_50ml_screw_cap")
    if cap_body is None:
        cap_body = spec.body("centrifuge_50ml_screw_cap")
    if cap_body is None:
        cap_body = next(
            (
                body
                for body in spec.bodies
                if body.name.endswith("centrifuge_50ml_screw_cap")
            ),
            None,
        )
    if cap_body is None:
        return []

    name_prefix = cap_body.name.removesuffix("centrifuge_50ml_screw_cap")
    rib_geom_names = []
    for geom in cap_body.geoms:
        is_cap_rib = (
            geom.type == mujoco.mjtGeom.mjGEOM_BOX
            and int(geom.group) == 3
            and int(geom.contype) == 2
            and int(geom.conaffinity) == 4
        )
        if not is_cap_rib:
            continue

        geom.name = f"{name_prefix}{SHADOW_CAP_RIB_GEOM_PREFIX}{len(rib_geom_names):02d}"
        rib_geom_names.append(geom.name)

    return rib_geom_names


def is_shadow_collision_geom_name(name: str) -> bool:
    short_name = name.rsplit("/", 1)[-1]
    return (
        name.startswith("robot/rh_")
        or name.startswith("robot/hand:rh_")
        or short_name.startswith("rh_")
        or short_name.startswith("hand:rh_")
    ) and "_collision_" in short_name


def is_shadow_body_name(name: str) -> bool:
    short_name = name.rsplit("/", 1)[-1]
    return (
        short_name.startswith("rh_")
        or short_name.startswith("hand:rh_")
        or short_name.startswith("lh_")
        or short_name.startswith("hand:lh_")
    )


def name_ur5e_shadow_collision_geoms(body: mujoco.MjsBody) -> None:
    if is_shadow_body_name(body.name):
        collision_idx = 0
        for geom in body.geoms:
            if (
                not geom.name
                and int(geom.group) == 3
                and (int(geom.contype) or int(geom.conaffinity))
            ):
                geom.name = f"{body.name}_collision_{collision_idx:02d}"
                collision_idx += 1

    for child in body.bodies:
        name_ur5e_shadow_collision_geoms(child)


def body_prefix_for_suffix(spec: mujoco.MjSpec, body_suffix: str) -> str | None:
    body = next((body for body in spec.bodies if body.name.endswith(body_suffix)), None)
    if body is None:
        return None
    return body.name.removesuffix(body_suffix)


def add_threaded_tube_ur5e_shadow_contact_pairs(spec: mujoco.MjSpec) -> None:
    for body in spec.bodies:
        name_ur5e_shadow_collision_geoms(body)

    cap_contact_geoms = name_shadow_tube_rotate_rib_geoms(spec)
    geom_names = {geom.name for geom in spec.geoms}
    shadow_collision_geoms = sorted(
        name for name in geom_names if is_shadow_collision_geom_name(name)
    )

    pair_defs = []
    cap_prefix = body_prefix_for_suffix(spec, "centrifuge_50ml_screw_cap")
    tube_prefix = body_prefix_for_suffix(spec, "centrifuge_50ml_screw_body")
    if cap_prefix is not None and tube_prefix is not None:
        pair_defs.extend(
            [
                (
                    f"{cap_prefix}nut",
                    f"{tube_prefix}bolt",
                    GRIPPER_CAP_PAIR_FRICTION,
                    (0.002, 1.0),
                    None,
                    None,
                ),
                (
                    f"{cap_prefix}nut",
                    f"{tube_prefix}body_cyl",
                    (0.01, 0.01, 0.005, 0.0001, 0.0001),
                    (0.002, 1.0),
                    (0.9, 0.95, 0.001),
                    None,
                ),
            ]
        )
    pair_defs.extend(
        (
            cap_geom_name,
            geom_name,
            SHADOW_CAP_PAIR_FRICTION,
            GRIPPER_CAP_PAIR_SOLREF,
            GRIPPER_CAP_PAIR_SOLIMP,
            SHADOW_CAP_PAIR_CONDIM,
        )
        for cap_geom_name in cap_contact_geoms
        for geom_name in shadow_collision_geoms
    )

    for pair_def in pair_defs:
        if len(pair_def) == 5:
            geom1, geom2, friction, solref, solimp = pair_def
            condim = None
        else:
            geom1, geom2, friction, solref, solimp, condim = pair_def
        if geom1 not in geom_names or geom2 not in geom_names:
            warnings.warn(
                f"Skip threaded tube/UR5e Shadow contact pair ({geom1}, {geom2}): "
                "geom missing in scene.",
                stacklevel=2,
            )
            continue

        if solimp is not None and len(solimp) == 3:
            solimp = (*solimp, 0.5, 2.0)
        spec.add_pair(
            geomname1=geom1,
            geomname2=geom2,
            friction=friction,
            solref=solref,
            solimp=solimp,
            condim=condim,
        )


def make_shadow_tube_rotate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    del play

    robot_cfg = copy.deepcopy(get_ur5e_shadow_cfg())
    robot_cfg.init_state.pos = UR5E_SHADOW_BASE_POS
    robot_cfg.init_state.rot = UR5E_SHADOW_BASE_ROT
    robot_cfg.init_state.joint_pos.update(UR5E_SHADOW_INIT_JOINT_POS)
    tube_holder_robot_cfg = get_ur5e_shadow_with_attached_tube_cfg()
    tube_holder_robot_cfg.init_state.pos = UR5E_SHADOW_TUBE_HOLDER_BASE_POS
    tube_holder_robot_cfg.init_state.rot = UR5E_SHADOW_TUBE_HOLDER_BASE_ROT
    tube_holder_robot_cfg.init_state.joint_pos.update(
        UR5E_SHADOW_TUBE_HOLDER_INIT_JOINT_POS
    )
    tube_holder_robot_cfg.init_state.joint_pos.update(
        SHADOW_CAP_TUBE_HOLDER_GRASP_JOINT_POS
    )
    gs_tube_holder_wrist_cam = make_gaussian_camera(
        name="gs_tube_holder_wrist_cam_rgb",
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention big table",
        parent_body="tube_holder_robot/hand:lh_forearm",
        pos=(0.0457487621, -0.209015521, 0.169722361),
        quat=(0.22788814, 0.97310022, -0.0166475917, -0.0294247336),
        fovy=58.0083322,
        width=640,
        height=480,
    )
    gs_wrist_cam = make_gaussian_camera(
        name="gs_wrist_cam_rgb",
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention big table",
        camera_name="robot/wrist_cam",
        parent_body="robot/wrist_3_link",
        fovy=58.0,
        width=320,
        height=240,
    )
    gs_front_cam = make_gaussian_camera(
        name="gs_front_cam_rgb",
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention big table",
        parent_body="table/vention big table",
        pos=(-1.2, 0.0, 1.45),
        quat=(0.579, 0.406, -0.406, -0.579),
        fovy=45.0,
        width=320,
        height=240,
    )

    scene = SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="plane",
        ),
        entities={
            "robot": robot_cfg,
            "tube_holder_robot": tube_holder_robot_cfg,
            "tube_cap": get_shadow_tube_rotate_free_cap_cfg(),
            "table": get_table_cfg(),
        },
        spec_fn=add_threaded_tube_ur5e_shadow_contact_pairs,
        num_envs=1,
        env_spacing=3.0,
        sensors=(gs_tube_holder_wrist_cam, gs_wrist_cam, gs_front_cam),
    )

    robot_joint_cfg = SceneEntityCfg(
        "robot",
        joint_names=UR5E_SHADOW_ARM_JOINT_ORDER + UR5E_SHADOW_HAND_JOINT_ORDER,
        preserve_order=True,
    )
    tube_holder_robot_joint_cfg = SceneEntityCfg(
        "tube_holder_robot",
        joint_names=UR5E_SHADOW_ARM_JOINT_ORDER + UR5E_SHADOW_LEFT_HAND_JOINT_ORDER,
        preserve_order=True,
    )
    cap_body_cfg = SceneEntityCfg(
        "tube_cap",
        body_names=["centrifuge_50ml_screw_cap"],
    )
    tube_body_cfg = SceneEntityCfg(
        "tube_holder_robot",
        body_names=[r".*centrifuge_50ml_screw_body"],
    )

    observations = {
        "current_qpos": ObservationGroupCfg(
            {
                "current_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={"asset_cfg": robot_joint_cfg},
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tube_holder_qpos": ObservationGroupCfg(
            {
                "tube_holder_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={"asset_cfg": tube_holder_robot_joint_cfg},
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
                            body_names=["hand:rh_palm"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tube_holder_palm_body_pose_w": ObservationGroupCfg(
            {
                "tube_holder_palm_body_pose_w": ObservationTermCfg(
                    func=body_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "tube_holder_robot",
                            body_names=["hand:lh_palm"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "attached_tube_body_pose_w": ObservationGroupCfg(
            {
                "attached_tube_body_pose_w": ObservationTermCfg(
                    func=body_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "tube_holder_robot",
                            body_names=["centrifuge_50ml_screw_body"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tube_cap_pose_w": ObservationGroupCfg(
            {
                "tube_cap_pose_w": ObservationTermCfg(
                    func=body_pose_w,
                    params={"asset_cfg": cap_body_cfg},
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tube_body_pose_w": ObservationGroupCfg(
            {
                "tube_body_pose_w": ObservationTermCfg(
                    func=body_pose_w,
                    params={"asset_cfg": tube_body_cfg},
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "gs_tube_holder_wrist_cam_rgb": ObservationGroupCfg(
            {
                "gs_tube_holder_wrist_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_tube_holder_wrist_cam_rgb"},
                ),
            },
            enable_corruption=False,
        ),
        "gs_wrist_cam_rgb": ObservationGroupCfg(
            {
                "gs_wrist_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_wrist_cam_rgb"},
                ),
            },
            enable_corruption=False,
        ),
        "gs_front_cam_rgb": ObservationGroupCfg(
            {
                "gs_front_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_front_cam_rgb"},
                ),
            },
            enable_corruption=False,
        ),
    }

    actions = {
        "site_pose": DifferentialIKActionCfg(
            entity_name="robot",
            actuator_names=UR5E_SHADOW_ARM_JOINT_ORDER,
            frame_type="body",
            frame_name=r".*rh_palm",
            use_relative_mode=False,
            position_weight=1,
            orientation_weight=0.3,
        ),
        "joint_pos": OrderedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=UR5E_SHADOW_HAND_JOINT_ORDER,
            use_default_offset=False,
            preserve_order=True,
            clip=None,
        ),
        "tendon": TendonLengthActionCfg(
            entity_name="robot",
            actuator_names=UR5E_SHADOW_HAND_TENDON,
            preserve_order=True,
            clip=None,
            transmission_type=TransmissionType.TENDON,
        ),
        "tube_holder_site_pose": DifferentialIKActionCfg(
            entity_name="tube_holder_robot",
            actuator_names=UR5E_SHADOW_ARM_JOINT_ORDER,
            frame_type="body",
            frame_name=r".*lh_palm",
            use_relative_mode=False,
            position_weight=1,
            orientation_weight=1,
        ),
        "tube_holder_joint_pos": OrderedJointPositionActionCfg(
            entity_name="tube_holder_robot",
            actuator_names=UR5E_SHADOW_LEFT_HAND_JOINT_ORDER,
            use_default_offset=False,
            preserve_order=True,
            clip=None,
        ),
        "tube_holder_tendon": TendonLengthActionCfg(
            entity_name="tube_holder_robot",
            actuator_names=UR5E_SHADOW_LEFT_HAND_TENDON,
            preserve_order=True,
            clip=None,
            transmission_type=TransmissionType.TENDON,
        ),
    }

    events = {
        "reset_scene": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "reset_robot_arm_randomization": EventTermCfg(
            func=reset_joints_and_tendons_by_range,
            mode="reset",
            params={
                "joint_lows": SHADOW_CAP_ARM_RESET_JOINT_LOWS,
                "joint_highs": SHADOW_CAP_ARM_RESET_JOINT_HIGHS,
                "joint_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_SHADOW_ARM_JOINT_ORDER,
                    preserve_order=True,
                ),
            },
        ),
        "reset_tube_holder_robot_arm_randomization": EventTermCfg(
            func=reset_joints_and_tendons_by_range,
            mode="reset",
            params={
                "joint_lows": SHADOW_CAP_ARM_RESET_JOINT_LOWS,
                "joint_highs": SHADOW_CAP_ARM_RESET_JOINT_HIGHS,
                "joint_asset_cfg": SceneEntityCfg(
                    "tube_holder_robot",
                    joint_names=UR5E_SHADOW_ARM_JOINT_ORDER,
                    preserve_order=True,
                ),
                "forward_after_write": True,
            },
        ),
        "reset_shadow_tube_rotate_tube_cap_pose": EventTermCfg(
            func=reset_shadow_tube_rotate_tube_cap_pose,
            mode="reset",
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
                "cap_cfg": cap_body_cfg,
                "tube_cfg": tube_body_cfg,
                "min_cap_tube_dist": SHADOW_CAP_SUCCESS_MIN_CAP_TUBE_DIST,
                "min_cap_height": SHADOW_CAP_DROPPED_MAX_HEIGHT,
                "stability_steps": 3,
            },
        ),
        "cap_dropped": TerminationTermCfg(
            func=check_cap_dropped,
            params={
                "cap_cfg": cap_body_cfg,
                "max_height": SHADOW_CAP_DROPPED_MAX_HEIGHT,
            },
        ),
    }

    sim = SimulationCfg(
        nconmax=2048,
        njmax=8192,
        mujoco=MujocoCfg(
            timestep=0.002,
            impratio=10,
            cone="elliptic",
            ccd_iterations=500,
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
        viewer=ViewerConfig(azimuth=0.0),
        decimation=25,
        episode_length_s=SHADOW_CAP_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )


__all__ = [
    "CAP_BODY_TO_DEXONOMY_OBJECT_POS",
    "CAP_BODY_TO_DEXONOMY_OBJECT_ROT",
    "SHADOW_CAP_EPISODE_LENGTH_S",
    "SHADOW_CAP_DROPPED_MAX_HEIGHT",
    "SHADOW_CAP_SUCCESS_MAX_PALM_DIST",
    "SHADOW_CAP_SUCCESS_MIN_CAP_TUBE_DIST",
    "THREADED_TUBE_CAP_INIT_POS",
    "THREADED_TUBE_CAP_INIT_ROT",
    "add_threaded_tube_ur5e_shadow_contact_pairs",
    "check_cap_dropped",
    "check_success",
    "get_shadow_tube_rotate_free_cap_cfg",
    "get_shadow_tube_rotate_tube_cfg",
    "get_ur5e_shadow_with_attached_tube_cfg",
    "make_shadow_tube_rotate_env_cfg",
    "reset_shadow_tube_rotate_tube_cap_pose",
]
