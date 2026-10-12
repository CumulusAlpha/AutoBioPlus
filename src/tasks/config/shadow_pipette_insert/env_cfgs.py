import copy
from pathlib import Path

import mujoco
import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityCfg
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

from assets.mjcf.misc.big_table import get_table_cfg
from assets.mjcf.robot.ur5e_shadow import (
    UR5E_SHADOW_ARM_JOINT_ORDER,
    UR5E_SHADOW_HAND_JOINT_ORDER,
    UR5E_SHADOW_HAND_TENDON,
    get_ur5e_shadow_cfg,
)
from assets.mjcf.robot.ur5e_shadow import (
    get_spec as get_ur5e_shadow_spec,
)
from src import MJCF_ROOT
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.actions import OrderedJointPositionActionCfg
from src.tasks.mdp.events import add_noise_to_joints
from src.tasks.mdp.observations import (
    body_pose_w,
    camera_rgb,
    joint_pos_abs,
    root_pose_w,
    site_pose_w,
)
from src.tasks.success_diagnostics import record_success_diagnostics

PIPETTE_XML = MJCF_ROOT / "object" / "pipette.gen.xml"
PIPETTE_STAND_XML = MJCF_ROOT / "misc" / "pipette_stand.xml"
TIP_BOX_XML = MJCF_ROOT / "object" / "tip_box.gen.xml"
PIPETTE_TIP_XML = MJCF_ROOT / "object" / "pipette_tip.gen.xml"
PIPETTE_TIP_WELD_NAME = "pipette_tip_weld"
PIPETTE_TIP_WELD_SITE = "tip_weld_site"
PIPETTE_TIP_ATTACH_XY_TOL = 0.003
PIPETTE_TIP_ATTACH_Z_GAP = 0.002
PIPETTE_TIP_ATTACH_Z_TOL = 0.002

IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
PIPETTE_STAND_POS = (0.097, -0.33, 1.056)
TIP_BOX_POS = (0.0, -0.33, 0.824)
TIP_IN_BOX_OFFSET = (-0.02, -0.012, 0.0499)
TIP_IN_BOX_POS = tuple(a + b for a, b in zip(TIP_BOX_POS, TIP_IN_BOX_OFFSET))
TIP_BOX_SLOT_XS = (-0.020, -0.012, -0.004, 0.004, 0.012, 0.020)
TIP_BOX_SLOT_YS = (-0.012, -0.004, 0.004, 0.012)
TIP_BOX_SLOT_Z = TIP_IN_BOX_OFFSET[2]
PIPETTE_INIT_POS = (0.15, -0.33, 0.906)
PIPETTE_INIT_ROT = (-1, 0.0, 0.0, 0)
# Sum of the current grasp, transfer, attach, and final-lift phase lengths.
SHADOW_TIP_POLICY_DONE_STEPS = 384
SHADOW_TIP_ENV_STEP_S = 0.05
SHADOW_TIP_EPISODE_LENGTH_S = max(30.0, SHADOW_TIP_POLICY_DONE_STEPS * SHADOW_TIP_ENV_STEP_S * 2.0)

assert PIPETTE_XML.exists(), f"Missing MJCF: {PIPETTE_XML}"
assert PIPETTE_STAND_XML.exists(), f"Missing MJCF: {PIPETTE_STAND_XML}"
assert TIP_BOX_XML.exists(), f"Missing MJCF: {TIP_BOX_XML}"
assert PIPETTE_TIP_XML.exists(), f"Missing MJCF: {PIPETTE_TIP_XML}"

_GAUSSIAN_SCENE = gaussian_scene_for_task("Shadow-Pipette-Insert")


def get_shadow_pipette_insert_robot_spec() -> mujoco.MjSpec:
    spec = get_ur5e_shadow_spec()
    wrist_cam = spec.camera("wrist_cam")
    wrist_cam.pos = (-0.14266717, 0.0168906141, -0.10597364)
    wrist_cam.quat = (-0.0627922176, 0.688557002, 0.720658193, 0.050972163)
    wrist_cam.fovy = 58.0
    return spec


def get_pipette_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PIPETTE_XML))

    attached_tip = spec.body("tip/pipette_tip")
    if attached_tip is not None:
        spec.delete(attached_tip)

    pipette_body = spec.body("pipette")
    pipette_body.add_site(
        name="tip_attach_site",
        pos=(0.0, 0.0, -0.003),
        quat=IDENTITY_QUAT_WXYZ,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.003,),
        group=5,
        rgba=(0.0, 1.0, 0.0, 1.0),
    )
    pipette_body.add_site(
        name="tip_attach_axis_site",
        pos=(0.0, 0.0, -0.025),
        quat=IDENTITY_QUAT_WXYZ,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.002,),
        group=5,
        rgba=(0.0, 0.2, 1.0, 1.0),
    )
    pipette_body.add_freejoint(name="pipette_freejoint")
    # enable_gravity_compensation(pipette_body)

    return spec


def get_pipette_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=PIPETTE_INIT_POS,
            rot=PIPETTE_INIT_ROT,
        ),
        spec_fn=get_pipette_spec,
    )


def get_pipette_stand_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PIPETTE_STAND_XML))
    return spec


def get_pipette_stand_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=PIPETTE_STAND_POS,
            rot=IDENTITY_QUAT_WXYZ,
        ),
        spec_fn=get_pipette_stand_spec,
    )


def get_tip_box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(TIP_BOX_XML))
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.contype = 0
            geom.conaffinity = 0
            continue
        if geom.contype != 0 or geom.conaffinity != 0:
            geom.condim = 6
            geom.friction = (2.0, 0.05, 0.005)
            geom.solref = (0.01, 1.0)
    return spec


def get_tip_box_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=TIP_BOX_POS,
            rot=IDENTITY_QUAT_WXYZ,
        ),
        spec_fn=get_tip_box_spec,
    )


def get_pipette_tip_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PIPETTE_TIP_XML))

    tip_body = spec.body("pipette_tip")
    tip_body.add_site(
        name=PIPETTE_TIP_WELD_SITE,
        pos=(0.0, 0.0, 0.0),
        quat=IDENTITY_QUAT_WXYZ,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.003,),
        group=5,
        rgba=(1.0, 0.0, 0.0, 1.0),
    )
    tip_body.add_site(
        name="tip_axis_site",
        pos=(0.0, 0.0, -0.025),
        quat=IDENTITY_QUAT_WXYZ,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.002,),
        group=5,
        rgba=(1.0, 0.8, 0.0, 1.0),
    )
    tip_body.add_freejoint(name="pipette_tip_freejoint")
    for geom in list(spec.geoms):
        if geom.meshname == "pipette_tip-visual_visual":
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 2
            tip_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=geom.meshname,
                contype=1,
                conaffinity=1,
                group=3,
            )
        elif geom.meshname == "pipette_tip-collision_tip_collision_0":
            spec.delete(geom)
        elif geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and geom.parent == tip_body:
            spec.delete(geom)

    for geom in spec.geoms:
        if geom.contype != 0 or geom.conaffinity != 0:
            geom.condim = 6
            geom.friction = (2.0, 0.05, 0.005)
            geom.solref = (0.01, 1.0)

    return spec


def get_pipette_tip_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=TIP_IN_BOX_POS,
            rot=IDENTITY_QUAT_WXYZ,
        ),
        spec_fn=get_pipette_tip_spec,
    )


# -----------------------------------------------------------------------------
# Domain randomization
# -----------------------------------------------------------------------------


def sample_tip_box_slot_offsets(
    env,
    num_envs: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    slot_offsets = torch.tensor(
        [(x, y, TIP_BOX_SLOT_Z) for y in TIP_BOX_SLOT_YS for x in TIP_BOX_SLOT_XS],
        device=env.device,
        dtype=dtype,
    )
    slot_ids = torch.randint(
        low=0,
        high=slot_offsets.shape[0],
        size=(num_envs,),
        device=env.device,
    )
    return slot_offsets[slot_ids]


def reset_fixed_support_and_loose_object(
    env,
    env_ids: torch.Tensor,
    support_cfg: SceneEntityCfg,
    loose_cfg: SceneEntityCfg,
    loose_rel_pos: torch.Tensor | None = None,
) -> None:
    support: Entity = env.scene[support_cfg.name]
    loose: Entity = env.scene[loose_cfg.name]

    default_support_pose = support.data.default_root_state[env_ids, :7].clone()
    default_loose_pose = loose.data.default_root_state[env_ids, :7].clone()
    dtype = default_support_pose.dtype
    env_origins = env.scene.env_origins[env_ids].to(dtype=dtype)

    support_pos = default_support_pose[:, :3] + env_origins
    support_pose = torch.cat((support_pos, default_support_pose[:, 3:7]), dim=-1)
    support.write_mocap_pose_to_sim(support_pose, env_ids=env_ids)

    if loose_rel_pos is None:
        loose_rel_pos = default_loose_pose[:, :3] - default_support_pose[:, :3]
    else:
        loose_rel_pos = loose_rel_pos.to(device=env.device, dtype=dtype)
    loose_pos = support_pos + loose_rel_pos
    loose_pose = torch.cat((loose_pos, default_loose_pose[:, 3:7]), dim=-1)

    loose.write_root_link_pose_to_sim(loose_pose, env_ids=env_ids)
    zero_vel = torch.zeros_like(loose.data.root_link_vel_w[env_ids])
    loose.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)


def reset_shadow_pipette_insert_objects(
    env,
    env_ids: torch.Tensor | None,
    pipette_cfg: SceneEntityCfg,
    pipette_stand_cfg: SceneEntityCfg,
    tip_box_cfg: SceneEntityCfg,
    tip_cfg: SceneEntityCfg,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    reset_fixed_support_and_loose_object(
        env=env,
        env_ids=env_ids,
        support_cfg=pipette_stand_cfg,
        loose_cfg=pipette_cfg,
    )
    tip_slot_offsets = sample_tip_box_slot_offsets(
        env=env,
        num_envs=env_ids.numel(),
        dtype=env.scene[tip_box_cfg.name].data.default_root_state.dtype,
    )
    reset_fixed_support_and_loose_object(
        env=env,
        env_ids=env_ids,
        support_cfg=tip_box_cfg,
        loose_cfg=tip_cfg,
        loose_rel_pos=tip_slot_offsets,
    )
    env.sim.forward()


def add_pipette_tip_weld(spec: mujoco.MjSpec) -> None:
    weld = spec.add_equality()
    weld.name = PIPETTE_TIP_WELD_NAME
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_SITE
    weld.name1 = "pipette/tip_attach_site"
    weld.name2 = f"pipette_tip/{PIPETTE_TIP_WELD_SITE}"
    weld.active = 0
    weld.solref = (0.002, 1.0)
    weld.solimp = (0.95, 0.99, 0.001, 0.5, 2.0)


def pipette_tip_attachment_candidates(
    attach_pos: torch.Tensor,
    tip_pos: torch.Tensor,
    *,
    max_xy_dist: float,
    target_z_gap: float,
    max_z_error: float,
) -> torch.Tensor:
    """Return environments where the pipette is seated on the loose tip."""
    if attach_pos.shape != tip_pos.shape or attach_pos.ndim != 2 or attach_pos.shape[1] != 3:
        raise ValueError(
            "Pipette and tip positions must both have shape (num_envs, 3), "
            f"got {tuple(attach_pos.shape)} and {tuple(tip_pos.shape)}."
        )
    delta = attach_pos - tip_pos
    xy_dist = torch.linalg.norm(delta[:, :2], dim=-1)
    z_error = torch.abs(delta[:, 2] - float(target_z_gap))
    return (xy_dist <= float(max_xy_dist)) & (z_error <= float(max_z_error))


def pipette_tip_weld_id(env) -> int:
    weld_id = mujoco.mj_name2id(
        env.sim.mj_model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        PIPETTE_TIP_WELD_NAME,
    )
    if weld_id < 0:
        raise ValueError(f"MuJoCo model has no equality {PIPETTE_TIP_WELD_NAME!r}.")
    return int(weld_id)


def set_pipette_tip_weld_active(
    env,
    env_ids: torch.Tensor,
    active: bool | torch.Tensor,
) -> None:
    weld_id = pipette_tip_weld_id(env)
    active_values = torch.as_tensor(active, device=env.device, dtype=torch.bool)
    if active_values.ndim == 0:
        active_values = active_values.expand(env_ids.numel())
    if active_values.shape != (env_ids.numel(),):
        raise ValueError(
            f"Expected {env_ids.numel()} equality values, got {tuple(active_values.shape)}."
        )
    env.sim.data.eq_active[env_ids, weld_id] = active_values

    env0 = torch.nonzero(env_ids == 0, as_tuple=False).flatten()
    if env0.numel() > 0:
        env.sim.mj_data.eq_active[weld_id] = bool(active_values[env0[0]].item())


def reset_pipette_tip_attachment(
    env,
    env_ids: torch.Tensor | None,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    set_pipette_tip_weld_active(env, env_ids, False)


def update_pipette_tip_attachment(
    env,
    env_ids: torch.Tensor | None,
    pipette_attach_site_cfg: SceneEntityCfg,
    tip_weld_site_cfg: SceneEntityCfg,
    max_xy_dist: float,
    target_z_gap: float,
    max_z_error: float,
) -> None:
    """Activate the press-fit weld independently of the policy implementation."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    pipette = env.scene[pipette_attach_site_cfg.name]
    tip = env.scene[tip_weld_site_cfg.name]
    pipette_site_ids = pipette_attach_site_cfg.site_ids
    tip_site_ids = tip_weld_site_cfg.site_ids
    assert pipette_site_ids is not None, "Please specify the pipette attach site."
    assert tip_site_ids is not None, "Please specify the tip weld site."

    weld_id = pipette_tip_weld_id(env)
    already_active = env.sim.data.eq_active[env_ids, weld_id].to(dtype=torch.bool)
    if already_active.all():
        return

    attach_pos = pipette.data.site_pose_w[env_ids, pipette_site_ids[0], :3]
    tip_pos = tip.data.site_pose_w[env_ids, tip_site_ids[0], :3]
    candidates = pipette_tip_attachment_candidates(
        attach_pos,
        tip_pos,
        max_xy_dist=max_xy_dist,
        target_z_gap=target_z_gap,
        max_z_error=max_z_error,
    )
    newly_active = (~already_active) & candidates
    if newly_active.any():
        set_pipette_tip_weld_active(
            env,
            env_ids[newly_active],
            True,
        )


def check_success(
    env,
    pipette_attach_site_cfg: SceneEntityCfg,
    tip_weld_site_cfg: SceneEntityCfg,
    max_attach_dist: float,
) -> torch.Tensor:
    pipette = env.scene[pipette_attach_site_cfg.name]
    tip = env.scene[tip_weld_site_cfg.name]
    pipette_site_ids = pipette_attach_site_cfg.site_ids
    tip_site_ids = tip_weld_site_cfg.site_ids
    assert pipette_site_ids is not None, "Please specify the pipette attach site."
    assert tip_site_ids is not None, "Please specify the tip weld site."

    attach_pos = pipette.data.site_pose_w[:, pipette_site_ids[0], :3]
    tip_pos = tip.data.site_pose_w[:, tip_site_ids[0], :3]
    attach_dist = torch.linalg.norm(attach_pos - tip_pos, dim=-1)
    weld_id = pipette_tip_weld_id(env)
    weld_active = env.sim.data.eq_active[:, weld_id].to(dtype=torch.bool)
    attached = weld_active & (attach_dist <= max_attach_dist)
    record_success_diagnostics(
        env,
        object_complete=attached,
        full_success=attached,
        components={"tip_weld_active": weld_active},
        metrics={"attach_dist_m": attach_dist},
        thresholds={"max_attach_dist_m": max_attach_dist},
    )
    return attached


def make_shadow_pipette_insert_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = copy.deepcopy(get_ur5e_shadow_cfg())
    robot_cfg.spec_fn = get_shadow_pipette_insert_robot_spec
    robot_cfg.init_state.pos = (0.438, 0.312, 0.824)
    robot_cfg.init_state.rot = (0.0, 0.0, 0.0, -1.0)
    robot_cfg.init_state.joint_pos.update(
        {
            "shoulder_pan": 2.57610,
            "shoulder_lift": -1.5707,
            "elbow": -1.5707,
            "wrist_1": -3.1415,
            "wrist_2": -1.5707,
            "wrist_3": 1.507,
        }
    )

    wrist_cam_size = (640, 360) if play else (1280, 720)
    world_cam_size = (640, 480) if play else (1280, 960)

    gs_wrist_cam, gs_front_cam, gs_left_cam = make_gaussian_camera_triplet(
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention big table",
        scene_parent_body="table/vention big table",
        wrist_camera_name="robot/wrist_cam",
        wrist_parent_body="robot/wrist_3_link",
        wrist_width=wrist_cam_size[0],
        wrist_height=wrist_cam_size[1],
        front_pos=(0.0939134551, -0.816907555, 1.03351801),
        front_quat=(0.727514923, 0.686091857, -5.61420435e-34, -1.99856244e-48),
        front_fovy=45.0,
        front_width=world_cam_size[0],
        front_height=world_cam_size[1],
        left_pos=(-0.528760561, -0.00737354209, 1.22026021),
        left_quat=(0.560549895, 0.442472389, -0.433711427, -0.549450996),
        left_width=world_cam_size[0],
        left_height=world_cam_size[1],
    )
    scene = SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="plane",
        ),
        entities={
            "robot": robot_cfg,
            "table": get_table_cfg(),
            "pipette": get_pipette_cfg(),
            "pipette_stand": get_pipette_stand_cfg(),
            "tip_box": get_tip_box_cfg(),
            "pipette_tip": get_pipette_tip_cfg(),
        },
        num_envs=1,
        env_spacing=3.0,
        sensors=(gs_wrist_cam, gs_front_cam, gs_left_cam),
        spec_fn=add_pipette_tip_weld,
    )

    observations = {
        "current_qpos": ObservationGroupCfg(
            {
                "current_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            joint_names=UR5E_SHADOW_ARM_JOINT_ORDER
                            + UR5E_SHADOW_HAND_JOINT_ORDER,
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
                            body_names=["hand:rh_palm"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "pipette_body_pose_w": ObservationGroupCfg(
            {
                "pipette_body_pose_w": ObservationTermCfg(
                    func=root_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg("pipette"),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tip_body_pose_w": ObservationGroupCfg(
            {
                "tip_body_pose_w": ObservationTermCfg(
                    func=root_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg("pipette_tip"),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "pipette_tip_attach_site_pose_w": ObservationGroupCfg(
            {
                "pipette_tip_attach_site_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "pipette",
                            site_names=["tip_attach_site"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tip_weld_site_pose_w": ObservationGroupCfg(
            {
                "tip_weld_site_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "pipette_tip",
                            site_names=[PIPETTE_TIP_WELD_SITE],
                        ),
                    },
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
        "gs_left_cam_rgb": ObservationGroupCfg(
            {
                "gs_left_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_left_cam_rgb"},
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
            orientation_weight=1,
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
    }

    events = {
        "reset_scene": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "randomize_robot_joints": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": (-0.20, -0.15, -0.15, -0.15, -0.15, -0.20),
                "highs": (0.20, 0.15, 0.15, 0.15, 0.15, 0.20),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_SHADOW_ARM_JOINT_ORDER,
                ),
            },
        ),
        "reset_objects": EventTermCfg(
            func=reset_shadow_pipette_insert_objects,
            mode="reset",
            params={
                "pipette_cfg": SceneEntityCfg("pipette"),
                "pipette_stand_cfg": SceneEntityCfg("pipette_stand"),
                "tip_box_cfg": SceneEntityCfg("tip_box"),
                "tip_cfg": SceneEntityCfg("pipette_tip"),
            },
        ),
        "reset_tip_attachment": EventTermCfg(
            func=reset_pipette_tip_attachment,
            mode="reset",
        ),
        "update_tip_attachment": EventTermCfg(
            func=update_pipette_tip_attachment,
            mode="step",
            params={
                "pipette_attach_site_cfg": SceneEntityCfg(
                    "pipette",
                    site_names=["tip_attach_site"],
                ),
                "tip_weld_site_cfg": SceneEntityCfg(
                    "pipette_tip",
                    site_names=[PIPETTE_TIP_WELD_SITE],
                ),
                "max_xy_dist": PIPETTE_TIP_ATTACH_XY_TOL,
                "target_z_gap": PIPETTE_TIP_ATTACH_Z_GAP,
                "max_z_error": PIPETTE_TIP_ATTACH_Z_TOL,
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
                "pipette_attach_site_cfg": SceneEntityCfg(
                    "pipette",
                    site_names=["tip_attach_site"],
                ),
                "tip_weld_site_cfg": SceneEntityCfg(
                    "pipette_tip",
                    site_names=[PIPETTE_TIP_WELD_SITE],
                ),
                "max_attach_dist": 0.025,
            },
        ),
    }

    sim = SimulationCfg(
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
        decimation=25,
        episode_length_s=SHADOW_TIP_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
