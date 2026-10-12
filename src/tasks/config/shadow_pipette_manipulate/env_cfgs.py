import copy
from pathlib import Path

import mujoco
import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import DifferentialIKActionCfg, TendonLengthActionCfg, JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from src import MJCF_ROOT
from assets.mjcf.misc.big_table import get_table_cfg
from assets.mjcf.robot.common import enable_gravity_compensation
from assets.mjcf.robot.ur5e_shadow import (
    UR5E_SHADOW_ARM_JOINT_ORDER,
    UR5E_SHADOW_HAND_JOINT_ORDER,
    UR5E_SHADOW_HAND_TENDON,
    get_spec as get_ur5e_shadow_spec,
    get_ur5e_shadow_cfg,
)
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.actions import OrderedJointPositionActionCfg
from src.tasks.success_diagnostics import (
    axis_tilt_from_quats,
    record_success_diagnostics,
    stable_success,
)
from src.tasks.mdp.events import (
    reset_centrifuge_on_rack,
    reset_joints_and_tendons_by_range,
)
from src.tasks.mdp.observations import (
    body_pose_w,
    camera_rgb,
    joint_pos_abs,
    root_pose_w,
    site_pose_w,
)
from src.utils.quat_util import quat_rotate_vec


PIPETTE_XML = MJCF_ROOT / "object" / "pipette.gen.xml"
PIPETTE_STAND_XML = MJCF_ROOT / "misc" / "pipette_stand.xml"
CRYOVIAL_XML = MJCF_ROOT / "object" / "cryovial_1-8ml.xml"
CRYOVIAL_RACK_XML = MJCF_ROOT / "object" / "centrifuge_plate_60well.gen.xml"

_GAUSSIAN_SCENE = gaussian_scene_for_task("Shadow-Pipette-Manipulate")

IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
PIPETTE_STAND_POS = (0.097, -0.33, 1.056)
PIPETTE_INIT_POS = (0.15, -0.33, 0.906)
PIPETTE_INIT_ROT = (-1, 0.0, 0.0, 0)
CRYOVIAL_INIT_POS = (-0.15, -0.33, 0.824)
CRYOVIAL_RACK_INIT_POS = (-0.1923, -0.3016, 0.8345)
CRYOVIAL_SLOT_ROWS = 5
CRYOVIAL_SLOT_COLS = 12
CRYOVIAL_SLOT_ROW_GAP = 0.0132
CRYOVIAL_SLOT_COL_GAP = 0.0162
CRYOVIAL_SLOT_HEIGHT = 0.0
PIPETTE_TIP_VISUAL_RGBA = (1.0, 1.0, 1.0, 1.0)
CRYOVIAL_BODY_RACK_GEOM_PREFIX = "cryovial_body_rack_collision"
RACK_CRYOVIAL_BODY_GEOM_PREFIX = "rack_cryovial_body_collision"
CRYOVIAL_RACK_PAIR_FRICTION = (2.0, 0.02, 0.001, 0.0001, 0.0001)
CRYOVIAL_RACK_PAIR_SOLREF = (0.0005, 1.0)
CRYOVIAL_RACK_PAIR_SOLIMP = (0.99, 0.999, 0.00005, 0.5, 2.0)
SIM_TIMESTEP = 0.002
SIM_DECIMATION = 25
EPISODE_LENGTH_STEPS = 400
EPISODE_LENGTH_S = 2.0 * EPISODE_LENGTH_STEPS * SIM_TIMESTEP * SIM_DECIMATION
PIPETTE_ARM_RESET_JOINT_LOWS = (
    2.199116500,
    -1.759293200,
    -1.664995000,
    -3.330090700,
    -1.884957000,
    0.502655200,
)
PIPETTE_ARM_RESET_JOINT_HIGHS = (
    2.764603600,
    -0.753982800,
    -1.288015000,
    -2.764603600,
    -1.319469900,
    1.633629400,
)
PIPETTE_HAND_RESET_JOINT_LOWS = (
    -0.9472,
    0.1,
    -0.10944,
    -0.1,
    -0.161799,
    -0.249066,
    0.0,
    -0.249066,
    0.0,
    -0.249066,
    0.0,
    0.1,
    -0.249066,
    0.0,
)
PIPETTE_HAND_RESET_JOINT_HIGHS = (
    0.9472,
    1.12173,
    0.10944,
    0.2316127,
    0.3423534,
    -0.0772105,
    0.8912274,
    -0.1270237,
    0.5891965,
    0.1818182,
    0.7198045,
    0.392699,
    0.0,
    0.7606194,
)
PIPETTE_RESET_TENDON_LOWS = (
    0.0,
    0.0,
    0.0,
    0.0,
)
PIPETTE_RESET_TENDON_HIGHS = (
    0.4398226,
    0.3612828,
    0.5497782,
    0.8325213,
)

assert PIPETTE_XML.exists(), f"Missing MJCF: {PIPETTE_XML}"
assert PIPETTE_STAND_XML.exists(), f"Missing MJCF: {PIPETTE_STAND_XML}"
assert CRYOVIAL_XML.exists(), f"Missing MJCF: {CRYOVIAL_XML}"
assert CRYOVIAL_RACK_XML.exists(), f"Missing MJCF: {CRYOVIAL_RACK_XML}"


def get_shadow_pipette_manipulate_robot_spec() -> mujoco.MjSpec:
    spec = get_ur5e_shadow_spec()
    wrist_cam = spec.camera("wrist_cam")
    wrist_cam.pos = (-0.14266717, 0.0168906141, -0.10597364)
    wrist_cam.quat = (-0.0627922176, 0.688557002, 0.720658193, 0.050972163)
    wrist_cam.fovy = 58.0
    return spec


def get_pipette_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PIPETTE_XML))

    pipette_body = spec.body("pipette")
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


def get_cryovial_spec() -> mujoco.MjSpec:
    cryovial_spec = mujoco.MjSpec.from_file(str(CRYOVIAL_XML))
    cryovial_spec.delete(cryovial_spec.body("cryovial_1-8ml_cap"))

    body_collision_idx = 0
    for body in cryovial_spec.bodies:
        if body.name != "cryovial_1-8ml_body":
            continue
        for geom in body.geoms:
            if geom.group == 2:
                geom.rgba = PIPETTE_TIP_VISUAL_RGBA
            if geom.group == 3:
                geom.name = f"{CRYOVIAL_BODY_RACK_GEOM_PREFIX}_{body_collision_idx}"
                body_collision_idx += 1

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cryovial_1-8ml")
    body.add_joint(
        name="cryovial_1-8ml_joint",
        type=mujoco.mjtJoint.mjJNT_FREE,
    )
    frame = body.add_frame()
    spec.attach(child=cryovial_spec, prefix="", frame=frame)
    return spec


def get_cryovial_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=CRYOVIAL_INIT_POS,
            rot=IDENTITY_QUAT_WXYZ,
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_cryovial_spec,
    )


def get_cryovial_rack_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(CRYOVIAL_RACK_XML))
    rack_collision_idx = 0
    for geom in spec.geoms:
        if geom.group == 3:
            geom.name = f"{RACK_CRYOVIAL_BODY_GEOM_PREFIX}_{rack_collision_idx}"
            rack_collision_idx += 1
    return spec


def get_cryovial_rack_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=CRYOVIAL_RACK_INIT_POS,
            rot=IDENTITY_QUAT_WXYZ,
        ),
        spec_fn=get_cryovial_rack_spec,
    )


def add_scene_contact_pairs(spec: mujoco.MjSpec) -> None:
    cryovial_body_geoms = tuple(
        geom.name
        for geom in spec.geoms
        if geom.name.startswith(f"cryovial/{CRYOVIAL_BODY_RACK_GEOM_PREFIX}_")
    )
    rack_geoms = tuple(
        geom.name
        for geom in spec.geoms
        if geom.name.startswith(f"cryovial_rack/{RACK_CRYOVIAL_BODY_GEOM_PREFIX}_")
    )

    for cryovial_body_geom in cryovial_body_geoms:
        for rack_geom in rack_geoms:
            spec.add_pair(
                geomname1=cryovial_body_geom,
                geomname2=rack_geom,
                friction=CRYOVIAL_RACK_PAIR_FRICTION,
                solref=CRYOVIAL_RACK_PAIR_SOLREF,
                solimp=CRYOVIAL_RACK_PAIR_SOLIMP,
            )


def check_success(
    env,
    pipette_tip_site_cfg: SceneEntityCfg,
    pipette_button_cfg: SceneEntityCfg,
    cryovial_cfg: SceneEntityCfg,
    max_tip_radial_dist: float,
    min_tip_axial_offset: float,
    max_tip_axial_offset: float,
    max_tip_axis_tilt: float,
    min_tip_inserted_steps: int,
    max_pressed_button_qpos: float,
) -> torch.Tensor:
    pipette = env.scene[pipette_tip_site_cfg.name]
    cryovial = env.scene[cryovial_cfg.name]
    pipette_tip_site_ids = pipette_tip_site_cfg.site_ids
    assert pipette_tip_site_ids is not None, "Please specify the pipette tip site."
    button_joint_ids = pipette_button_cfg.joint_ids
    assert button_joint_ids is not None and len(button_joint_ids) == 1, (
        "Please specify exactly one pipette button joint."
    )

    tip_pose = pipette.data.site_pose_w[:, pipette_tip_site_ids[0], :7]
    cryovial_pose = cryovial.data.root_link_pose_w[:, :7]
    tip_delta = tip_pose[:, :3] - cryovial_pose[:, :3]
    cryovial_axis = torch.zeros_like(tip_delta)
    cryovial_axis[:, 2] = 1.0
    cryovial_axis = quat_rotate_vec(cryovial_pose[:, 3:7], cryovial_axis)
    tip_axial_offset = torch.sum(tip_delta * cryovial_axis, dim=-1)
    tip_radial_delta = tip_delta - tip_axial_offset.unsqueeze(-1) * cryovial_axis
    tip_radial_dist = torch.linalg.norm(tip_radial_delta, dim=-1)
    tip_axis_tilt = axis_tilt_from_quats(
        tip_pose[:, 3:7],
        cryovial_pose[:, 3:7],
    )
    tip_radially_aligned = tip_radial_dist <= max_tip_radial_dist
    tip_depth_inserted = (
        (tip_axial_offset >= min_tip_axial_offset)
        & (tip_axial_offset <= max_tip_axial_offset)
    )
    tip_axis_aligned = tip_axis_tilt <= max_tip_axis_tilt
    tip_inserted_now = tip_radially_aligned & tip_depth_inserted & tip_axis_aligned
    tip_inserted_stable, tip_inserted_steps = stable_success(
        env,
        tip_inserted_now,
        state_attr="pipette_tip_inserted_stable_count",
        required_steps=min_tip_inserted_steps,
    )
    button_qpos = pipette.data.joint_pos[:, button_joint_ids[0]]
    button_pressed = button_qpos < max_pressed_button_qpos
    button_released = button_qpos >= 0.0
    pressed_while_inserted_now = tip_inserted_stable & button_pressed

    pressed_while_inserted_history = getattr(
        env,
        "pipette_button_pressed_while_inserted",
        torch.zeros((env.num_envs,), device=tip_pose.device, dtype=torch.bool),
    )

    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_mask = episode_length_buf <= 1
        pressed_while_inserted_history = torch.where(
            reset_mask,
            torch.zeros_like(pressed_while_inserted_history),
            pressed_while_inserted_history,
        )

    pressed_while_inserted_history = (
        pressed_while_inserted_history | pressed_while_inserted_now
    )
    success = (
        pressed_while_inserted_history & tip_inserted_stable & button_released
    )
    env.pipette_button_pressed_while_inserted = pressed_while_inserted_history

    record_success_diagnostics(
        env,
        object_complete=success,
        full_success=success,
        components={
            "tip_radially_aligned": tip_radially_aligned,
            "tip_depth_inserted": tip_depth_inserted,
            "tip_axis_aligned": tip_axis_aligned,
            "tip_inserted_now": tip_inserted_now,
            "tip_inserted_stable": tip_inserted_stable,
            "pipette_button_pressed_now": button_pressed,
            "pipette_button_pressed_while_inserted_now": pressed_while_inserted_now,
            "pipette_button_released_while_inserted_now": success,
            "pipette_button_pressed_while_inserted_ever": pressed_while_inserted_history,
        },
        metrics={
            "tip_cryovial_radial_dist_m": tip_radial_dist,
            "tip_cryovial_axial_offset_m": tip_axial_offset,
            "tip_cryovial_axis_tilt_rad": tip_axis_tilt,
            "tip_inserted_stable_steps": tip_inserted_steps,
            "pipette_button_qpos_m": button_qpos,
        },
        thresholds={
            "max_tip_cryovial_radial_dist_m": max_tip_radial_dist,
            "min_tip_cryovial_axial_offset_m": min_tip_axial_offset,
            "max_tip_cryovial_axial_offset_m": max_tip_axial_offset,
            "max_tip_cryovial_axis_tilt_rad": max_tip_axis_tilt,
            "min_tip_inserted_stable_steps": min_tip_inserted_steps,
            "max_pressed_button_qpos_m": max_pressed_button_qpos,
            "min_released_button_qpos_m": 0.0,
        },
    )
    return success


def make_shadow_pipette_manipulate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    del play

    robot_cfg = copy.deepcopy(get_ur5e_shadow_cfg())
    robot_cfg.spec_fn = get_shadow_pipette_manipulate_robot_spec
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

    gs_wrist_cam, gs_front_cam, gs_left_cam = make_gaussian_camera_triplet(
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention big table",
        scene_parent_body="table/vention big table",
        wrist_camera_name=None,
        wrist_parent_body="robot/hand:rh_palm",
        wrist_pos=(-0.14266717, 0.0168906141, -0.10597364),
        wrist_quat=(-0.0627922176, 0.688557002, 0.720658193, 0.050972163),
        wrist_fovy=58.0,
        wrist_width=320,
        wrist_height=240,
        front_pos=(-0.0578878102, -0.904093925, 1.03863107),
        front_quat=(0.719477506, 0.694515744, 8.8737741e-33, 2.66213223e-32),
        front_width=320,
        front_height=240,
        left_pos=(-0.59809019, 0.00183101078, 1.1704055),
        left_quat=(0.56270455, 0.381301052, -0.428639635, -0.595181619),
        left_width=320,
        left_height=240,
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
            "cryovial_rack": get_cryovial_rack_cfg(),
            "cryovial": get_cryovial_cfg(),
        },
        spec_fn=add_scene_contact_pairs,
        num_envs=1,
        env_spacing=3.0,
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
        "pipette_tip_site_pose_w": ObservationGroupCfg(
            {
                "pipette_tip_site_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "pipette",
                            site_names=["tip_site"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "cryovial_pose_w": ObservationGroupCfg(
            {
                "cryovial_pose_w": ObservationTermCfg(
                    func=root_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg("cryovial"),
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
        "gs_wrist_cam_rgb": ObservationGroupCfg(
            {
                "gs_wrist_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_wrist_cam_rgb"},
                ),
            },
            enable_corruption=False,
        ),
    }

    actions = {
        # "ur5e_joint_pos": JointPositionActionCfg(
        #     entity_name="robot",
        #     actuator_names=UR5E_SHADOW_ARM_JOINT_ORDER,
        #     use_default_offset=False,
        #     clip=None,
        # ),
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
        "reset_robot_slider_randomization": EventTermCfg(
            func=reset_joints_and_tendons_by_range,
            mode="reset",
            params={
                "joint_lows": PIPETTE_ARM_RESET_JOINT_LOWS
                + PIPETTE_HAND_RESET_JOINT_LOWS,
                "joint_highs": PIPETTE_ARM_RESET_JOINT_HIGHS
                + PIPETTE_HAND_RESET_JOINT_HIGHS,
                "joint_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_SHADOW_ARM_JOINT_ORDER
                    + UR5E_SHADOW_HAND_JOINT_ORDER,
                    preserve_order=True,
                ),
                "joint_action_terms": ("joint_pos",),
                "tendon_lows": PIPETTE_RESET_TENDON_LOWS,
                "tendon_highs": PIPETTE_RESET_TENDON_HIGHS,
                "tendon_asset_cfg": SceneEntityCfg(
                    "robot",
                    tendon_names=UR5E_SHADOW_HAND_TENDON,
                    preserve_order=True,
                ),
                "tendon_action_terms": ("tendon",),
            },
        ),
        "reset_cryovial_on_rack": EventTermCfg(
            func=reset_centrifuge_on_rack,
            mode="reset",
            params={
                "site_asset_cfg": SceneEntityCfg(
                    "cryovial_rack",
                    site_names=["centrifuge_plate-0"],
                ),
                "asset_cfg": SceneEntityCfg("cryovial"),
                "rows": CRYOVIAL_SLOT_COLS,
                "cols": CRYOVIAL_SLOT_ROWS,
                "row_gap": CRYOVIAL_SLOT_COL_GAP,
                "col_gap": CRYOVIAL_SLOT_ROW_GAP,
                "height": CRYOVIAL_SLOT_HEIGHT,
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
                "pipette_tip_site_cfg": SceneEntityCfg(
                    "pipette",
                    site_names=["tip_site"],
                ),
                "pipette_button_cfg": SceneEntityCfg(
                    "pipette",
                    joint_names=["pipette_button"],
                ),
                "cryovial_cfg": SceneEntityCfg("cryovial"),
                "max_tip_radial_dist": 0.003,
                "min_tip_axial_offset": 0.02,
                "max_tip_axial_offset": 0.05,
                "max_tip_axis_tilt": 0.261799,
                "min_tip_inserted_steps": 3,
                "max_pressed_button_qpos": -0.005,
            },
        ),
    }

    sim = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=SIM_TIMESTEP,
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
        decimation=SIM_DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
