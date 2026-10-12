import copy
from pathlib import Path

import mujoco
import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import TendonLengthActionCfg
from mjlab.envs.mdp.actions.differential_ik import DifferentialIKActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from assets.mjcf.misc.big_table import get_table_spec as get_big_table_spec
from assets.mjcf.robot.ur5e_shadow import (
    UR5E_SHADOW_ARM_JOINT_ORDER,
    UR5E_SHADOW_HAND_JOINT_ORDER,
    UR5E_SHADOW_HAND_TENDON,
    get_ur5e_shadow_cfg,
)
from src import MJCF_ROOT
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.actions import OrderedJointPositionActionCfg
from src.tasks.mdp.events import (
    reset_centrifuge_5430_tubes,
    reset_joints_and_tendons_by_range,
)
from src.tasks.mdp.observations import (
    body_pose_w,
    camera_rgb,
    joint_pos_abs,
    site_pose_w,
)
from src.tasks.success_diagnostics import (
    record_success_diagnostics,
)
from src.utils.quat_util import quat_rotate_vec

CENTRIFUGE_5910_RI_XML: Path = (
    MJCF_ROOT / "instrument" / "centrifuge_eppendorf_5910_ri.xml"
)
_GAUSSIAN_SCENE = gaussian_scene_for_task("Shadow-Centrifuge-Insert")
RACK_XML: Path = MJCF_ROOT / "object" / "centrifuge_10slot.gen.xml"
TUBE_XML: Path = MJCF_ROOT / "object" / "centrifuge_50ml.xml"
TUBE_ADAPTER_HARD_CONTACT_SOLREF = (0.0005, 1.0)
TUBE_ADAPTER_HARD_CONTACT_SOLIMP = (0.95, 0.995, 0.0001, 0.5, 2.0)
TUBE_ADAPTER_HARD_CONTACT_FRICTION = (1.0, 1.0, 0.005, 0.0001, 0.0001)
SHADOW_CENTRIFUGE_INSERT_ENV_STEP_S = 0.05
# Policy interpolation advances once per environment step, independent of decimation.
SHADOW_CENTRIFUGE_INSERT_POLICY_DONE_STEPS = 1000
SHADOW_CENTRIFUGE_INSERT_EPISODE_LENGTH_S = (
    2.0
    * SHADOW_CENTRIFUGE_INSERT_POLICY_DONE_STEPS
    * SHADOW_CENTRIFUGE_INSERT_ENV_STEP_S
)

CENTRIFUGE_50ML_SLOT_SITE_NAMES = [
    f"adapter-7x50-{adapter_id}-slot-{slot_row}{slot_id}"
    for adapter_id in range(2)
    for slot_row, count in (("bottom", 2), ("middle", 3), ("top", 2))
    for slot_id in range(count)
]
CENTRIFUGE_50ML_SAFE_SLOT_IDS = tuple(
    index for index, name in enumerate(CENTRIFUGE_50ML_SLOT_SITE_NAMES) if "-top" not in name
)
SLIDER_RESET_JOINT_LOWS = (
    -0.5026552,
    -1.1938061,
    0.69113,
    0.5026552,
    0.5026552,
    -1.1938061,
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
SLIDER_RESET_JOINT_HIGHS = (
    0.5026552,
    -1.0681423,
    1.06811,
    0.6911509,
    0.6911509,
    0.0,
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
SLIDER_RESET_TENDON_LOWS = (
    0.0,
    0.0,
    0.0,
    0.0,
)
SLIDER_RESET_TENDON_HIGHS = (
    0.4398226,
    0.3612828,
    0.5497782,
    0.8325213,
)


def load_spec_with_assets(xml_path: Path, prefix: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    asset_root = (xml_path.parent / spec.meshdir).resolve() if spec.meshdir else xml_path.parent
    assets = {}

    def resolve_asset_path(asset_file: str) -> Path:
        full_path = (asset_root / asset_file).resolve()
        if full_path.exists():
            return full_path

        raise FileNotFoundError(
            f"Missing asset referenced by {xml_path}: {asset_file} "
            f"(resolved under {asset_root})"
        )

    for mesh in spec.meshes:
        if not mesh.file:
            continue
        full_path = resolve_asset_path(mesh.file)
        new_key = f"{prefix}/{mesh.file}"
        with open(full_path, "rb") as f:
            assets[new_key] = f.read()
        mesh.file = new_key

    for texture in spec.textures:
        if not texture.file:
            continue
        full_path = resolve_asset_path(texture.file)
        new_key = f"{prefix}/{texture.file}"
        with open(full_path, "rb") as f:
            assets[new_key] = f.read()
        texture.file = new_key

    spec.assets = assets
    return spec


def get_centrifuge_5910_ri_spec() -> mujoco.MjSpec:
    return load_spec_with_assets(
        CENTRIFUGE_5910_RI_XML,
        "centrifuge_eppendorf_5910_ri",
    )


def get_centrifuge_5910_ri_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(-0.4, 0.1, 0.824),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spec_fn=get_centrifuge_5910_ri_spec,
    )


def get_centrifuge_table_spec() -> mujoco.MjSpec:
    spec = get_big_table_spec()
    table_geoms = list(spec.geoms)

    for geom_id in (0, 2, 4, 6, 8, 10):
        table_geoms[geom_id].pos[0] = -0.82
    for geom_id in (1, 3, 5, 7, 9, 11):
        table_geoms[geom_id].pos[0] = 0.82
    for geom_id in (0, 1, 4, 5, 8, 9, 13):
        table_geoms[geom_id].pos[1] = -1.02
    for geom_id in (2, 3, 6, 7, 10, 11, 12):
        table_geoms[geom_id].pos[1] = 1.02

    table_geoms[14].size[:2] = (0.90, 1.05)
    table_geoms[15].size[:2] = (0.902, 1.052)
    table_geoms[16].size[:2] = (0.85, 1.00)
    table_geoms[17].size[:2] = (0.852, 1.002)
    table_geoms[18].size[:2] = (0.852, 1.002)
    return spec


def get_centrifuge_table_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spec_fn=get_centrifuge_table_spec,
    )


def get_rack_spec() -> mujoco.MjSpec:
    return load_spec_with_assets(RACK_XML, "centrifuge_10slot")


def get_rack_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.4, -0.45, 0.854),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spec_fn=get_rack_spec,
    )


def get_tube_spec() -> mujoco.MjSpec:
    tube_spec = load_spec_with_assets(TUBE_XML, "centrifuge_50ml_screw")
    for joint in list(tube_spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            tube_spec.delete(joint)

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="centrifuge_50ml_screw")
    body.add_joint(
        name="centrifuge_50ml_screw_joint",
        type=mujoco.mjtJoint.mjJNT_FREE,
    )
    frame = body.add_frame()
    spec.attach(child=tube_spec, prefix="", frame=frame)
    return spec


def get_tube_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.272, 0.406, 0.84),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spec_fn=get_tube_spec,
    )


def geom_parent_name(geom) -> str:
    parent = getattr(geom, "parent", None)
    return "" if parent is None else str(getattr(parent, "name", ""))


def is_tube_body_contact_geom(geom) -> bool:
    parent_name = geom_parent_name(geom)
    if not parent_name.endswith("centrifuge_50ml_screw_body"):
        return False
    return int(geom.group) == 3 and geom.type in (
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    )


def is_adapter_7x50_slot_fence_geom(geom) -> bool:
    parent_name = geom_parent_name(geom)
    if "adapter-7x50" not in parent_name:
        return False
    if geom.type != mujoco.mjtGeom.mjGEOM_BOX or int(geom.group) != 3:
        return False
    size = geom.size
    return float(size[2]) > 0.02 and min(float(size[0]), float(size[1])) <= 0.001


def ensure_geom_name(geom, prefix: str, index: int) -> str:
    if geom.name:
        return geom.name
    geom.name = f"{prefix}_{index:03d}"
    return geom.name


def harden_tube_adapter_contacts(spec: mujoco.MjSpec) -> None:
    tube_body_geoms = [
        ensure_geom_name(geom, "tube_body_collision", geom_id)
        for geom_id, geom in enumerate(spec.geoms)
        if is_tube_body_contact_geom(geom)
    ]
    adapter_fence_geoms = [
        ensure_geom_name(geom, "centrifuge_adapter_7x50_fence_collision", geom_id)
        for geom_id, geom in enumerate(spec.geoms)
        if is_adapter_7x50_slot_fence_geom(geom)
    ]

    for tube_body_geom in tube_body_geoms:
        for adapter_fence_geom in adapter_fence_geoms:
            spec.add_pair(
                geomname1=tube_body_geom,
                geomname2=adapter_fence_geom,
                friction=TUBE_ADAPTER_HARD_CONTACT_FRICTION,
                solref=TUBE_ADAPTER_HARD_CONTACT_SOLREF,
                solimp=TUBE_ADAPTER_HARD_CONTACT_SOLIMP,
            )


def configure_centrifuge_scene_spec(spec: mujoco.MjSpec) -> None:
    spec.nconmax = 2048
    spec.njmax = 2048
    harden_tube_adapter_contacts(spec)


def check_success(
    env,
    tube_cfg: SceneEntityCfg,
    max_tube_target_radial_dist: float,
    min_tube_insertion_depth: float,
    max_tube_insertion_depth: float,
) -> torch.Tensor:
    tube = env.scene[tube_cfg.name]

    target_tube_pose_w = env.centrifuge_final_tube_pose_w
    tube_pose_w = tube.data.root_link_pose_w
    tube_pos = tube_pose_w[:, :3]
    target_tube_pos = target_tube_pose_w[:, :3]
    target_tube_quat = target_tube_pose_w[:, 3:7]

    # The target is a slot-relative planning reference. After release, the tube
    # physically settles deeper than that pose, so measure depth along the
    # randomized slot axis instead of using a fixed world-Z window.
    target_axis = torch.zeros_like(tube_pos)
    target_axis[:, 2] = 1.0
    target_axis = quat_rotate_vec(target_tube_quat, target_axis)
    target_delta = tube_pos - target_tube_pos
    target_axis_offset = torch.sum(target_delta * target_axis, dim=-1)
    tube_insertion_depth = -target_axis_offset
    target_radial_delta = (
        target_delta - target_axis_offset.unsqueeze(-1) * target_axis
    )
    tube_target_radial_dist = torch.linalg.norm(target_radial_delta, dim=-1)

    tube_radially_aligned = (
        tube_target_radial_dist <= max_tube_target_radial_dist
    )
    tube_depth_inserted = (
        (tube_insertion_depth >= min_tube_insertion_depth)
        & (tube_insertion_depth <= max_tube_insertion_depth)
    )
    tube_inserted = tube_radially_aligned & tube_depth_inserted

    record_success_diagnostics(
        env,
        object_complete=tube_inserted,
        full_success=tube_inserted,
        components={
            "tube_inserted_now": tube_inserted,
            "tube_radially_aligned": tube_radially_aligned,
            "tube_depth_inserted": tube_depth_inserted,
        },
        metrics={
            "tube_target_radial_dist_m": tube_target_radial_dist,
            "tube_insertion_depth_m": tube_insertion_depth,
        },
        thresholds={
            "max_tube_target_radial_dist_m": max_tube_target_radial_dist,
            "min_tube_insertion_depth_m": min_tube_insertion_depth,
            "max_tube_insertion_depth_m": max_tube_insertion_depth,
        },
    )
    return tube_inserted


def make_shadow_centrifuge_insert_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = copy.deepcopy(get_ur5e_shadow_cfg())
    # Match dexbiolab insert_centrifuge_5430 UR5e base pose and joint home.
    robot_cfg.init_state.pos = (0.5, 0.3, 0.824)
    robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
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
        background_frame_body_name="table/vention big table",
        scene_parent_body="table/vention big table",
        wrist_camera_name="robot/wrist_cam",
        wrist_parent_body="robot/wrist_3_link",
        wrist_width=320,
        wrist_height=240,
        front_pos=(-0.0506275237, -1.7329027, 1.97410775),
        front_quat=(0.848035758, 0.529801155, -0.00154353368, -0.0119877342),
        front_width=320,
        front_height=240,
        left_pos=(-1.52111104, 0.256739549, 1.9016409),
        left_quat=(0.601758845, 0.293769001, -0.366938899, -0.645710393),
        left_width=320,
        left_height=240,
    )
    # --------------------------
    # Scene
    # --------------------------
    scene = SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="plane",
        ),
        entities={
            "robot": robot_cfg,
            "table": get_centrifuge_table_cfg(),
            "centrifuge": get_centrifuge_5910_ri_cfg(),
            "rack": get_rack_cfg(),
            "tube": get_tube_cfg(),
            "tube_opposite": get_tube_cfg(),
        },
        num_envs=1,
        env_spacing=3.0,
        sensors=(gs_wrist_cam, gs_front_cam, gs_left_cam),
        spec_fn=configure_centrifuge_scene_spec,
    )

    # --------------------------
    # Observations
    # --------------------------
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
        "centrifuge_qpos": ObservationGroupCfg(
            {
                "centrifuge_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "centrifuge",
                            joint_names=["lid", "rotor-body"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "centrifuge_lid_pose_w": ObservationGroupCfg(
            {
                "centrifuge_lid_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "centrifuge",
                            site_names=["lid"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "centrifuge_slot_poses_w": ObservationGroupCfg(
            {
                "centrifuge_slot_poses_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "centrifuge",
                            site_names=CENTRIFUGE_50ML_SLOT_SITE_NAMES,
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "rack_origin_pose_w": ObservationGroupCfg(
            {
                "rack_origin_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "rack",
                            site_names=["centrifuge_10slot-50ml"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tube_body_pose_w": ObservationGroupCfg(
            {
                "tube_body_pose_w": ObservationTermCfg(
                    func=body_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "tube",
                            body_names=["centrifuge_50ml_screw_body"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "tube_opposite_body_pose_w": ObservationGroupCfg(
            {
                "tube_opposite_body_pose_w": ObservationTermCfg(
                    func=body_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "tube_opposite",
                            body_names=["centrifuge_50ml_screw_body"],
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
                    params={
                        "sensor_name": "gs_wrist_cam_rgb",
                    },
                ),
            },
            enable_corruption=False,
        ),
        "gs_front_cam_rgb": ObservationGroupCfg(
            {
                "gs_front_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={
                        "sensor_name": "gs_front_cam_rgb",
                    },
                ),
            },
            enable_corruption=False,
        ),
        "gs_left_cam_rgb": ObservationGroupCfg(
            {
                "gs_left_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={
                        "sensor_name": "gs_left_cam_rgb",
                    },
                ),
            },
            enable_corruption=False,
        ),
    }


    # --------------------------
    # Absolute IK control
    # --------------------------
    actions = {
        # "ur5e_joint_pos": OrderedJointPositionActionCfg(
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
            orientation_weight=0.3,
        ),
        "shadow_joint_pos": OrderedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=UR5E_SHADOW_HAND_JOINT_ORDER,
            use_default_offset=False,
            preserve_order=True,
            clip=None,
        ),
        "shadow_tendon": TendonLengthActionCfg(
            entity_name="robot",
            actuator_names=UR5E_SHADOW_HAND_TENDON,
            preserve_order=True,
            clip=None,
            transmission_type=TransmissionType.TENDON,
        ),
    }

    # --------------------------
    # events
    # --------------------------
    events = {
        "reset_scene": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "reset_robot_slider_randomization": EventTermCfg(
            func=reset_joints_and_tendons_by_range,
            mode="reset",
            params={
                "joint_lows": SLIDER_RESET_JOINT_LOWS,
                "joint_highs": SLIDER_RESET_JOINT_HIGHS,
                "joint_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_SHADOW_ARM_JOINT_ORDER
                    + UR5E_SHADOW_HAND_JOINT_ORDER,
                ),
                "joint_action_terms": ("shadow_joint_pos",),
                "tendon_lows": SLIDER_RESET_TENDON_LOWS,
                "tendon_highs": SLIDER_RESET_TENDON_HIGHS,
                "tendon_asset_cfg": SceneEntityCfg(
                    "robot",
                    tendon_names=UR5E_SHADOW_HAND_TENDON,
                ),
                "tendon_action_terms": ("shadow_tendon",),
            },
        ),
        "reset_centrifuge_5430_tubes": EventTermCfg(
            func=reset_centrifuge_5430_tubes,
            mode="reset",
            params={
                "centrifuge_cfg": SceneEntityCfg(
                    "centrifuge",
                    site_names=CENTRIFUGE_50ML_SLOT_SITE_NAMES,
                ),
                "rack_cfg": SceneEntityCfg(
                    "rack",
                    site_names=["centrifuge_10slot-50ml"],
                ),
                "tube_cfg": SceneEntityCfg("tube"),
                "tube_opposite_cfg": SceneEntityCfg("tube_opposite"),
                "rotor_joint_cfg": SceneEntityCfg(
                    "centrifuge",
                    joint_names=["rotor-body"],
                ),
                "rack_row_gap": 0.036,
                "rack_col_gap": 0.036,
                "rack_height": 0.06,
                "randomize_rack_position": True,
                "rack_row_range": (0, 4),
                "rack_col_range": (0, 1),
                "target_slot_ids": CENTRIFUGE_50ML_SAFE_SLOT_IDS,
            },
        ),
    }
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
                "max_tube_target_radial_dist": 0.015,
                "min_tube_insertion_depth": 0.05,
                "max_tube_insertion_depth": 0.10,
            },
        ),
    }

    # --------------------------
    # Simulation
    # --------------------------
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
        episode_length_s=SHADOW_CENTRIFUGE_INSERT_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
