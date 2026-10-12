import math
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.differential_ik import DifferentialIKActionCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul

from assets.mjcf.misc.simple_table import get_table_cfg
from assets.mjcf.object.centrifuge_10slot import (
    get_centrifuge_10slot_1_cfg,
    get_centrifuge_10slot_2_cfg,
    get_centrifuge_10slot_spec,
)
from assets.mjcf.object.centrifuge_50ml import get_centrifuge_50ml_cfg
from assets.mjcf.robot import (
    UR5E_JOINT_ORDER,
    XHAND_JOINT_ORDER,
    get_ur5e_xhand_cfg,
)
from src.tasks.config.gaussian_cameras import make_gaussian_camera
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.observations import (
    body_pose_w,
    camera_rgb,
    joint_pos_abs,
    site_pose_w,
)
from src.tasks.success_diagnostics import (
    record_success_diagnostics,
)
from src.tasks.task_goals import requested_task_goal_values
from src.utils.quat_util import quat_to_mat

_GAUSSIAN_SCENE = gaussian_scene_for_task("XHand-Tube-Insert")
_TABLE_GEOM_NAMES = tuple(
    f"table_geom_{idx}" for idx, _ in enumerate(get_table_cfg().spec_fn().geoms)
)
_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES = (
    "terrain",
    *(f"table/{name}" for name in _TABLE_GEOM_NAMES),
)
_XHAND_TUBE_INSERT_ENV_STEP_S = 1.0 / 30.0
# make_policy uses 40-step main trajectories; shorter alignment/hold phases
# bring the complete current stage sequence to 308 control steps.
_XHAND_TUBE_INSERT_POLICY_DONE_STEPS = 308
_XHAND_TUBE_INSERT_EPISODE_LENGTH_S = max(
    30.0,
    2.0 * _XHAND_TUBE_INSERT_POLICY_DONE_STEPS * _XHAND_TUBE_INSERT_ENV_STEP_S,
)
_SQRT_HALF = math.sqrt(0.5)
_INSERT_ROBOT_BASE_POS = (0.0, 0.0, 0.0)
_INSERT_ROBOT_BASE_ROT = (_SQRT_HALF, 0.0, 0.0, _SQRT_HALF)
_INSERT_ROBOT_ARM_JOINT_POS = {
    "shoulder_pan": -0.5 * math.pi,
    "shoulder_lift": -0.5 * math.pi,
    "elbow": -0.5 * math.pi,
    "wrist_1": math.pi,
    "wrist_2": -0.5 * math.pi,
    "wrist_3": -0.25 * math.pi,
}
_XHAND_OPEN_JOINT_POS = {joint_name: 0.0 for joint_name in XHAND_JOINT_ORDER}
# D405 eye-in-hand calibration from logs/extrinsics_wrist(1).yaml. The transform
# is relative to wrist_3_link; its OpenCV optical axes are converted to MuJoCo
# camera axes by right-multiplying the rotation by diag(1, -1, -1).
_XHAND_TUBE_INSERT_WRIST_CAM_POS = (
    0.0019726789193085913,
    0.14683233403995666,
    0.08800518082599727,
)
_XHAND_TUBE_INSERT_WRIST_CAM_QUAT = (
    0.4947696993930762,
    -0.0016321675140926822,
    -0.6360825620954048,
    0.5921142244447742,
)
_D405_FACTORY_WRIST_CAM_INTRINSICS = (
    393.6855773925781,
    393.1253967285156,
    324.22430419921875,
    244.115966796875,
)
_D405_FACTORY_WRIST_CAM_DISTORTION = (
    -0.05310627445578575,
    0.05576562136411667,
    -0.0005511727067641914,
    0.00048147179768420756,
    -0.018590731546282768,
)
_D405_FACTORY_WRIST_CAM_RESOLUTION = (640, 480)
_XHAND_TUBE_INSERT_WRIST_CAM_FOVY = 2.0 * math.degrees(
    math.atan(
        _D405_FACTORY_WRIST_CAM_RESOLUTION[1]
        / (2.0 * _D405_FACTORY_WRIST_CAM_INTRINSICS[1])
    )
)
# logs/video/intrinsics_d435_factory.yaml, calibrated at 640x480.
_D435_FACTORY_FRONT_CAM_INTRINSICS = (
    607.0804443359375,
    606.4393310546875,
    322.630126953125,
    242.50436401367188,
)
_D435_FACTORY_FRONT_CAM_FOVY = 2.0 * math.degrees(
    math.atan(480.0 / (2.0 * _D435_FACTORY_FRONT_CAM_INTRINSICS[1]))
)
_RACK_XY_OFFSET_RANGE_M = (0.01, 0.02)
_CAMERA_ROTATION_RANGE_DEG = (1.0, 5.0)
# Calibrated against /home/ubuntu/datasets_lerobot after the complete hybrid
# Gaussian/MuJoCo rendering pipeline, which noticeably brightens foregrounds.
_RACK_BLUE_RGB_RANGE = ((0.12, 0.155), (0.215, 0.265), (0.37, 0.45))
_TUBE_CAP_RED_RGB_RANGE = ((0.52, 0.62), (0.20, 0.28), (0.06, 0.12))
_TUBE_BODY_ALPHA_RANGE = (0.68, 0.86)


def sim_world_pos_to_real_base(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = pos
    return y - 0.30, -x, z - 0.824


def sim_world_quat_to_real_base(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, x, y, z = quat
    c = _SQRT_HALF
    return c * (w + z), c * (x + y), c * (y - x), c * (z - w)


def rotate_pos_z90(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = pos
    return -y, x, z


def rotate_quat_z90(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, x, y, z = quat
    c = _SQRT_HALF
    return c * (w - z), c * (x - y), c * (x + y), c * (w + z)

# FOV settings


def get_xhand_tube_insert_robot_spec():
    robot_cfg = get_ur5e_xhand_cfg()
    spec = robot_cfg.spec_fn()
    wrist_cam = spec.camera("wrist_cam")
    wrist_cam.frame.pos = (0.0, 0.0, 0.0)
    wrist_cam.frame.quat = (1.0, 0.0, 0.0, 0.0)
    wrist_cam.pos = _XHAND_TUBE_INSERT_WRIST_CAM_POS
    wrist_cam.quat = _XHAND_TUBE_INSERT_WRIST_CAM_QUAT
    wrist_cam.fovy = _XHAND_TUBE_INSERT_WRIST_CAM_FOVY
    wrist_cam.resolution = _D405_FACTORY_WRIST_CAM_RESOLUTION
    wrist_cam.sensor_size = _D405_FACTORY_WRIST_CAM_RESOLUTION
    wrist_cam.focal_pixel = _D405_FACTORY_WRIST_CAM_INTRINSICS[:2]
    wrist_cam.principal_pixel = _D405_FACTORY_WRIST_CAM_INTRINSICS[2:]
    return spec


def get_xhand_tube_insert_rack_spec():
    spec = get_centrifuge_10slot_spec()
    material = spec.material("centrifuge_10slot")
    material.specular = 0.6
    material.shininess = 0.55
    return spec


def sample_rack_tube_pose(
    site_pose: torch.Tensor,
    rows: int,
    cols: int,
    row_gap: float,
    col_gap: float,
    height: float,
    row_idx: int | None = None,
    col_idx: int | None = None,
    indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    site_pos = site_pose[:, :3]
    site_quat = site_pose[:, 3:]
    site_mat = quat_to_mat(site_quat)
    row_dir = site_mat[:, :, 1]
    col_dir = site_mat[:, :, 0]

    if indices is not None:
        indices = torch.as_tensor(indices, device=site_pose.device, dtype=torch.long)
        if tuple(indices.shape) != (site_pose.shape[0], 2):
            raise ValueError(
                f"Rack indices must have shape {(site_pose.shape[0], 2)}, "
                f"got {tuple(indices.shape)}."
            )
        row = indices[:, 0:1]
        col = indices[:, 1:2]
    elif row_idx is None:
        row = torch.randint(0, rows, (site_pose.shape[0], 1), device=site_pose.device)
    else:
        row = torch.full((site_pose.shape[0], 1), int(row_idx), device=site_pose.device)
    if indices is None:
        if col_idx is None:
            col = torch.randint(0, cols, (site_pose.shape[0], 1), device=site_pose.device)
        else:
            col = torch.full((site_pose.shape[0], 1), int(col_idx), device=site_pose.device)

    indices = torch.cat((row, col), dim=-1)
    if bool(((row < 0) | (row >= rows) | (col < 0) | (col >= cols)).any().item()):
        raise ValueError(f"Rack indices must satisfy row < {rows} and col < {cols}.")
    row_offset = row.to(dtype=site_pos.dtype)
    col_offset = col.to(dtype=site_pos.dtype)
    pos = site_pos + row_offset * row_gap * row_dir + col_offset * col_gap * col_dir
    pos[:, 2] += height

    # The workcell receives an additional +90-degree world-Z rotation, so the
    # upright tube returns to identity orientation after every reset.
    quat = torch.zeros((site_pose.shape[0], 4), device=site_pose.device, dtype=site_pos.dtype)
    quat[:, 0] = 1.0
    return torch.cat([pos, quat], dim=-1), indices


def reset_xhand_tube_insert_tube_and_target(
    env,
    env_ids: torch.Tensor | None,
    source_site_asset_cfg: SceneEntityCfg,
    target_site_asset_cfg: SceneEntityCfg,
    tube_cfg: SceneEntityCfg,
    rows: int = 5,
    cols: int = 2,
    row_gap: float = 0.036,
    col_gap: float = 0.036,
    height: float = 0.0,
    source_row_idx: int | None = None,
    source_col_idx: int | None = None,
    target_row_idx: int | None = None,
    target_col_idx: int | None = None,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    source_rack = env.scene[source_site_asset_cfg.name]
    target_rack = env.scene[target_site_asset_cfg.name]
    tube = env.scene[tube_cfg.name]

    env.sim.forward()
    source_site_ids = source_site_asset_cfg.site_ids
    target_site_ids = target_site_asset_cfg.site_ids
    assert source_site_ids is not None, "Please specify source site_names in SceneEntityCfg"
    assert target_site_ids is not None, "Please specify target site_names in SceneEntityCfg"

    source_site_pose = source_rack.data.site_pose_w[env_ids][:, source_site_ids[0]]
    target_site_pose = target_rack.data.site_pose_w[env_ids][:, target_site_ids[0]]
    requested_target = requested_task_goal_values(
        env,
        "XHand-Tube-Insert",
        env_ids,
    )
    source_tube_pose, _ = sample_rack_tube_pose(
        source_site_pose,
        rows,
        cols,
        row_gap,
        col_gap,
        height,
        source_row_idx,
        source_col_idx,
    )
    target_tube_pose, target_indices = sample_rack_tube_pose(
        target_site_pose,
        rows,
        cols,
        row_gap,
        col_gap,
        height,
        target_row_idx,
        target_col_idx,
        indices=requested_target,
    )

    zero_vel = torch.zeros_like(tube.data.root_link_vel_w[env_ids])
    tube.write_root_link_pose_to_sim(source_tube_pose, env_ids=env_ids)
    tube.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)

    env.xhand_tube_insert_target_tube_pose_w = getattr(
        env,
        "xhand_tube_insert_target_tube_pose_w",
        torch.zeros((env.num_envs, 7), device=env.device, dtype=target_tube_pose.dtype),
    )
    env.xhand_tube_insert_target_tube_pose_w[env_ids] = target_tube_pose
    env.xhand_tube_insert_target_indices = getattr(
        env,
        "xhand_tube_insert_target_indices",
        torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.long),
    )
    env.xhand_tube_insert_target_indices[env_ids] = target_indices
    env.sim.forward()


def randomize_rack_xy(
    env,
    env_ids: torch.Tensor | slice,
    rack_cfg: SceneEntityCfg,
    offset_range_m: tuple[float, float],
) -> None:
    rack = env.scene[rack_cfg.name]
    pose = rack.data.default_root_state[env_ids, :7].clone()
    count = pose.shape[0]
    direction = 2.0 * math.pi * torch.rand(count, device=env.device)
    distance = torch.empty(count, device=env.device).uniform_(*offset_range_m)
    pose[:, 0] += distance * torch.cos(direction)
    pose[:, 1] += distance * torch.sin(direction)
    pose[:, :3] += env.scene.env_origins[env_ids]
    rack.write_mocap_pose_to_sim(pose, env_ids=env_ids)


@requires_model_fields("mat_rgba")
def randomize_material_rgba(
    env,
    env_ids: torch.Tensor | slice,
    asset_cfg: SceneEntityCfg,
    rgb_range: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ],
    alpha_range: tuple[float, float] = (1.0, 1.0),
) -> None:
    material_ids = asset_cfg.material_ids
    asset = env.scene[asset_cfg.name]
    global_material_ids = asset.indexing.mat_ids[material_ids].long()
    if global_material_ids.numel() == 0:
        raise ValueError("asset_cfg.material_names must resolve to at least one material.")

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    indices = (env_ids[:, None], global_material_ids[None, :])
    rgba = env.sim.model.mat_rgba[indices].clone()
    for channel, channel_range in enumerate(rgb_range):
        rgba[..., channel].uniform_(*channel_range)
    rgba[..., 3].uniform_(*alpha_range)
    env.sim.model.mat_rgba[indices] = rgba


@requires_model_fields("cam_quat")
def randomize_camera_rotation(
    env,
    env_ids: torch.Tensor | slice,
    camera_name: str,
    rotation_range_deg: tuple[float, float],
) -> None:
    try:
        camera = env.scene[camera_name]
    except KeyError:
        return

    camera_id = camera.camera_idx
    camera_quat = env.sim.get_default_field("cam_quat")[camera_id].expand(
        env.sim.model.cam_quat[env_ids, camera_id].shape
    )
    axis = torch.randn((camera_quat.shape[0], 3), device=env.device)
    angle = torch.empty(camera_quat.shape[0], device=env.device).uniform_(
        math.radians(rotation_range_deg[0]),
        math.radians(rotation_range_deg[1]),
    )
    rotation = quat_from_angle_axis(angle, axis)
    env.sim.model.cam_quat[env_ids, camera_id] = quat_mul(camera_quat, rotation)


def target_tube_pose_w(env) -> torch.Tensor:
    target_pose = getattr(env, "xhand_tube_insert_target_tube_pose_w", None)
    if target_pose is not None:
        return target_pose
    tube = env.scene["centrifuge_50ml"]
    return tube.data.root_link_pose_w.clone()


def check_success(
    env,
    tube_cfg: SceneEntityCfg,
    max_tube_target_xy_dist: float,
    min_tube_z: float,
    max_tube_z: float,
) -> torch.Tensor:
    tube = env.scene[tube_cfg.name]

    target_pose = env.xhand_tube_insert_target_tube_pose_w
    tube_pos = tube.data.root_link_pose_w[:, :3]
    target_pos = target_pose[:, :3]
    tube_xy_dist = torch.linalg.norm(tube_pos[:, :2] - target_pos[:, :2], dim=-1)
    tube_z = tube_pos[:, 2]
    tube_inserted = (
        (tube_xy_dist < max_tube_target_xy_dist)
        & (tube_z > min_tube_z)
        & (tube_z < max_tube_z)
    )
    record_success_diagnostics(
        env,
        object_complete=tube_inserted,
        full_success=tube_inserted,
        components={
            "tube_inserted_now": tube_inserted,
        },
        metrics={
            "tube_target_xy_dist_m": tube_xy_dist,
            "tube_z_m": tube_z,
        },
        thresholds={
            "max_tube_target_xy_dist_m": max_tube_target_xy_dist,
            "min_tube_z_m": min_tube_z,
            "max_tube_z_m": max_tube_z,
        },
    )
    return tube_inserted


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


def make_xhand_tube_insert_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = get_ur5e_xhand_cfg()
    robot_cfg.spec_fn = get_xhand_tube_insert_robot_spec
    robot_cfg.init_state.pos = _INSERT_ROBOT_BASE_POS
    robot_cfg.init_state.rot = _INSERT_ROBOT_BASE_ROT
    robot_cfg.init_state.joint_pos.update(_INSERT_ROBOT_ARM_JOINT_POS)
    robot_cfg.init_state.joint_pos.update(_XHAND_OPEN_JOINT_POS)
    table_cfg = get_gaussian_hidden_table_cfg()
    table_cfg.init_state.pos = rotate_pos_z90(
        sim_world_pos_to_real_base(table_cfg.init_state.pos)
    )
    table_cfg.init_state.rot = rotate_quat_z90(
        sim_world_quat_to_real_base(table_cfg.init_state.rot)
    )
    rack_1_cfg = get_centrifuge_10slot_1_cfg()
    rack_1_pos = rotate_pos_z90(
        sim_world_pos_to_real_base(
            (0.025000005960464478, -0.3799999475479126, 0.8539999993294477)
        )
    )
    rack_1_cfg.init_state.pos = (
        rack_1_pos[0] + 0.01998,
        rack_1_pos[1] - 0.01494,
        rack_1_pos[2],
    )
    rack_1_cfg.init_state.rot = rotate_quat_z90(
        sim_world_quat_to_real_base(rack_1_cfg.init_state.rot)
    )
    rack_1_cfg.spec_fn = get_xhand_tube_insert_rack_spec
    rack_2_cfg = get_centrifuge_10slot_2_cfg()
    rack_2_pos = rotate_pos_z90(
        sim_world_pos_to_real_base(
            (-0.24099993705749512, -0.3935001611709595, 0.8539999993294477)
        )
    )
    rack_2_cfg.init_state.pos = (
        rack_2_pos[0] + 0.050,
        rack_2_pos[1] - 0.014,
        rack_2_pos[2],
    )
    rack_2_cfg.init_state.rot = rotate_quat_z90(
        sim_world_quat_to_real_base(rack_2_cfg.init_state.rot)
    )
    rack_2_cfg.spec_fn = get_xhand_tube_insert_rack_spec
    tube_cfg = get_centrifuge_50ml_cfg()
    tube_cfg.init_state.pos = rotate_pos_z90(
        sim_world_pos_to_real_base(tube_cfg.init_state.pos)
    )
    tube_cfg.init_state.rot = rotate_quat_z90(
        sim_world_quat_to_real_base(tube_cfg.init_state.rot)
    )
    terrain_cfg = TerrainEntityCfg(terrain_type="plane")
    terrain_cfg.init_state.pos = sim_world_pos_to_real_base(terrain_cfg.init_state.pos)
    terrain_cfg.init_state.rot = sim_world_quat_to_real_base(terrain_cfg.init_state.rot)

    gs_wrist_cam = make_gaussian_camera(
        name="gs_wrist_cam_rgb",
        camera_name="robot/wrist_cam",
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        fovy=_XHAND_TUBE_INSERT_WRIST_CAM_FOVY,
        intrinsics=_D405_FACTORY_WRIST_CAM_INTRINSICS,
        distortion_model="inverse_brown_conrady",
        distortion=_D405_FACTORY_WRIST_CAM_DISTORTION,
        width=_D405_FACTORY_WRIST_CAM_RESOLUTION[0],
        height=_D405_FACTORY_WRIST_CAM_RESOLUTION[1],
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )
    gs_front_cam = make_gaussian_camera(
        name="gs_front_cam_rgb",
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        parent_body="table/vention table",
        # D435 eye-to-hand calibration from
        # logs/extrinsics_d435_eye_to_hand.yaml. X_CammountCam is the OpenCV
        # optical-frame pose in the robot's authored base body. Convert its
        # optical axes to MuJoCo camera axes, compose both the entity root and
        # base-body rotations, then express it in this table-local frame.
        pos=(
            0.1630905837682807,
            -1.1955311720394552,
            1.5699469785672653,
        ),
        quat=(
            0.8639680470337237,
            0.5034648926147376,
            0.0055491991066283185,
            -0.0071778825874318055,
        ),
        fovy=_D435_FACTORY_FRONT_CAM_FOVY,
        intrinsics=_D435_FACTORY_FRONT_CAM_INTRINSICS,
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )
    # --------------------------
    # Scene
    # --------------------------
    scene = SceneCfg(
        terrain=terrain_cfg,
        entities={
            "robot": robot_cfg,
            "table": table_cfg,
            "centrifuge_10slot_1": rack_1_cfg,
            "centrifuge_10slot_2": rack_2_cfg,
            "centrifuge_50ml": tube_cfg,
        },
        num_envs=1,
        env_spacing=3.0,
        sensors=(gs_wrist_cam, gs_front_cam),
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
                            joint_names=UR5E_JOINT_ORDER + XHAND_JOINT_ORDER,
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        
        "pinch_site_pose_w": ObservationGroupCfg(
            {
                "pinch_site_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            site_names=[r".*xhand:right_palm"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),

        "slot1_site_pose_w": ObservationGroupCfg(
            {
                "slot1_site_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "centrifuge_10slot_1",
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
                            "centrifuge_50ml",
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
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "centrifuge_50ml",
                            body_names=["centrifuge_50ml_screw_cap"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),

        "target_tube_pose_w": ObservationGroupCfg(
            {
                "target_tube_pose_w": ObservationTermCfg(
                    func=target_tube_pose_w,
                ),
            },
            enable_corruption=False,
            history_length=1,
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
    }


    # --------------------------
    # ✅ Absolute ik control
    # --------------------------
    actions = {
    # "joint_pos": JointPositionActionCfg(
    #     entity_name="robot",
    #     actuator_names=UR5E_JOINT_ORDER,
    #     use_default_offset=False,
    #     clip=None,
    # ),
    "site_pose": DifferentialIKActionCfg(
        entity_name="robot",
        actuator_names=UR5E_JOINT_ORDER,
        frame_type="site",
        frame_name=r".*xhand:right_palm",
        use_relative_mode=False,
        position_weight=1,
        orientation_weight=0.5,
    ),
    "xhand": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=XHAND_JOINT_ORDER,
        use_default_offset=False,
        clip=None,
    ),
    }

    # --------------------------
    # events
    # --------------------------
    events = {
    # Required for fixed-base mocap robots in vectorized envs:
    # place each hand at its own env origin on every reset.
    "reset_scene": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
    ),
    "randomize_source_rack_xy": EventTermCfg(
      func=randomize_rack_xy,
      mode="reset",
      params={
        "rack_cfg": SceneEntityCfg("centrifuge_10slot_1"),
        "offset_range_m": _RACK_XY_OFFSET_RANGE_M,
      },
    ),
    "randomize_target_rack_xy": EventTermCfg(
      func=randomize_rack_xy,
      mode="reset",
      params={
        "rack_cfg": SceneEntityCfg("centrifuge_10slot_2"),
        "offset_range_m": _RACK_XY_OFFSET_RANGE_M,
      },
    ),
    "randomize_source_rack_color": EventTermCfg(
      func=randomize_material_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
            "centrifuge_10slot_1",
            material_names=("centrifuge_10slot",),
        ),
        "rgb_range": _RACK_BLUE_RGB_RANGE,
      },
    ),
    "randomize_target_rack_color": EventTermCfg(
      func=randomize_material_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
            "centrifuge_10slot_2",
            material_names=("centrifuge_10slot",),
        ),
        "rgb_range": _RACK_BLUE_RGB_RANGE,
      },
    ),
    "randomize_tube_cap_color": EventTermCfg(
      func=randomize_material_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
            "centrifuge_50ml",
            material_names=("centrifuge_50ml_screw_cap",),
        ),
        "rgb_range": _TUBE_CAP_RED_RGB_RANGE,
      },
    ),
    "randomize_tube_body_alpha": EventTermCfg(
      func=randomize_material_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
            "centrifuge_50ml",
            material_names=("centrifuge_50ml_screw_body",),
        ),
        "rgb_range": ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        "alpha_range": _TUBE_BODY_ALPHA_RANGE,
      },
    ),
    "reset_xhand_tube_insert_tube_and_target": EventTermCfg(
      func=reset_xhand_tube_insert_tube_and_target,
      mode="reset",
      params={
        # Keep the demonstrated direction in the fixed front-camera view:
        # rack 1 is image-right (source), rack 2 is image-left (target).
        "source_site_asset_cfg": SceneEntityCfg(
                    "centrifuge_10slot_1",
                     site_names=["centrifuge_10slot-50ml"],
                    ),
        "target_site_asset_cfg": SceneEntityCfg(
                    "centrifuge_10slot_2",
                     site_names=["centrifuge_10slot-50ml"],
                    ),
        "tube_cfg": SceneEntityCfg("centrifuge_50ml"),
        },
    ),
    "randomize_head_camera_rotation": EventTermCfg(
      func=randomize_camera_rotation,
      mode="reset",
      params={
        "camera_name": "gs_front_cam_rgb",
        "rotation_range_deg": _CAMERA_ROTATION_RANGE_DEG,
      },
    ),
    "randomize_wrist_camera_rotation": EventTermCfg(
      func=randomize_camera_rotation,
      mode="reset",
      params={
        "camera_name": "gs_wrist_cam_rgb",
        "rotation_range_deg": _CAMERA_ROTATION_RANGE_DEG,
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
                "tube_cfg": SceneEntityCfg("centrifuge_50ml"),
                "max_tube_target_xy_dist": 0.015,
                "min_tube_z": 0.005,
                "max_tube_z": 0.007,
            },
        ),
    }

    # --------------------------
    # Simulation
    # --------------------------
    sim = SimulationCfg(
       mujoco=MujocoCfg(
        timestep=1.0 / 750.0,
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
        episode_length_s=_XHAND_TUBE_INSERT_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
