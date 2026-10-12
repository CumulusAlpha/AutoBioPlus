import copy
from pathlib import Path

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import (
    DifferentialIKActionCfg,
    JointPositionActionCfg,
    TendonLengthActionCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from assets.mjcf.misc.simple_table import get_table_cfg
from assets.mjcf.object.centrifuge_10slot import (
    get_centrifuge_10slot_1_cfg,
    get_centrifuge_10slot_2_cfg,
)
from assets.mjcf.object.centrifuge_50ml import get_centrifuge_50ml_cfg
from assets.mjcf.robot.ur5e_shadow import (
    UR5E_SHADOW_ARM_JOINT_ORDER,
    UR5E_SHADOW_HAND_JOINT_ORDER,
    UR5E_SHADOW_HAND_TENDON,
    get_ur5e_shadow_cfg,
)
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.actions import OrderedJointPositionActionCfg
from src.tasks.mdp.events import (
    reset_centrifuge_on_rack,
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
from src.tasks.task_goals import requested_task_goal_values
from src.utils.quat_util import quat_to_mat

_GAUSSIAN_SCENE = gaussian_scene_for_task("Shadow-Tube-Insert")
SHADOW_TUBE_INSERT_ENV_STEP_S = 0.05
# Policy interpolation advances once per environment step, independent of decimation.
SHADOW_TUBE_INSERT_POLICY_DONE_STEPS = 1000
SHADOW_TUBE_INSERT_EPISODE_LENGTH_S = (
    2.0 * SHADOW_TUBE_INSERT_POLICY_DONE_STEPS * SHADOW_TUBE_INSERT_ENV_STEP_S
)


UR_SHADOW_ARM_RESET_JOINT_LOWS = (
    -0.188495700,
    -1.759293200,
    1.256600000,
    -1.884957000,
    -0.753982800,
    0.628319000,
)
UR_SHADOW_ARM_RESET_JOINT_HIGHS = (
    0.251327600,
    -1.193806100,
    1.602165000,
    -1.068142300,
    0.376991400,
    2.073452700,
)
UR_SHADOW_HAND_RESET_JOINT_LOWS = (
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
UR_SHADOW_HAND_RESET_JOINT_HIGHS = (
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
UR_SHADOW_RESET_TENDON_LOWS = (
    0.0,
    0.0,
    0.0,
    0.0,
)
UR_SHADOW_RESET_TENDON_HIGHS = (
    0.4398226,
    0.3612828,
    0.5497782,
    0.8325213,
)


def rack_tube_positions_w(
    site_pose_w: torch.Tensor,
    rows: int,
    cols: int,
    row_gap: float,
    col_gap: float,
    height: float,
) -> torch.Tensor:
    site_pos = site_pose_w[:, :3]
    site_quat = site_pose_w[:, 3:]
    site_mat = quat_to_mat(site_quat)
    row_dir = site_mat[:, :, 1]
    col_dir = site_mat[:, :, 0]

    row_ids = torch.arange(rows, device=site_pose_w.device, dtype=site_pos.dtype)
    col_ids = torch.arange(cols, device=site_pose_w.device, dtype=site_pos.dtype)
    row_offset = row_ids.view(1, rows, 1, 1) * row_gap * row_dir[:, None, None, :]
    col_offset = col_ids.view(1, 1, cols, 1) * col_gap * col_dir[:, None, None, :]
    tube_pos = site_pos[:, None, None, :] + row_offset + col_offset
    tube_pos[..., 2] += height
    return tube_pos.reshape(site_pose_w.shape[0], rows * cols, 3)


def reset_shadow_tube_insert_target(
    env,
    env_ids: torch.Tensor | None,
    rows: int = 5,
    cols: int = 2,
) -> None:
    """Sample the target rack slot as episode state owned by the environment."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    target_indices = getattr(env, "shadow_tube_insert_target_indices", None)
    if target_indices is None or target_indices.shape != (env.num_envs, 2):
        target_indices = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.long)
    requested_target = requested_task_goal_values(
        env,
        "Shadow-Tube-Insert",
        env_ids,
    )
    if requested_target is None:
        target_indices[env_ids, 0] = torch.randint(
            0,
            rows,
            (env_ids.numel(),),
            device=env.device,
        )
        target_indices[env_ids, 1] = torch.randint(
            0,
            cols,
            (env_ids.numel(),),
            device=env.device,
        )
    else:
        requested_target = requested_target.to(dtype=torch.long)
        if bool(
            (
                (requested_target[:, 0] < 0)
                | (requested_target[:, 0] >= rows)
                | (requested_target[:, 1] < 0)
                | (requested_target[:, 1] >= cols)
            ).any().item()
        ):
            raise ValueError(f"Rack indices must satisfy row < {rows} and col < {cols}.")
        target_indices[env_ids] = requested_target
    env.shadow_tube_insert_target_indices = target_indices

def check_success(
    env,
    tube_cfg: SceneEntityCfg,
    target_site_cfg: SceneEntityCfg,
    rows: int,
    cols: int,
    row_gap: float,
    col_gap: float,
    height: float,
    max_tube_target_xy_dist: float,
    min_tube_z: float,
    max_tube_z: float,
) -> torch.Tensor:
    tube = env.scene[tube_cfg.name]
    target_rack = env.scene[target_site_cfg.name]

    target_site_ids = target_site_cfg.site_ids
    assert target_site_ids is not None, "Please specify target site_names in SceneEntityCfg"

    target_site_pose_w = target_rack.data.site_pose_w[:, target_site_ids[0]]
    target_tube_positions_w = rack_tube_positions_w(
        target_site_pose_w,
        rows,
        cols,
        row_gap,
        col_gap,
        height,
    )

    target_indices = getattr(env, "shadow_tube_insert_target_indices", None)
    if target_indices is None:
        raise RuntimeError("Shadow tube insert target was not initialized on reset.")
    target_indices = torch.as_tensor(target_indices, device=env.device, dtype=torch.long)
    if torch.any((target_indices[:, 0] < 0) | (target_indices[:, 0] >= rows)):
        raise ValueError("Shadow tube insert target row is outside the configured rack.")
    if torch.any((target_indices[:, 1] < 0) | (target_indices[:, 1] >= cols)):
        raise ValueError("Shadow tube insert target column is outside the configured rack.")
    flat_target_ids = target_indices[:, 0] * cols + target_indices[:, 1]
    target_tube_pos = target_tube_positions_w[
        torch.arange(env.num_envs, device=env.device),
        flat_target_ids,
    ]

    tube_pos = tube.data.root_link_pose_w[:, :3]
    tube_xy_dist = torch.linalg.norm(tube_pos[:, :2] - target_tube_pos[:, :2], dim=-1)
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


def make_shadow_tube_insert_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    del play

    robot_cfg = copy.deepcopy(get_ur5e_shadow_cfg())
    robot_cfg.init_state.pos = (0.011252234, 0.147835851, 0.824000001)
    robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    robot_cfg.init_state.joint_pos.update(
        {
            "shoulder_pan": 0.00000000,
            "shoulder_lift": -1.633629400,
            "elbow": -1.507920000,
            "wrist_1": -3.015931200,
            "wrist_2": -1.507965600,
            "wrist_3": -0.062831900,
        }
    )
    gs_wrist_cam, gs_front_cam, gs_left_cam = make_gaussian_camera_triplet(
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        scene_parent_body="table/vention table",
        wrist_camera_name="robot/wrist_cam",
        wrist_parent_body="robot/wrist_3_link",
        wrist_width=320,
        wrist_height=240,
        front_pos=(-0.0506275237, -1.7329027, 1.97410775),
        front_quat=(0.848035758, 0.529801155, -0.00154353368, -0.0119877342),
        front_width=320,
        front_height=240,
        left_pos=(-1.34242754, 0.0207416009, 1.77852816),
        left_quat=(0.612746699, 0.337879452, -0.348873132, -0.623431228),
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
            "centrifuge_10slot_1": get_centrifuge_10slot_1_cfg(),
            "centrifuge_10slot_2": get_centrifuge_10slot_2_cfg(),
            "centrifuge_50ml": get_centrifuge_50ml_cfg(),
        },
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
        "reset_centrifuge_pose": EventTermCfg(
            func=reset_centrifuge_on_rack,
            mode="reset",
            params={
                "site_asset_cfg": SceneEntityCfg(
                    "centrifuge_10slot_2",
                    site_names=["centrifuge_10slot-50ml"],
                ),
                "asset_cfg": SceneEntityCfg("centrifuge_50ml"),
                "rows": 5,
                "cols": 2,
                "row_gap": 0.036,
                "col_gap": 0.036,
                "height": 0.0,
            },
        ),
        "reset_target_rack_slot": EventTermCfg(
            func=reset_shadow_tube_insert_target,
            mode="reset",
            params={"rows": 5, "cols": 2},
        ),
        "reset_robot_slider_randomization": EventTermCfg(
            func=reset_joints_and_tendons_by_range,
            mode="reset",
            params={
                "joint_lows": UR_SHADOW_ARM_RESET_JOINT_LOWS
                + UR_SHADOW_HAND_RESET_JOINT_LOWS,
                "joint_highs": UR_SHADOW_ARM_RESET_JOINT_HIGHS
                + UR_SHADOW_HAND_RESET_JOINT_HIGHS,
                "joint_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_SHADOW_ARM_JOINT_ORDER
                    + UR5E_SHADOW_HAND_JOINT_ORDER,
                    preserve_order=True,
                ),
                "joint_action_terms": ("joint_pos",),
                "tendon_lows": UR_SHADOW_RESET_TENDON_LOWS,
                "tendon_highs": UR_SHADOW_RESET_TENDON_HIGHS,
                "tendon_asset_cfg": SceneEntityCfg(
                    "robot",
                    tendon_names=UR5E_SHADOW_HAND_TENDON,
                    preserve_order=True,
                ),
                "tendon_action_terms": ("tendon",),
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
                "tube_cfg": SceneEntityCfg("centrifuge_50ml"),
                "target_site_cfg": SceneEntityCfg(
                    "centrifuge_10slot_1",
                    site_names=["centrifuge_10slot-50ml"],
                ),
                "rows": 5,
                "cols": 2,
                "row_gap": 0.036,
                "col_gap": 0.036,
                "height": 0.0,
                "max_tube_target_xy_dist": 0.015,
                "min_tube_z": 0.829,
                "max_tube_z": 0.831,
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
        decimation=25,
        episode_length_s=SHADOW_TUBE_INSERT_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
