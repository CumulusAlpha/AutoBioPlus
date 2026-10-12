from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.envs.mdp.actions.differential_ik import DifferentialIKActionCfg

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.managers.event_manager import EventTermCfg

from assets.mjcf.robot import (
    UR5E_JOINT_ORDER,
    XHAND_JOINT_ORDER,
    get_ur5e_xhand_cfg,
)
from assets.mjcf.robot.ur5e_xhand import get_spec as get_ur5e_xhand_spec
from assets.mjcf.instrument.thermal_mixer_eppendorf_c import get_mixer_cfg
from assets.mjcf.misc.simple_table import get_table_cfg
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.observations import (
    builtin_sensor_data,
    camera_rgb,
    joint_pos_abs,
    site_pose_w,
)
from src.tasks.success_diagnostics import record_success_diagnostics
from src.tasks.mdp.events import (
    add_noise_to_joints,
    get_mixer_ui_state,
    reset_mixer_ui_target,
    update_mixer_display_texture,
    update_mixer_ui_state,
)


SIM_TIMESTEP = 0.002
SIM_DECIMATION = 25
# Three parameter-button cycles followed by the final lift.
EPISODE_LENGTH_STEPS = 573
EPISODE_LENGTH_S = 2.0 * EPISODE_LENGTH_STEPS * SIM_TIMESTEP * SIM_DECIMATION
MIXER_INITIAL_UI_STATE = (60.0, 25.0, 900.0)
MIXER_SUCCESS_WEIGHTS = (0.3, 0.5, 0.2)


def reset_mixer_success_state(env, env_ids) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    seen = getattr(env, "xhand_mixerpanel_manipulate_success_seen", None)
    if seen is None or seen.shape[0] != env.num_envs:
        seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    seen[env_ids] = False
    env.xhand_mixerpanel_manipulate_success_seen = seen


def check_success(
    env,
) -> torch.Tensor:
    ui_state = get_mixer_ui_state(env)
    target_state = getattr(env, "mixer_ui_target_state", None)
    if ui_state is None or target_state is None:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

    current = torch.stack(
        (
            ui_state["time_s"][: env.num_envs],
            ui_state["temp_c"][: env.num_envs],
            ui_state["freq_rpm"][: env.num_envs],
        ),
        dim=-1,
    ).to(device=env.device, dtype=torch.float32)
    target = torch.as_tensor(target_state, device=env.device, dtype=torch.float32)
    if target.ndim == 1:
        target = target.unsqueeze(0).repeat(env.num_envs, 1)
    target = target[: env.num_envs]

    initial = torch.tensor(
        MIXER_INITIAL_UI_STATE,
        device=env.device,
        dtype=torch.float32,
    )
    target_range = torch.abs(target - initial)
    component_scores = torch.where(
        target_range > 0.0,
        torch.clamp(1.0 - torch.abs(current - target) / target_range, min=0.0),
        torch.ones_like(target_range),
    )
    weights = torch.tensor(
        MIXER_SUCCESS_WEIGHTS,
        device=env.device,
        dtype=torch.float32,
    )
    success_score = torch.clamp(
        torch.sum(component_scores * weights, dim=-1),
        min=0.0,
        max=1.0,
    )
    ui_reached = success_score >= 1.0

    record_success_diagnostics(
        env,
        object_complete=ui_reached,
        full_success=ui_reached,
        metrics={
            "ui_success_score": success_score,
            "ui_time_score": component_scores[:, 0],
            "ui_temperature_score": component_scores[:, 1],
            "ui_frequency_score": component_scores[:, 2],
        },
        thresholds={"ui_success_score": 1.0},
    )
    seen = getattr(env, "xhand_mixerpanel_manipulate_success_seen", None)
    if seen is None or seen.shape[0] != env.num_envs:
        seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    seen |= ui_reached
    env.xhand_mixerpanel_manipulate_success_seen = seen
    return seen & ui_reached


_GAUSSIAN_SCENE = gaussian_scene_for_task("XHand-MixerPanel-Manipulate")


def make_xhand_mixerpanel_manipulate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = get_ur5e_xhand_cfg()
    # Match dexbiolab mani_thermal_mixer UR5e base pose and joint home.
    robot_cfg.init_state.pos = (0.5, -0.0, 0.824)
    robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    robot_cfg.init_state.joint_pos.update(
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -1.5708,
            "elbow": 1.5708,
            "wrist_1": -1.5708,
            "wrist_2": -1.5708,
            "wrist_3": -1.5708,
        }
    )

    gs_wrist_cam, gs_front_cam, gs_left_cam = make_gaussian_camera_triplet(
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        scene_parent_body="table/vention table",
        wrist_camera_name=None,
        wrist_parent_body="robot/xhand:right_hand_link",
        wrist_pos=(0.101312558, -0.218962477, 0.0782291544),
        wrist_quat=(-0.0433631994, 0.0753653321, -0.269549844, 0.959052961),
        wrist_fovy=None,
        wrist_width=640,
        wrist_height=480,
        front_pos=(0.0, -1.5, 1.65),
        front_quat=(0.819, 0.574, 0.0, 0.0),
        front_width=1280,
        front_height=960,
        left_pos=(-1.2, 0.0, 1.65),
        left_quat=(0.579, 0.406, -0.406, -0.579),
        left_width=1280,
        left_height=960,
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
            "table": get_table_cfg(),
            "mixer": get_mixer_cfg(),
        },
        num_envs=1,
        env_spacing=3,
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
                            joint_names=UR5E_JOINT_ORDER + XHAND_JOINT_ORDER,
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        
        "index_tip_pose_w": ObservationGroupCfg(
            {
                "index_tip_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            site_names=[r".*xhand:right_index_tip"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "mixer_button_sites_pose_w": ObservationGroupCfg(
            {
                "mixer_button_sites_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "mixer",
                            site_names=[
                                "button-speed-up",
                                "button-speed-down",
                                "button-temp-up",
                                "button-temp-down",
                                "button-time-up",
                                "button-time-down",
                            ],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "mixer_button_touch": ObservationGroupCfg(
            {
                "mixer_button_touch": ObservationTermCfg(
                    func=builtin_sensor_data,
                    params={
                        "sensor_names": (
                            "mixer/button-speed-up",
                            "mixer/button-speed-down",
                            "mixer/button-temp-up",
                            "mixer/button-temp-down",
                            "mixer/button-time-up",
                            "mixer/button-time-down",
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
        frame_name=r".*xhand:right_index_tip",
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
    "add_noise_to_joints": EventTermCfg(
        func=add_noise_to_joints,
        mode="reset",
        params={
            "lows": (-0.2, 0.0, 0.0, -0.05, 0.0, -0.1),
            "highs": (0.2, 0.3, 0.1, 0.05, 0.3, 0.1),
            "asset_cfg": SceneEntityCfg(
                            "robot",
                            joint_names=UR5E_JOINT_ORDER,
            ),
        },
    ),
    "reset_mixer_ui_target": EventTermCfg(
        func=reset_mixer_ui_target,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "mixer",
                material_names="body-display",
                geom_names="body-display",
            ),
        },
    ),
    "update_mixer_ui_state_on_reset": EventTermCfg(
        func=update_mixer_ui_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "mixer",
                material_names="body-display",
                geom_names="body-display",
            ),
        },
    ),
    "update_mixer_display_on_reset": EventTermCfg(
        func=update_mixer_display_texture,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                            "mixer",
                            material_names="body-display",
                            geom_names="body-display",
                        ),
        },
    ),
    "reset_mixer_success_state": EventTermCfg(
        func=reset_mixer_success_state,
        mode="reset",
    ),
    "update_mixer_ui_state": EventTermCfg(
        func=update_mixer_ui_state,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg(
                "mixer",
                material_names="body-display",
                geom_names="body-display",
            ),
        },
    ),
    "update_mixer_display": EventTermCfg(
        func=update_mixer_display_texture,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg(
                            "mixer",
                            material_names="body-display",
                            geom_names="body-display",
                        ),
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
        ),
    }

    # --------------------------
    # Simulation
    # --------------------------
    sim = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=SIM_TIMESTEP,
            iterations=10,
        ),
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
