import copy
from pathlib import Path
from typing import Literal

import mujoco
import torch
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

from assets.mjcf.instrument.thermal_cycler_biorad_c1000 import (
    get_cycler_cfg,
    get_cycler_spec,
)
from assets.mjcf.misc.simple_table import get_table_cfg
from assets.mjcf.robot import UR5E_JOINT_ORDER, XHAND_JOINT_ORDER, get_ur5e_xhand_cfg
from assets.mjcf.robot.ur5e_xhand import get_spec as get_ur5e_xhand_spec
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.events import add_noise_to_joints
from src.tasks.mdp.observations import camera_rgb, joint_pos_abs, site_pose_w
from src.tasks.success_diagnostics import (
    record_success_diagnostics,
    stable_success,
)

ThermalCyclerMode = Literal["close", "open"]

_GAUSSIAN_SCENE = gaussian_scene_for_task("XHand-ThermalCycler-Close")

_ARM_HOME = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.5708,
    "elbow": 1.5708,
    "wrist_1": -1.5708,
    "wrist_2": -1.5708,
    "wrist_3": -1.5708,
}

# The cycler sits at (0.0, 0.05, 0.824). Use the same base pose for open and close.
_ROBOT_BASE_POS = (-0.53, -0.06, 0.824)
_ROBOT_BASE_ROT = (0.0, 0.0, 0.0, 1.0)
_ROBOT_ARM_RANDOM_LOW = (-0.16, -0.16, -0.16, -0.16, -0.16, -0.16)
_ROBOT_ARM_RANDOM_HIGH = (0.16, 0.16, 0.16, 0.16, 0.16, 0.16)
_CYCLER_RANDOM_POS_LOW = (-0.08, 0.0, 0.0)
_CYCLER_RANDOM_POS_HIGH = (0.0, 0.0, 0.0)
_WRIST_CAM_PARENT_BODY = "xhand:right_hand_link"
_WRIST_CAM_POS = (-0.0118319995, 0.0665813929, 0.0618941082)
_WRIST_CAM_QUAT = (-0.0268727319, -0.163885006, -0.0139188419, 0.986015125)
_WRIST_CAM_FOVY = 58.0
_THERMAL_CYCLER_POLICY_DONE_STEPS: dict[ThermalCyclerMode, int] = {
    "close": 792,
    "open": 702,
}
_THERMAL_CYCLER_ENV_STEP_S = 0.05
_THERMAL_CYCLER_EPISODE_LENGTH_S: dict[ThermalCyclerMode, float] = {
    mode: max(30.0, steps * _THERMAL_CYCLER_ENV_STEP_S * 2.0)
    for mode, steps in _THERMAL_CYCLER_POLICY_DONE_STEPS.items()
}

_XHAND_OPEN = {joint_name: 0.0 for joint_name in XHAND_JOINT_ORDER}

_CYCLER_INITIAL_JOINTS: dict[ThermalCyclerMode, dict[str, float]] = {
    # DexBioLab thermal_cycler_close starts from the model default: lid and lever open.
    "close": {
        "lid": 0.0,
        "lid-lever": 0.0,
        "lid-force-knob": 0.0,
    },
    # DexBioLab thermal_cycler_open first places the cycler in the closed/latched state.
    "open": {
        "lid": -1.9,
        "lid-lever": -0.94,
        "lid-force-knob": 1.5707963267948966,
    },
}
_CYCLER_JOINT_NAMES = ("lid", "lid-lever", "lid-force-knob")
_CYCLER_SUCCESS_JOINT_LOW: dict[ThermalCyclerMode, dict[str, float]] = {
    "close": {
        "lid": -2.05,
        "lid-lever": -1.08,
        "lid-force-knob": 1.40,
    },
    "open": {
        "lid": -0.25,
        "lid-lever": -0.20,
        "lid-force-knob": -0.20,
    },
}
_CYCLER_SUCCESS_JOINT_HIGH: dict[ThermalCyclerMode, dict[str, float]] = {
    "close": {
        "lid": -1.70,
        "lid-lever": -0.75,
        "lid-force-knob": 1.75,
    },
    "open": {
        "lid": 0.20,
        "lid-lever": 0.20,
        "lid-force-knob": 0.30,
    },
}

_HIGH_GRIP_FRICTION = (4.0, 0.08, 0.008)


def randomize_thermal_cycler_objects(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    pos_low: tuple[float, float, float],
    pos_high: tuple[float, float, float],
    joint_pos: tuple[float, float, float],
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    cycler = env.scene[asset_cfg.name]
    joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
    default_pose = cycler.data.default_root_state[env_ids, :7].clone()
    dtype = default_pose.dtype
    low = torch.tensor(pos_low, device=env.device, dtype=dtype)
    high = torch.tensor(pos_high, device=env.device, dtype=dtype)
    pos_delta = torch.rand((env_ids.numel(), 3), device=env.device, dtype=dtype) * (high - low) + low
    env_origins = env.scene.env_origins[env_ids].to(device=env.device, dtype=dtype)

    pose = default_pose.clone()
    pose[:, :3] = default_pose[:, :3] + pos_delta + env_origins
    cycler.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    qpos = torch.tensor(joint_pos, device=env.device, dtype=cycler.data.joint_pos.dtype)
    qpos = qpos.unsqueeze(0).expand(env_ids.numel(), -1)
    qvel = torch.zeros_like(qpos)
    cycler.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids, joint_ids=joint_ids)
    env.sim.forward()


def check_success(
    env,
    asset_cfg: SceneEntityCfg,
    joint_low: tuple[float, float, float],
    joint_high: tuple[float, float, float],
    stability_steps: int,
) -> torch.Tensor:
    cycler = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    assert joint_ids is not None, "Please specify the thermal cycler joints."

    current = cycler.data.joint_pos[:, joint_ids]
    low = torch.tensor(
        joint_low,
        device=env.device,
        dtype=current.dtype,
    ).unsqueeze(0)
    high = torch.tensor(
        joint_high,
        device=env.device,
        dtype=current.dtype,
    ).unsqueeze(0)
    joints_in_range = (current >= low) & (current <= high)
    joints_complete = torch.all(joints_in_range, dim=-1)
    success, stable_count = stable_success(
        env,
        joints_complete,
        state_attr="xhand_thermal_cycler_stable_steps",
        required_steps=stability_steps,
    )
    record_success_diagnostics(
        env,
        object_complete=joints_complete,
        full_success=success,
        components={
            f"joint_{index}_in_range": joints_in_range[:, index]
            for index in range(current.shape[1])
        },
        metrics={
            f"joint_{index}_qpos_rad": current[:, index]
            for index in range(current.shape[1])
        } | {"stable_steps": stable_count},
        thresholds={
            **{
                f"joint_{index}_low_rad": value
                for index, value in enumerate(joint_low)
            },
            **{
                f"joint_{index}_high_rad": value
                for index, value in enumerate(joint_high)
            },
        },
    )
    return success


def disable_spec_contacts(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    return spec


def get_xhand_thermalcycler_spec(disable_contacts: bool = False) -> mujoco.MjSpec:
    spec = get_ur5e_xhand_spec()
    spec.delete(spec.camera("wrist_cam"))
    spec.body(_WRIST_CAM_PARENT_BODY).add_camera(
        name="wrist_cam",
        pos=_WRIST_CAM_POS,
        quat=_WRIST_CAM_QUAT,
        fovy=_WRIST_CAM_FOVY,
    )
    if disable_contacts:
        return disable_spec_contacts(spec)
    for geom in spec.geoms:
        geom_name = geom.name.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if not geom_name.startswith("collision_hand_right_"):
            continue
        geom.contype = 1
        geom.friction = _HIGH_GRIP_FRICTION
    return spec


def get_xhand_thermalcycler_cycler_spec(disable_contacts: bool = False) -> mujoco.MjSpec:
    spec = get_cycler_spec()
    for joint in spec.joints:
        if joint.name.rsplit("/", 1)[-1] == "lid-lever":
            joint.springdamper = (0.0, 0.0)
            joint.damping = 0.02
            joint.frictionloss = 0.02
    if disable_contacts:
        return disable_spec_contacts(spec)
    for geom in spec.geoms:
        if geom.group == 3:
            geom.friction = _HIGH_GRIP_FRICTION
    return spec


def make_robot_cfg(disable_contacts: bool):
    robot_cfg = copy.deepcopy(get_ur5e_xhand_cfg())
    robot_cfg.spec_fn = lambda: get_xhand_thermalcycler_spec(disable_contacts=disable_contacts)
    robot_cfg.init_state.pos = _ROBOT_BASE_POS
    robot_cfg.init_state.rot = _ROBOT_BASE_ROT
    robot_cfg.init_state.joint_pos.update(_ARM_HOME)
    robot_cfg.init_state.joint_pos.update(_XHAND_OPEN)
    return robot_cfg


def make_cycler_cfg(mode: ThermalCyclerMode, disable_contacts: bool):
    cycler_cfg = get_cycler_cfg()
    cycler_cfg.spec_fn = lambda: get_xhand_thermalcycler_cycler_spec(disable_contacts=disable_contacts)
    cycler_cfg.init_state.joint_pos.update(_CYCLER_INITIAL_JOINTS[mode])
    return cycler_cfg


def make_table_cfg(disable_contacts: bool):
    table_cfg = get_table_cfg()
    if disable_contacts:
        base_spec_fn = table_cfg.spec_fn
        table_cfg.spec_fn = lambda: disable_spec_contacts(base_spec_fn())
    return table_cfg


def make_xhand_thermalcycler_env_cfg(
    play: bool = False,
    mode: ThermalCyclerMode = "close",
    disable_contacts: bool = False,
) -> ManagerBasedRlEnvCfg:
    if mode not in _CYCLER_INITIAL_JOINTS:
        raise ValueError(f"Unsupported thermal cycler XHand mode: {mode!r}")

    wrist_cam_size = (640, 360) if play else (1280, 720)
    world_cam_size = (640, 480) if play else (1280, 960)

    gs_wrist_cam, gs_front_cam, gs_left_cam = make_gaussian_camera_triplet(
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        scene_parent_body="table/vention table",
        wrist_camera_name="robot/wrist_cam",
        wrist_parent_body="robot/wrist_3_link",
        wrist_width=wrist_cam_size[0],
        wrist_height=wrist_cam_size[1],
        front_pos=(0.0199307693, -1.28108239, 1.4354398),
        front_quat=(0.799207466, 0.600732511, 0.0162787512, 0.0110850336),
        front_width=world_cam_size[0],
        front_height=world_cam_size[1],
        left_pos=(-1.38908104, -0.00645868605, 1.3656858),
        left_quat=(0.56963666, 0.430158293, -0.422042829, -0.558889764),
        left_width=world_cam_size[0],
        left_height=world_cam_size[1],
    )

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": make_robot_cfg(disable_contacts=disable_contacts),
            "thermal_cycler": make_cycler_cfg(mode, disable_contacts=disable_contacts),
            "table": make_table_cfg(disable_contacts=disable_contacts),
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
                            joint_names=UR5E_JOINT_ORDER + XHAND_JOINT_ORDER,
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "cycler_qpos": ObservationGroupCfg(
            {
                "cycler_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "thermal_cycler",
                            joint_names=list(_CYCLER_JOINT_NAMES),
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "palm_pose_w": ObservationGroupCfg(
            {
                "palm_pose_w": ObservationTermCfg(
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
        "thumb_tip_pose_w": ObservationGroupCfg(
            {
                "thumb_tip_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            site_names=[r".*xhand:right_thumb_tip"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "lever_site_pose_w": ObservationGroupCfg(
            {
                "lever_site_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "thermal_cycler",
                            site_names=["lid-lever"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "knob_site_pose_w": ObservationGroupCfg(
            {
                "knob_site_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "thermal_cycler",
                            site_names=["lid-force-knob"],
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
        "palm_pose": DifferentialIKActionCfg(
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

    events = {
        "reset_scene": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "randomize_robot_joints": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": _ROBOT_ARM_RANDOM_LOW,
                "highs": _ROBOT_ARM_RANDOM_HIGH,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=UR5E_JOINT_ORDER,
                ),
            },
        ),
        "randomize_objects": EventTermCfg(
            func=randomize_thermal_cycler_objects,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "thermal_cycler",
                    joint_names=list(_CYCLER_JOINT_NAMES),
                ),
                "pos_low": _CYCLER_RANDOM_POS_LOW,
                "pos_high": _CYCLER_RANDOM_POS_HIGH,
                "joint_pos": tuple(_CYCLER_INITIAL_JOINTS[mode][name] for name in _CYCLER_JOINT_NAMES),
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
                "asset_cfg": SceneEntityCfg(
                    "thermal_cycler",
                    joint_names=list(_CYCLER_JOINT_NAMES),
                ),
                "joint_low": tuple(
                    _CYCLER_SUCCESS_JOINT_LOW[mode][name] for name in _CYCLER_JOINT_NAMES
                ),
                "joint_high": tuple(
                    _CYCLER_SUCCESS_JOINT_HIGH[mode][name] for name in _CYCLER_JOINT_NAMES
                ),
                "stability_steps": 3,
            },
        ),
    }

    sim = SimulationCfg(
        njmax=128,
        mujoco=MujocoCfg(
            timestep=0.002,
            impratio=10,
            cone="elliptic",
            ccd_iterations=50,
            multiccd=True,
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
        episode_length_s=_THERMAL_CYCLER_EPISODE_LENGTH_S[mode],
        scale_rewards_by_dt=False,
    )
