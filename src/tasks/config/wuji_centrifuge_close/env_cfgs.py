import copy
from pathlib import Path

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

from assets.mjcf.robot.ur5e_wuji import (
    UR5E_JOINT_ORDER,
    WUJI_JOINT_ORDER,
    get_ur5e_wuji_cfg,
)
from src.tasks.config.gaussian_cameras import make_gaussian_camera_triplet
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.config.shadow_centrifuge_insert.env_cfgs import (
    CENTRIFUGE_50ML_SLOT_SITE_NAMES,
    get_centrifuge_5910_ri_cfg,
    get_centrifuge_table_cfg,
    get_tube_cfg,
)
from src.tasks.mdp.events import add_noise_to_joints
from src.tasks.mdp.observations import (
    body_pose_w,
    camera_rgb,
    joint_pos_abs,
    site_pose_w,
)
from src.tasks.success_diagnostics import (
    record_success_diagnostics,
    stable_success,
)
from src.utils.quat_util import compose_pose_wxyz

_GAUSSIAN_SCENE = gaussian_scene_for_task("Wuji-Centrifuge-Close")
_TABLE_GEOM_NAMES = tuple(
    f"table_geom_{idx}" for idx, _ in enumerate(get_centrifuge_table_cfg().spec_fn().geoms)
)
_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES = (
    "terrain",
    *(f"table/{name}" for name in _TABLE_GEOM_NAMES),
)

# Interpolation includes both endpoints for each of the five policy phases.
WUJI_CENTRIFUGE_CLOSE_POLICY_DONE_STEPS = 121 + 31 + 61 + 181 + 81
WUJI_CENTRIFUGE_CLOSE_ENV_STEP_S = 0.05
WUJI_CENTRIFUGE_CLOSE_EPISODE_LENGTH_S = (
    2.0 * WUJI_CENTRIFUGE_CLOSE_POLICY_DONE_STEPS * WUJI_CENTRIFUGE_CLOSE_ENV_STEP_S
)
WUJI_CENTRIFUGE_LID_LATCH_QPOS_RAD = 1.70
WUJI_CENTRIFUGE_MAX_LID_CLOSED_ERROR_RAD = 0.03
WUJI_CENTRIFUGE_SUCCESS_STABILITY_STEPS = 3
WUJI_CENTRIFUGE_BASE_POS = (-0.30, 0.1, 0.824)
WUJI_CENTRIFUGE_RANDOM_POS_LOW = (-0.025, -0.025, 0.0)
WUJI_CENTRIFUGE_RANDOM_POS_HIGH = (0.025, 0.025, 0.0)
WUJI_INITIAL_ARM_JOINT_NOISE_LOW = (-0.10, -0.10, -0.10, -0.10, -0.10, -0.10)
WUJI_INITIAL_ARM_JOINT_NOISE_HIGH = (0.10, 0.10, 0.10, 0.10, 0.10, 0.10)
WUJI_INITIAL_HAND_JOINT_NOISE = (0.0,) * len(WUJI_JOINT_ORDER)


def centrifuge_lid_lock_eq_id(env) -> int:
    cached = getattr(env, "_wuji_centrifuge_lid_lock_eq_id", None)
    if cached is not None:
        return int(cached)

    model = env.sim.mj_model
    for eq_id in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
        if name == "lid-lock" or (name is not None and name.endswith("/lid-lock")):
            env._wuji_centrifuge_lid_lock_eq_id = int(eq_id)
            return int(eq_id)
    raise RuntimeError("Could not resolve centrifuge lid-lock equality.")


def set_centrifuge_lid_lock_active(
    env,
    env_ids: torch.Tensor,
    active: bool | torch.Tensor,
) -> None:
    lid_lock_eq_id = centrifuge_lid_lock_eq_id(env)
    active_values = torch.as_tensor(active, device=env.device, dtype=torch.bool)
    if active_values.ndim == 0:
        active_values = active_values.expand(env_ids.numel())
    if active_values.shape != (env_ids.numel(),):
        raise ValueError(
            f"Expected {env_ids.numel()} equality values, got {tuple(active_values.shape)}."
        )
    env.sim.data.eq_active[env_ids, lid_lock_eq_id] = active_values

    env0 = torch.nonzero(env_ids == 0, as_tuple=False).flatten()
    if env0.numel() > 0:
        env.sim.mj_data.eq_active[lid_lock_eq_id] = bool(
            active_values[env0[0]].item()
        )


def get_gaussian_hidden_centrifuge_table_cfg() -> EntityCfg:
    table_cfg = get_centrifuge_table_cfg()
    base_spec_fn = table_cfg.spec_fn

    def table_spec_with_geom_names():
        spec = base_spec_fn()
        for geom, name in zip(spec.geoms, _TABLE_GEOM_NAMES):
            geom.name = name
        return spec

    table_cfg.spec_fn = table_spec_with_geom_names
    return table_cfg


def randomize_centrifuge_position(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    pos_low: tuple[float, float, float] = WUJI_CENTRIFUGE_RANDOM_POS_LOW,
    pos_high: tuple[float, float, float] = WUJI_CENTRIFUGE_RANDOM_POS_HIGH,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    asset = env.scene[asset_cfg.name]
    if not asset.is_mocap:
        raise RuntimeError(
            f"Wuji centrifuge randomization requires '{asset_cfg.name}' to be mocap-wrapped."
        )

    default_pose = asset.data.default_root_state[env_ids, :7].clone()
    dtype = default_pose.dtype
    low = torch.tensor(pos_low, device=env.device, dtype=dtype)
    high = torch.tensor(pos_high, device=env.device, dtype=dtype)
    pos_delta = torch.rand((env_ids.numel(), 3), device=env.device, dtype=dtype) * (high - low) + low
    env_origins = env.scene.env_origins[env_ids].to(device=env.device, dtype=dtype)

    pose = default_pose.clone()
    pose[:, :3] = default_pose[:, :3] + pos_delta + env_origins
    asset.write_mocap_pose_to_sim(pose, env_ids=env_ids)
    env.wuji_centrifuge_randomized_pose_w = pose
    env.sim.forward()


def reset_wuji_centrifuge_success_state(env, env_ids) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    complete = getattr(env, "wuji_centrifuge_close_policy_complete", None)
    if complete is None or complete.shape[0] != env.num_envs:
        complete = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    complete[env_ids] = False
    env.wuji_centrifuge_close_policy_complete = complete

    stable_steps = getattr(env, "wuji_centrifuge_close_stable_steps", None)
    if stable_steps is None or stable_steps.shape[0] != env.num_envs:
        stable_steps = torch.zeros(
            (env.num_envs,),
            device=env.device,
            dtype=torch.long,
        )
    stable_steps[env_ids] = 0
    env.wuji_centrifuge_close_stable_steps = stable_steps

    set_centrifuge_lid_lock_active(env, env_ids, False)


def update_wuji_centrifuge_lid_lock(
    env,
    env_ids: torch.Tensor | None,
    centrifuge_cfg: SceneEntityCfg,
    engage_lid_qpos_rad: float,
) -> None:
    """Engage the physical lid latch once the lid reaches its catch point."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    centrifuge = env.scene[centrifuge_cfg.name]
    joint_ids = centrifuge_cfg.joint_ids
    assert joint_ids is not None and len(joint_ids) == 1, "Please specify the lid joint."

    lid_qpos = centrifuge.data.joint_pos[env_ids, joint_ids[0]]
    lid_lock_eq_id = centrifuge_lid_lock_eq_id(env)
    currently_active = env.sim.data.eq_active[env_ids, lid_lock_eq_id]
    should_engage = lid_qpos >= engage_lid_qpos_rad
    set_centrifuge_lid_lock_active(
        env,
        env_ids,
        currently_active | should_engage,
    )


def check_success(
    env,
    centrifuge_cfg: SceneEntityCfg,
    max_lid_closed_error_rad: float,
    stability_steps: int,
) -> torch.Tensor:
    centrifuge = env.scene[centrifuge_cfg.name]
    joint_ids = centrifuge_cfg.joint_ids
    assert joint_ids is not None and len(joint_ids) >= 1, "Please specify the lid joint."
    lid_joint_id = joint_ids[0]
    lid_qpos = centrifuge.data.joint_pos[:, lid_joint_id]
    closed_lid_qpos = centrifuge.data.soft_joint_pos_limits[:, lid_joint_id, 1]
    lid_closed_error = closed_lid_qpos - lid_qpos
    object_complete = lid_closed_error <= max_lid_closed_error_rad
    success, stable_count = stable_success(
        env,
        object_complete,
        state_attr="wuji_centrifuge_close_stable_steps",
        required_steps=stability_steps,
    )

    record_success_diagnostics(
        env,
        object_complete=object_complete,
        full_success=success,
        metrics={
            "lid_qpos_rad": lid_qpos,
            "closed_lid_qpos_rad": closed_lid_qpos,
            "lid_closed_error_rad": lid_closed_error,
            "stable_steps": stable_count,
        },
        thresholds={
            "max_lid_closed_error_rad": max_lid_closed_error_rad,
            "required_stability_steps": stability_steps,
        },
    )
    return success


def reset_tube_in_centrifuge(
    env,
    env_ids,
    centrifuge_cfg: SceneEntityCfg,
    tube_cfg: SceneEntityCfg,
) -> None:
    """Place one 50 ml tube in a fixed centrifuge slot."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    centrifuge = env.scene[centrifuge_cfg.name]
    tube = env.scene[tube_cfg.name]
    site_ids = centrifuge_cfg.site_ids
    assert site_ids is not None, "Please specify a centrifuge slot site."

    env.sim.forward()
    slot_pose = centrifuge.data.site_pose_w[env_ids][:, site_ids[0]]
    relative_pose = torch.tensor(
        [0.0, 0.0, -0.03, 1.0, 0.0, 0.0, -1.0],
        device=env.device,
        dtype=slot_pose.dtype,
    )
    relative_pose[3:] /= torch.linalg.norm(relative_pose[3:])
    tube_pose = compose_pose_wxyz(
        slot_pose,
        relative_pose.unsqueeze(0).expand(env_ids.numel(), -1),
    )
    tube.write_root_link_pose_to_sim(tube_pose, env_ids=env_ids)
    tube.write_root_link_velocity_to_sim(
        torch.zeros_like(tube.data.root_link_vel_w[env_ids]),
        env_ids=env_ids,
    )
    env.sim.forward()


def make_wuji_centrifuge_close_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    del play

    robot_cfg = copy.deepcopy(get_ur5e_wuji_cfg())
    # Use the calibrated Wuji centrifuge-close base pose, with the insert arm qpos.
    robot_cfg.init_state.pos = (0.749019563, 0.265338898, 0.824000001)
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
        background_frame_body_name="table/vention big table",
        scene_parent_body="table/vention big table",
        wrist_camera_name="robot/wuji:wrist_cam",
        front_pos=(-0.0506275237, -1.7329027, 1.97410775),
        front_quat=(0.848035758, 0.529801155, -0.00154353368, -0.0119877342),
        left_pos=(-1.52111104, 0.256739549, 1.9016409),
        left_quat=(0.601758845, 0.293769001, -0.366938899, -0.645710393),
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )

    centrifuge_cfg = get_centrifuge_5910_ri_cfg()
    centrifuge_cfg.init_state.pos = WUJI_CENTRIFUGE_BASE_POS

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": robot_cfg,
            "table": get_gaussian_hidden_centrifuge_table_cfg(),
            "centrifuge": centrifuge_cfg,
            "tube": get_tube_cfg(),
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
        "randomize_centrifuge_position": EventTermCfg(
            func=randomize_centrifuge_position,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("centrifuge"),
            },
        ),
        "reset_tube_in_centrifuge": EventTermCfg(
            func=reset_tube_in_centrifuge,
            mode="reset",
            params={
                "centrifuge_cfg": SceneEntityCfg(
                    "centrifuge",
                    site_names=[CENTRIFUGE_50ML_SLOT_SITE_NAMES[0]],
                ),
                "tube_cfg": SceneEntityCfg("tube"),
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
        "reset_wuji_hand_joints": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": WUJI_INITIAL_HAND_JOINT_NOISE,
                "highs": WUJI_INITIAL_HAND_JOINT_NOISE,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=WUJI_JOINT_ORDER,
                ),
                "joint_action_terms": ("wuji_joint_pos",),
            },
        ),
        "reset_wuji_centrifuge_success_state": EventTermCfg(
            func=reset_wuji_centrifuge_success_state,
            mode="reset",
        ),
        "update_wuji_centrifuge_lid_lock": EventTermCfg(
            func=update_wuji_centrifuge_lid_lock,
            mode="step",
            params={
                "centrifuge_cfg": SceneEntityCfg(
                    "centrifuge",
                    joint_names=["lid"],
                ),
                "engage_lid_qpos_rad": WUJI_CENTRIFUGE_LID_LATCH_QPOS_RAD,
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
                "centrifuge_cfg": SceneEntityCfg(
                    "centrifuge",
                    joint_names=["lid"],
                ),
                "max_lid_closed_error_rad": (
                    WUJI_CENTRIFUGE_MAX_LID_CLOSED_ERROR_RAD
                ),
                "stability_steps": WUJI_CENTRIFUGE_SUCCESS_STABILITY_STEPS,
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
        episode_length_s=WUJI_CENTRIFUGE_CLOSE_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
