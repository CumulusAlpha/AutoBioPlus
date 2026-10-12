import warnings
from pathlib import Path

import mujoco
import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.envs.mdp.actions.differential_ik import DifferentialIKActionCfg

from assets.mjcf.misc.simple_table import get_table_cfg
from assets.mjcf.object.centrifuge_10slot import get_centrifuge_10slot_1_cfg
from assets.mjcf.object.centrifuge_50ml import get_centrifuge_50ml_cap_cfg, get_centrifuge_50ml_body_cfg
from assets.mjcf.robot import ALOHA_JOINT_ORDER, get_aloha_left_cfg
from src.tasks.config.gaussian_cameras import make_gaussian_camera
from src.tasks.config.gaussian_scenes import gaussian_scene_for_task
from src.tasks.mdp.observations import camera_rgb, joint_pos_abs, site_pose_w
from src.tasks.mdp.events import add_noise_to_joints, reset_centrifuge_on_rack
from src.tasks.success_diagnostics import record_success_diagnostics
from src.utils.quat_util import quat_conjugate, quat_mul, quat_rotate_vec


_GAUSSIAN_SCENE = gaussian_scene_for_task("Aloha-Screw-Tighten")
_TABLE_GEOM_NAMES = tuple(
    f"table_geom_{idx}" for idx, _ in enumerate(get_table_cfg().spec_fn().geoms)
)
_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES = (
    "terrain",
    *(f"table/{name}" for name in _TABLE_GEOM_NAMES),
)
_ALOHA_JOINT_NOISE_LOW = (-0.05, 0.0, 0.0, -0.0125, 0.0, -0.025, 0.0)
_ALOHA_JOINT_NOISE_HIGH = (0.05, 0.075, 0.025, 0.0125, 0.075, 0.025, 0.0)
_ALOHA_SCREW_TIGHTEN_ENV_STEP_S = 0.05
# Conservative timeout budget for the source-faithful three-cycle expert policy.
_ALOHA_SCREW_TIGHTEN_POLICY_DONE_STEPS = 928
_ALOHA_SCREW_TIGHTEN_EPISODE_LENGTH_S = (
    2.0 * _ALOHA_SCREW_TIGHTEN_POLICY_DONE_STEPS * _ALOHA_SCREW_TIGHTEN_ENV_STEP_S
)


def ensure_aloha_screw_table_collision_name(spec: mujoco.MjSpec) -> str:
    """Name this scene's table collision geom without changing shared table XML."""
    table_collision_name = "table/table_collision"
    for geom in spec.geoms:
        if geom.name == table_collision_name:
            return table_collision_name

    target_size = (0.604, 0.401, 0.411)
    target_pos = (0.0, 0.0, 0.413)
    for geom in spec.geoms:
        if geom.type != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if geom.group != 3 or geom.contype != 0 or geom.conaffinity != 1:
            continue
        size_ok = all(abs(float(a) - b) < 1e-9 for a, b in zip(geom.size, target_size))
        pos_ok = all(abs(float(a) - b) < 1e-9 for a, b in zip(geom.pos, target_pos))
        if size_ok and pos_ok:
            geom.name = table_collision_name
            return table_collision_name

    return table_collision_name


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

def add_scene_contact_pairs(spec: mujoco.MjSpec) -> None:
    """Inject explicit contact pairs for the assembled aloha_screw scene."""

    table_collision_name = ensure_aloha_screw_table_collision_name(spec)

    fingers = (
        "aloha_1/rightfinger", "aloha_1/leftfinger",
        "aloha_2/rightfinger", "aloha_2/leftfinger",
    )

    # Base pair templates: (geom1, geom2, friction, solref, solimp)
    pair_defs = []

    pair_defs.append(
        ("screw_cap/total_cap_cylinder", table_collision_name, 1.0, (0.002, 1.0), (0.9, 0.95, 0.001))
    )

    # Cap vs grippers
    pair_defs += [
        ("screw_cap/total_cap_cylinder", f, 1.0, (0.002, 1.0), (0.9, 0.95, 0.001))
        for f in fingers
    ]

    # Body vs grippers
    pair_defs += [
        ("screw_body/body_cyl", f, 1.0, (0.002, 1.0), (0.9, 0.95, 0.001))
        for f in fingers
    ]

    # Thread pairs
    pair_defs += [
        ("screw_cap/nut", "screw_body/bolt", 1.0, (0.002, 1.0), None),
        ("screw_cap/nut", "screw_body/body_cyl", 0.01, (0.002, 1.0), (0.9, 0.95, 0.001)),
    ]

    geom_names = {g.name for g in spec.geoms}

    for g1, g2, mu, solref, solimp in pair_defs:
        if g1 not in geom_names or g2 not in geom_names:
            warnings.warn(
                f"Skip contact pair ({g1}, {g2}): geom missing in scene.",
                stacklevel=2,
            )
            continue

        # Expand friction → 5D
        friction = (mu, mu, 0.005, 0.0001, 0.0001)

        # Expand solimp → 5D if needed
        if solimp is not None and len(solimp) == 3:
            solimp = (*solimp, 0.5, 2.0)

        spec.add_pair(
            geomname1=g1,
            geomname2=g2,
            friction=friction,
            solref=solref,
            solimp=solimp,
        )


def reset_aloha_screw_objects(
    env,
    env_ids: torch.Tensor | None,
    rack_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    cap_cfg: SceneEntityCfg,
    rows: int = 5,
    cols: int = 2,
    row_gap: float = 0.036,
    col_gap: float = 0.036,
    body_height: float = 0.0,
    cap_xy_noise: float = 0.01,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    if cap_xy_noise < 0:
        raise ValueError("cap_xy_noise must be non-negative.")

    cap = env.scene[cap_cfg.name]

    reset_centrifuge_on_rack(
        env=env,
        env_ids=env_ids,
        site_asset_cfg=rack_cfg,
        asset_cfg=body_cfg,
        rows=rows,
        cols=cols,
        row_gap=row_gap,
        col_gap=col_gap,
        height=body_height,
    )

    # Keep the cap near the tuned default pose; only its table-plane position
    # receives a small perturbation so height and orientation remain stable.
    cap_pose = cap.data.default_root_state[env_ids, :7].clone()
    env_origins = env.scene.env_origins[env_ids].to(device=env.device, dtype=cap_pose.dtype)
    cap_pose[:, :3] = cap_pose[:, :3] + env_origins
    cap_pose[:, :2] += torch.empty(
        (env_ids.numel(), 2),
        device=env.device,
        dtype=cap_pose.dtype,
    ).uniform_(-cap_xy_noise, cap_xy_noise)
    cap_vel = torch.zeros_like(cap.data.root_link_vel_w[env_ids])
    cap.write_root_link_pose_to_sim(cap_pose, env_ids=env_ids)
    cap.write_root_link_velocity_to_sim(cap_vel, env_ids=env_ids)

    env.aloha_screw_success_seen = getattr(
        env,
        "aloha_screw_success_seen",
        torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool),
    )
    env.aloha_screw_success_seen[env_ids] = False
    env.aloha_screw_thread_ref_valid = getattr(
        env,
        "aloha_screw_thread_ref_valid",
        torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool),
    )
    env.aloha_screw_thread_prev_rel_quat = getattr(
        env,
        "aloha_screw_thread_prev_rel_quat",
        torch.zeros((env.num_envs, 4), device=env.device, dtype=cap_pose.dtype),
    )
    env.aloha_screw_thread_ref_valid[env_ids] = False
    env.aloha_screw_thread_prev_rel_quat[env_ids] = 0.0
    env.aloha_screw_thread_tracking = getattr(
        env,
        "aloha_screw_thread_tracking",
        torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool),
    )
    env.aloha_screw_thread_tracking[env_ids] = False
    env.aloha_screw_thread_rotation = getattr(
        env,
        "aloha_screw_thread_rotation",
        torch.zeros((env.num_envs,), device=env.device, dtype=cap_pose.dtype),
    )
    env.aloha_screw_thread_rotation[env_ids] = 0.0
    env.aloha_screw_success_stable_count = getattr(
        env,
        "aloha_screw_success_stable_count",
        torch.zeros((env.num_envs,), device=env.device, dtype=torch.long),
    )
    env.aloha_screw_success_stable_count[env_ids] = 0
    env.aloha_screw_tighten_policy_complete = getattr(
        env,
        "aloha_screw_tighten_policy_complete",
        torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool),
    )
    env.aloha_screw_tighten_policy_complete[env_ids] = False
    env.sim.forward()


def check_success(
    env,
    cap_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    max_radial_dist: float,
    min_axial_offset: float,
    max_axial_offset: float,
    max_axis_tilt: float,
    thread_ref_max_radial_dist: float,
    thread_ref_max_axial_offset: float,
    thread_ref_max_axis_tilt: float,
    min_post_seat_rotation: float,
    min_stable_steps: int,
) -> torch.Tensor:
    cap = env.scene[cap_cfg.name]
    body = env.scene[body_cfg.name]

    cap_site_ids = cap_cfg.site_ids
    body_site_ids = body_cfg.site_ids
    assert cap_site_ids is not None, "Please specify cap site_names in SceneEntityCfg"
    assert body_site_ids is not None, "Please specify body site_names in SceneEntityCfg"

    cap_pos = cap.data.site_pose_w[:, cap_site_ids[0], :3]
    body_pos = body.data.site_pose_w[:, body_site_ids[0], :3]
    cap_quat = F.normalize(cap.data.site_pose_w[:, cap_site_ids[0], 3:], dim=-1)
    body_quat = F.normalize(body.data.site_pose_w[:, body_site_ids[0], 3:], dim=-1)

    # Measure the cap in the tube frame. World XY/Z distances are invalid once
    # the tube is grasped and tilted, and previously allowed visibly detached
    # caps to pass the success condition.
    body_quat_inv = quat_conjugate(body_quat)
    cap_pos_body = quat_rotate_vec(body_quat_inv, cap_pos - body_pos)
    radial_dist = torch.linalg.norm(cap_pos_body[:, :2], dim=-1)
    axial_offset = cap_pos_body[:, 2]
    cap_rel_quat = F.normalize(quat_mul(body_quat_inv, cap_quat), dim=-1)

    local_z = torch.zeros_like(cap_pos)
    local_z[:, 2] = 1.0
    cap_axis_w = quat_rotate_vec(cap_quat, local_z)
    body_axis_w = quat_rotate_vec(body_quat, local_z)
    # The cap's +Z axis points down the tube when correctly assembled.
    axis_alignment = torch.clamp(
        -torch.sum(cap_axis_w * body_axis_w, dim=-1),
        min=-1.0,
        max=1.0,
    )
    axis_tilt = torch.acos(axis_alignment)

    ref_valid = getattr(env, "aloha_screw_thread_ref_valid", None)
    prev_rel_quat = getattr(env, "aloha_screw_thread_prev_rel_quat", None)
    was_tracking = getattr(env, "aloha_screw_thread_tracking", None)
    accumulated_rotation = getattr(env, "aloha_screw_thread_rotation", None)
    if ref_valid is None or ref_valid.shape[0] != env.num_envs:
        ref_valid = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    if prev_rel_quat is None or prev_rel_quat.shape != cap_rel_quat.shape:
        prev_rel_quat = torch.zeros_like(cap_rel_quat)
    if was_tracking is None or was_tracking.shape[0] != env.num_envs:
        was_tracking = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    if accumulated_rotation is None or accumulated_rotation.shape[0] != env.num_envs:
        accumulated_rotation = torch.zeros(
            (env.num_envs,),
            device=env.device,
            dtype=cap_rel_quat.dtype,
        )

    thread_reference_ready = (
        (radial_dist <= thread_ref_max_radial_dist)
        & (axial_offset >= min_axial_offset)
        & (axial_offset <= thread_ref_max_axial_offset)
        & (axis_tilt <= thread_ref_max_axis_tilt)
    )
    new_ref = thread_reference_ready & (~ref_valid)
    ref_valid = ref_valid | thread_reference_ready
    env.aloha_screw_thread_ref_valid = ref_valid

    # Accumulate signed twist about the tube's local Z axis while the cap stays
    # in the thread envelope. A single reference quaternion loses full turns,
    # while unsigned increments would incorrectly reward back-and-forth motion.
    continue_tracking = thread_reference_ready & was_tracking & (~new_ref)
    rotation_delta = F.normalize(
        quat_mul(cap_rel_quat, quat_conjugate(prev_rel_quat)),
        dim=-1,
    )
    rotation_delta = torch.where(
        (rotation_delta[:, :1] < 0.0),
        -rotation_delta,
        rotation_delta,
    )
    twist_norm = torch.linalg.norm(rotation_delta[:, (0, 3)], dim=-1)
    twist_w = rotation_delta[:, 0] / torch.clamp(twist_norm, min=1e-8)
    twist_z = rotation_delta[:, 3] / torch.clamp(twist_norm, min=1e-8)
    signed_twist_step = 2.0 * torch.atan2(twist_z, twist_w)
    signed_twist_step = torch.where(
        twist_norm > 1e-8,
        signed_twist_step,
        torch.zeros_like(signed_twist_step),
    )
    accumulated_rotation = accumulated_rotation + torch.where(
        continue_tracking,
        signed_twist_step,
        torch.zeros_like(signed_twist_step),
    )
    prev_rel_quat = torch.where(
        thread_reference_ready.unsqueeze(-1),
        cap_rel_quat,
        prev_rel_quat,
    )
    was_tracking = thread_reference_ready & ref_valid
    env.aloha_screw_thread_prev_rel_quat = prev_rel_quat
    env.aloha_screw_thread_tracking = was_tracking
    env.aloha_screw_thread_rotation = accumulated_rotation
    post_seat_rotation = torch.abs(accumulated_rotation)

    radially_aligned = radial_dist <= max_radial_dist
    axially_seated = (
        (axial_offset >= min_axial_offset)
        & (axial_offset <= max_axial_offset)
    )
    axis_aligned = axis_tilt <= max_axis_tilt
    rotation_complete = post_seat_rotation >= min_post_seat_rotation
    instantaneous_success = (
        ref_valid
        & radially_aligned
        & axially_seated
        & axis_aligned
        & rotation_complete
    )

    stable_count = getattr(env, "aloha_screw_success_stable_count", None)
    if stable_count is None or stable_count.shape[0] != env.num_envs:
        stable_count = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
    stable_count = torch.where(
        instantaneous_success,
        stable_count + 1,
        torch.zeros_like(stable_count),
    )
    env.aloha_screw_success_stable_count = stable_count
    success = instantaneous_success & (stable_count >= min_stable_steps)

    record_success_diagnostics(
        env,
        object_complete=instantaneous_success,
        full_success=success,
        components={
            "thread_reference_seen": ref_valid,
            "radially_aligned": radially_aligned,
            "axially_seated": axially_seated,
            "axis_aligned": axis_aligned,
            "rotation_complete": rotation_complete,
            "stable_complete": stable_count >= min_stable_steps,
        },
        metrics={
            "cap_body_radial_dist_m": radial_dist,
            "cap_body_axial_offset_m": axial_offset,
            "cap_axis_tilt_rad": axis_tilt,
            "post_seat_twist_rad": post_seat_rotation,
            "success_stable_steps": stable_count,
        },
        thresholds={
            "max_cap_body_radial_dist_m": max_radial_dist,
            "min_cap_body_axial_offset_m": min_axial_offset,
            "max_cap_body_axial_offset_m": max_axial_offset,
            "max_cap_axis_tilt_rad": max_axis_tilt,
            "thread_ref_max_radial_dist_m": thread_ref_max_radial_dist,
            "thread_ref_max_axial_offset_m": thread_ref_max_axial_offset,
            "thread_ref_max_axis_tilt_rad": thread_ref_max_axis_tilt,
            "min_post_seat_twist_rad": min_post_seat_rotation,
            "min_success_stable_steps": min_stable_steps,
        },
    )
    seen = getattr(env, "aloha_screw_success_seen", None)
    if seen is None or seen.shape[0] != env.num_envs:
        seen = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    seen |= success
    env.aloha_screw_success_seen = seen
    return seen & success


def make_aloha_screw_tighten_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    gs_wrist_cam = make_gaussian_camera(
        name="gs_wrist_cam_rgb",
        camera_name="aloha_1/wrist_cam_left",
        fovy=58.0,
        width=640,
        height=360,
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )
    gs_front_cam = make_gaussian_camera(
        name="gs_front_cam_rgb",
        parent_body="table/vention table",
        pos=(0.0, -1.2, 1.45),
        quat=(0.819, 0.574, 0.0, 0.0),
        fovy=45.0,
        width=640,
        height=480,
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )
    gs_aloha_2_wrist_cam = make_gaussian_camera(
        name="gs_aloha_2_wrist_cam_rgb",
        camera_name="aloha_2/wrist_cam_left",
        fovy=58.0,
        width=640,
        height=360,
        background_scene=_GAUSSIAN_SCENE,
        background_frame_body_name="table/vention table",
        foreground_exclude_geom_names=_GAUSSIAN_FOREGROUND_EXCLUDE_GEOM_NAMES,
    )

    # --------------------------
    # Scene
    # --------------------------
    aloha_1_cfg = get_aloha_left_cfg()
    aloha_1_cfg.init_state.pos = (0.3, 0.0, 0.824)
    aloha_1_cfg.init_state.rot = (1.0, 0.0, 0.0, -1.0)

    aloha_2_cfg = get_aloha_left_cfg()
    aloha_2_cfg.init_state.pos = (-0.4, 0.0, 0.824)
    aloha_2_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    slot_cfg = get_centrifuge_10slot_1_cfg()
    slot_cfg.init_state.pos = (0.0, 0.0, 0.854)
    slot_cfg.init_state.rot = (1.0, 0.0, 0.0, 1.0)


    scene = SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="plane",
        ),
        entities={
            "aloha_1": aloha_1_cfg,
            "aloha_2": aloha_2_cfg,
            "table": get_gaussian_hidden_table_cfg(),
            "slot1": slot_cfg,
            "screw_cap": get_centrifuge_50ml_cap_cfg(),
            "screw_body": get_centrifuge_50ml_body_cfg(),
        },
        spec_fn=add_scene_contact_pairs,
        num_envs=1,
        env_spacing=4.0,
        sensors=(gs_wrist_cam, gs_front_cam, gs_aloha_2_wrist_cam),
    )

    # --------------------------
    # Observations
    # --------------------------
    observations = {
        "aloha_1_qpos": ObservationGroupCfg(
            {
                "aloha_1_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "aloha_1",
                            joint_names=ALOHA_JOINT_ORDER,
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "aloha_2_qpos": ObservationGroupCfg(
            {
                "aloha_2_qpos": ObservationTermCfg(
                    func=joint_pos_abs,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "aloha_2",
                            joint_names=ALOHA_JOINT_ORDER,
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "aloha_1_eef_pose": ObservationGroupCfg(
            {
                "aloha_1_eef_pose": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "aloha_1",
                            site_names=["gripper"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "aloha_2_eef_pose": ObservationGroupCfg(
            {
                "aloha_2_eef_pose": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "aloha_2",
                            site_names=["gripper"],
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
                            "slot1",
                            site_names=["centrifuge_10slot-50ml"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "screw_cap_pose_w": ObservationGroupCfg(
            {
                "screw_cap_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "screw_cap",
                            site_names=["origin3"],
                        ),
                    },
                ),
            },
            enable_corruption=False,
            history_length=1,
        ),
        "screw_body_pose_w": ObservationGroupCfg(
            {
                "screw_body_pose_w": ObservationTermCfg(
                    func=site_pose_w,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "screw_body",
                            site_names=["origin4"],
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
        "gs_wrist_cam_rgb": ObservationGroupCfg(
            {
                "gs_wrist_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_wrist_cam_rgb"},
                ),
            },
            enable_corruption=False,
        ),
        "gs_aloha_2_wrist_cam_rgb": ObservationGroupCfg(
            {
                "gs_aloha_2_wrist_cam_rgb": ObservationTermCfg(
                    func=camera_rgb,
                    params={"sensor_name": "gs_aloha_2_wrist_cam_rgb"},
                ),
            },
            enable_corruption=False,
        ),
    }

    # --------------------------
    # Actions
    # --------------------------
    actions = {

        # ### Joint position control 
        # "aloha_1_joint_pos": JointPositionActionCfg(
        #     entity_name="aloha_1",
        #     actuator_names=ALOHA_JOINT_ORDER,
        #     use_default_offset=False,
        #     clip=None,
        # ),
        # "aloha_2_joint_pos": JointPositionActionCfg(
        #     entity_name="aloha_2",
        #     actuator_names=ALOHA_JOINT_ORDER,
        #     use_default_offset=False,
        #     clip=None,
        # ),
        ###
        ### End effector control (Differential IK + gripper)
        "aloha_1_eef": DifferentialIKActionCfg(
        entity_name="aloha_1",
        actuator_names=tuple(j for j in ALOHA_JOINT_ORDER if j != "left_finger"),
        frame_type="site",
        frame_name="gripper",
        use_relative_mode=False,
        position_weight=1,
        orientation_weight=0.5,
        ),
        "aloha_1_gripper": JointPositionActionCfg(
            entity_name="aloha_1",
            actuator_names=["left_finger"],
            use_default_offset=False,
            clip=None,
        ),
        "aloha_2_eef": DifferentialIKActionCfg(
        entity_name="aloha_2",
        actuator_names=tuple(j for j in ALOHA_JOINT_ORDER if j != "left_finger"),
        frame_type="site",
        frame_name="gripper",
        use_relative_mode=False,
        position_weight=1,
        orientation_weight=0.5,
        ),
        "aloha_2_gripper": JointPositionActionCfg(
            entity_name="aloha_2",
            actuator_names=["left_finger"],
            use_default_offset=False,
            clip=None,
        ),
    }

    # --------------------------
    # Events
    # --------------------------
    events = {
        "reset_scene": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "reset_aloha_screw_objects": EventTermCfg(
            func=reset_aloha_screw_objects,
            mode="reset",
            params={
                "rack_cfg": SceneEntityCfg(
                    "slot1",
                    site_names=["centrifuge_10slot-50ml"],
                ),
                "body_cfg": SceneEntityCfg(
                    "screw_body",
                ),
                "cap_cfg": SceneEntityCfg(
                    "screw_cap",
                ),
            },
        ),
        "add_noise_to_aloha_1_joints": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": _ALOHA_JOINT_NOISE_LOW,
                "highs": _ALOHA_JOINT_NOISE_HIGH,
                "asset_cfg": SceneEntityCfg(
                    "aloha_1",
                    joint_names=ALOHA_JOINT_ORDER,
                ),
            },
        ),
        "add_noise_to_aloha_2_joints": EventTermCfg(
            func=add_noise_to_joints,
            mode="reset",
            params={
                "lows": _ALOHA_JOINT_NOISE_LOW,
                "highs": _ALOHA_JOINT_NOISE_HIGH,
                "asset_cfg": SceneEntityCfg(
                    "aloha_2",
                    joint_names=ALOHA_JOINT_ORDER,
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
            params={
                "cap_cfg": SceneEntityCfg(
                    "screw_cap",
                    site_names=["origin3"],
                ),
                "body_cfg": SceneEntityCfg(
                    "screw_body",
                    site_names=["origin4"],
                ),
                # The assembled XML gives origin3 relative to origin4 at
                # approximately 0.10231 m. Scripted completions settle near
                # 0.1039 m under contact dynamics.
                "max_radial_dist": 0.003,
                "min_axial_offset": 0.100,
                "max_axial_offset": 0.104,
                "max_axis_tilt": 0.0872665,
                "thread_ref_max_radial_dist": 0.005,
                "thread_ref_max_axial_offset": 0.108,
                "thread_ref_max_axis_tilt": 0.174533,
                "min_post_seat_rotation": 1.0,
                "min_stable_steps": 5,
            },
        ),
    }
    ### a much computing consuming one
    # sim = SimulationCfg(
    # nconmax=4096,
    # njmax=8192,
    # mujoco=MujocoCfg(
    #     timestep=0.001,          # 0.002 -> 0.001
    #     integrator="implicitfast",
    #     solver="newton",
    #     impratio=10,
    #     cone="elliptic",
    #     iterations=200,          # 100 -> 200
    #     tolerance=1e-10,         # 1e-8 -> 1e-10
    #     ls_iterations=100,       # 50 -> 100
    #     ls_tolerance=0.001,      # 0.01 -> 0.001
    #     ccd_iterations=200,      # 100 -> 200
    #     multiccd=True,
    # ),

    # --------------------------
    # Simulation
    # --------------------------
    sim = SimulationCfg(
       nconmax=4096,
       njmax=8192,
       mujoco=MujocoCfg(
        timestep=0.002,
        impratio=10,
        cone="elliptic",         
        ccd_iterations=1000,
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
        episode_length_s=_ALOHA_SCREW_TIGHTEN_EPISODE_LENGTH_S,
        scale_rewards_by_dt=False,
    )
