from __future__ import annotations

import sys
from functools import partial

import mujoco_warp as mjw
import torch
import tyro
import viser
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from mujoco_warp._src.types import vec_pluginattr
from utils.sdf_plugin import thread, thread_sdf_grad
from utils.viser_util import patch_viser_sdf_geom_rendering

import src.tasks  # noqa: F401
from src.task_goals import TASK_PROMPT_TEMPLATES
from scripts.utils.gaussian_viewer import (
    PARTICLE_GEOM_MARKERS as PARTICLE_GEOM_MARKERS,
    REPO_ROOT as REPO_ROOT,
    SCREENSHOT_ROOT as SCREENSHOT_ROOT,
    SH_C0 as SH_C0,
    GaussianSplats as GaussianSplats,
    GaussianViserPlayViewer,
    ViserParticleGaussianOverlay as ViserParticleGaussianOverlay,
    _camera_pose as _camera_pose,
    _find_background_frame as _find_background_frame,
    _install_gaussian_overlay as _install_gaussian_overlay,
    _load_gaussian_splats as _load_gaussian_splats,
    _matrix_to_quat_wxyz as _matrix_to_quat_wxyz,
    _next_capture_index as _next_capture_index,
    _quat_wxyz_to_matrix as _quat_wxyz_to_matrix,
    _sigmoid as _sigmoid,
)
from scripts.utils.manual_policies import (
    PolicyNoOp as PolicyNoOp,
    PolicyRandom as PolicyRandom,
    PolicyViserSlider as PolicyViserSlider,
)
from scripts.utils.play_config import (
    GaussianPlayConfig as GaussianPlayConfig,
    PlayConfig as PlayConfig,
)
from scripts.utils.play_viewer import (
    DexBioLabViserPlayViewer as DexBioLabViserPlayViewer,
    PlayTaskGoalDisplay as PlayTaskGoalDisplay,
    reset_policy_after_env_done as reset_policy_after_env_done,
)
from src.tasks.config.gaussian_scenes import (
    TASK_GAUSSIAN_SCENE_CONFIGS,
    gaussian_scene_for_task,
)


@wp.func
def _user_sdf(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> float:
    return thread(p, attr)


@wp.func
def _user_sdf_grad(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> wp.vec3:
    return thread_sdf_grad(p, attr)


mjw._src.collision_sdf.user_sdf = _user_sdf
mjw._src.collision_sdf.user_sdf_grad = _user_sdf_grad


def replace_ik_actions_with_joint_sliders(env_cfg):
    from mjlab.envs.mdp.actions.differential_ik import DifferentialIKActionCfg

    from src.tasks.mdp.actions import OrderedJointPositionActionCfg

    for action_name, action_cfg in list(env_cfg.actions.items()):
        if not isinstance(action_cfg, DifferentialIKActionCfg):
            continue

        env_cfg.actions[action_name] = OrderedJointPositionActionCfg(
            entity_name=action_cfg.entity_name,
            actuator_names=action_cfg.actuator_names,
            use_default_offset=False,
            preserve_order=True,
            clip=None,
        )


def is_gaussian_sensor_cfg(sensor_cfg) -> bool:
    from mugs_mjlab.sensors import GaussianSensorMjlabCfg

    if isinstance(sensor_cfg, GaussianSensorMjlabCfg):
        return True
    cls = sensor_cfg.__class__
    return cls.__name__ == "GaussianSensorMjlabCfg" or (
        cls.__module__.startswith("mugs_mjlab.") and "Gaussian" in cls.__name__
    )


def make_camera_sensor_cfg_from_gaussian(gaussian_sensor_cfg):
    if getattr(gaussian_sensor_cfg, "intrinsics", None) is not None:
        gaussian_sensor_cfg.render_mode = "mujoco_only"
        gaussian_sensor_cfg.background_ply_path = None
        return gaussian_sensor_cfg

    from mjlab.sensor import CameraSensorCfg

    return CameraSensorCfg(
        name=gaussian_sensor_cfg.name,
        camera_name=gaussian_sensor_cfg.camera_name,
        parent_body=gaussian_sensor_cfg.parent_body,
        pos=gaussian_sensor_cfg.pos,
        quat=gaussian_sensor_cfg.quat,
        fovy=gaussian_sensor_cfg.fovy,
        width=gaussian_sensor_cfg.width,
        height=gaussian_sensor_cfg.height,
        data_types=gaussian_sensor_cfg.data_types,
        use_textures=gaussian_sensor_cfg.use_textures,
        use_shadows=gaussian_sensor_cfg.use_shadows,
        enabled_geom_groups=gaussian_sensor_cfg.enabled_geom_groups,
        orthographic=gaussian_sensor_cfg.orthographic,
        clone_data=gaussian_sensor_cfg.clone_data,
    )


def use_standard_cameras_instead_of_gaussian(env_cfg) -> int:
    converted = 0
    sensors = []
    for sensor_cfg in getattr(env_cfg.scene, "sensors", ()):
        if is_gaussian_sensor_cfg(sensor_cfg):
            sensors.append(make_camera_sensor_cfg_from_gaussian(sensor_cfg))
            converted += 1
        else:
            sensors.append(sensor_cfg)
    env_cfg.scene.sensors = tuple(sensors)
    return converted


def count_gaussian_sensors(env_cfg) -> int:
    return sum(
        1
        for sensor_cfg in getattr(env_cfg.scene, "sensors", ())
        if is_gaussian_sensor_cfg(sensor_cfg)
    )


def run_play(
    task_id: str, cfg: PlayConfig, *, viewer_cls=DexBioLabViserPlayViewer
) -> None:
    agent = cfg.agent
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs

    if cfg.gaussian:
        print(
            "Gaussian rendering enabled; "
            f"using {count_gaussian_sensors(env_cfg)} Gaussian camera(s)."
        )
    else:
        converted_sensors = use_standard_cameras_instead_of_gaussian(env_cfg)
        if converted_sensors:
            print(
                "Gaussian rendering disabled; "
                f"using standard cameras for {converted_sensors} camera(s)."
            )

    if agent == "slider":
        replace_ik_actions_with_joint_sliders(env_cfg)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env)
    action_shape = env.unwrapped.action_space.shape
    server = viser.ViserServer()

    if agent == "random":
        policy = PolicyRandom(action_shape, device)
    elif agent == "slider":
        if action_shape[-1] == 0:
            policy = PolicyNoOp(action_shape, device)
        else:
            policy = PolicyViserSlider(action_shape, device, server, env.unwrapped)
    elif agent == "eef":
        policy = PolicyViserEEF(action_shape, device, server, env.unwrapped)
    elif agent == "auto":
        from src.policies import make_scene_policy

        policy = make_scene_policy(
            task_id,
            action_shape=action_shape,
            device=device,
            env=env.unwrapped,
            server=server,
            replay_path=cfg.replay_path,
        )
    else:
        raise RuntimeError(f"Unsupported generic agent type: {agent!r}")

    goal_display = PlayTaskGoalDisplay(
        task_id,
        env,
        server if cfg.viewer == "viser" else None,
    )
    reset_policy_after_env_done(env, policy, on_env_reset=goal_display.refresh_envs)

    if cfg.viewer == "viser":
        patch_viser_sdf_geom_rendering()
        viewer_cls(
            env,
            policy,
            goal_display=goal_display,
            viser_server=server,
        ).run()
    elif cfg.viewer == "mujoco":
        NativeMujocoViewer(env, policy).run()
    else:
        raise RuntimeError("Unsupported viewer type")

    env.close()


def list_dexbiolab_tasks() -> list[str]:
    return [task_id for task_id in list_tasks() if task_id in TASK_PROMPT_TEMPLATES]


def run_play_gaussian(task_id: str, cfg: GaussianPlayConfig) -> None:
    scene_cfg = gaussian_scene_for_task(task_id)
    ply_path = scene_cfg.ply_path.resolve()
    splats = _load_gaussian_splats(ply_path, cfg.max_points, cfg.splat_scale)
    frame_body_name = _find_background_frame(task_id, ply_path)

    play_cfg = PlayConfig(
        agent=cfg.agent,
        viewer=cfg.viewer,
        num_envs=cfg.num_envs,
        device=cfg.device,
        replay_path=cfg.replay_path,
        gaussian=False,
    )
    run_play(
        task_id,
        play_cfg,
        viewer_cls=partial(
            GaussianViserPlayViewer,
            task_id=task_id,
            scene_cfg=scene_cfg,
            frame_body_name=frame_body_name,
            splats=splats,
            gaussian_cfg=cfg,
        ),
    )


def main() -> None:
    # Keep one public entry point for every DexBioLab task. Tasks with a registered
    # Gaussian scene get the calibrated splat overlay; other tasks use the
    # shared standard play implementation.
    task_ids = tuple(sorted(list_dexbiolab_tasks()))
    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(task_ids),
        add_help=False,
        return_unknown_args=True,
    )
    cfg = tyro.cli(
        GaussianPlayConfig,
        args=remaining_args,
        default=GaussianPlayConfig(),
        prog=sys.argv[0] + f" {task_id}",
    )
    if task_id in TASK_GAUSSIAN_SCENE_CONFIGS and cfg.viewer == "viser":
        run_play_gaussian(task_id, cfg)
    else:
        run_play(
            task_id,
            PlayConfig(
                agent=cfg.agent,
                viewer=cfg.viewer,
                num_envs=cfg.num_envs,
                device=cfg.device,
                replay_path=cfg.replay_path,
                gaussian=False,
            ),
        )


if __name__ == "__main__":
    main()
