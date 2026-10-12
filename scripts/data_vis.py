import html
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.viewer.viser.scene import ViserMujocoScene
from utils.viser_util import patch_viser_sdf_geom_rendering
from viser._gui_api import GuiImageHandle

from scripts.utils.data_vis_config import DataVisConfig as DataVisConfig
from scripts.utils.replay_sources import (
    EpisodeData as EpisodeData,
    LeRobotReplaySource as LeRobotReplaySource,
    ReplaySource as ReplaySource,
    ZarrReplaySource as ZarrReplaySource,
    _rgb_frame_shape as _rgb_frame_shape,
    camera_names_for_count as camera_names_for_count,
    load_lerobot_source as load_lerobot_source,
    load_zarr_source as load_zarr_source,
    matrix_from_records as matrix_from_records,
    metadata_records as metadata_records,
    parquet_records as parquet_records,
    row_value as row_value,
)


def default_input_path(task_id: str) -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root
        / "thirdparty"
        / "openpi"
        / "datasets"
        / "dexbiolab_lerobot"
        / task_id,
        repo_root / "logs" / "collect" / task_id / "replay_buffer.zarr",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def main():
    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    cfg = tyro.cli(
        DataVisConfig,
        args=remaining_args,
        default=DataVisConfig(),
        prog=f"{__file__} {chosen_task}",
    )

    env_cfg = load_env_cfg(chosen_task, play=True)
    scene_cfg = env_cfg.scene
    scene_cfg.num_envs = 1
    scene_camera_names = [cam.name for cam in scene_cfg.sensors]

    if cfg.input is not None and cfg.pkl_path is not None:
        raise ValueError("Pass only one of -i/--input or --pkl-path.")

    input_path = cfg.input or default_input_path(chosen_task)
    if input_path is not None and (input_path / "meta" / "info.json").is_file():
        replay = load_lerobot_source(input_path)
    elif input_path is not None:
        replay = load_zarr_source(
            input_path,
            state_key=cfg.state_key,
            use_replay_qpos=cfg.replay,
            action_key=cfg.action_key,
            rgb_key=cfg.rgb_key,
            scene_camera_names=scene_camera_names,
        )
    else:
        raise FileNotFoundError(
            f"No final LeRobot dataset or collection found for {chosen_task!r}. "
            "Pass -i/--input explicitly."
        )

    scene = Scene(scene_cfg, "cpu")
    model = scene.spec.compile()
    data = mujoco.MjData(model)

    server_kwargs = {"label": "trajectory-viewer"}
    if cfg.port is not None:
        server_kwargs["port"] = cfg.port
    server = viser.ViserServer(**server_kwargs)

    patch_viser_sdf_geom_rendering()
    mj_scene = ViserMujocoScene.create(
        server=server,
        mj_model=model,
        num_envs=1,
    )
    mj_scene.camera_tracking_enabled = False

    current_episode = replay.load_episode(0)

    with server.gui.add_folder("Playback"):
        max_episode_idx = replay.num_episodes - 1
        max_step_idx = current_episode.num_steps - 1
        episode_slider = server.gui.add_slider(
            label="episode",
            min=0,
            max=max_episode_idx,
            step=1,
            initial_value=0,
        )
        step_slider = server.gui.add_slider(
            label="step",
            min=0,
            max=max_step_idx,
            step=1,
            initial_value=0,
        )
        autoplay_checkbox = server.gui.add_checkbox("autoplay", initial_value=False)
        episode_text = server.gui.add_text(
            "episode/step", initial_value="", disabled=True
        )
        length_text = server.gui.add_text(
            "episode length", initial_value="", disabled=True
        )
        episode_goal_info = server.gui.add_html("<div></div>")

    img_gui_handles: dict[str, GuiImageHandle] = {}
    initial_images = current_episode.get_images(0)
    with server.gui.add_folder("RGB"):
        for name in replay.camera_names:
            if name in initial_images:
                img_gui_handles[name] = server.gui.add_image(
                    initial_images[name], label=name
                )

    frame_lock = threading.RLock()

    def update_episode_info() -> None:
        goal = (
            "fixed (no variable goal parameters)"
            if current_episode.goal is None
            else ", ".join(
                f"<code>{html.escape(str(name))}={html.escape(str(value))}</code>"
                for name, value in current_episode.goal.items()
            )
        )
        episode_goal_info.content = (
            '<div style="font-size: 0.8rem; line-height: 1.35; overflow-wrap: anywhere; '
            'padding: 0.3rem 0.6rem 0.45rem">'
            f"<div><strong>Prompt:</strong> {html.escape(current_episode.prompt or '')}</div>"
            f"<div><strong>Goal:</strong> {goal}</div>"
            "</div>"
        )

    def load_episode(episode_idx: int) -> None:
        nonlocal current_episode
        with frame_lock:
            episode_idx = int(episode_idx)
            if current_episode.episode_idx != episode_idx:
                current_episode = replay.load_episode(episode_idx)

            step_slider.max = current_episode.num_steps - 1
            length_text.value = str(current_episode.num_steps)
            update_episode_info()
            if int(step_slider.value) >= current_episode.num_steps:
                step_slider.value = current_episode.num_steps - 1

    def update_frame(step_idx: int) -> None:
        with frame_lock:
            step_idx = min(max(int(step_idx), 0), current_episode.num_steps - 1)
            if step_idx != int(step_slider.value):
                step_slider.value = step_idx

            qpos = current_episode.get_qpos(step_idx)
            if not np.all(np.isfinite(qpos)):
                raise ValueError(
                    f"Qpos at episode {current_episode.episode_idx}, step {step_idx} "
                    "contains NaN or inf."
                )
            if qpos.shape[0] > model.nq:
                raise ValueError(
                    f"Recorded qpos width {qpos.shape[0]} exceeds scene model nq={model.nq}."
                )

            data.qpos[:] = model.qpos0
            data.qpos[: qpos.shape[0]] = qpos
            mujoco.mj_forward(model, data)
            mj_scene.update_from_mjdata(data)

            global_frame = current_episode.start_frame + step_idx
            episode_text.value = (
                f"{current_episode.episode_idx} / {step_idx} ({global_frame})"
            )

            for name, image in current_episode.get_images(step_idx).items():
                if name in img_gui_handles:
                    img_gui_handles[name].image = image

    @episode_slider.on_update
    def _(_) -> None:
        load_episode(int(episode_slider.value))
        update_frame(int(step_slider.value))

    @step_slider.on_update
    def _(_) -> None:
        update_frame(int(step_slider.value))

    length_text.value = str(current_episode.num_steps)
    update_episode_info()
    update_frame(0)

    while True:
        if autoplay_checkbox.value:
            with frame_lock:
                step_idx = int(step_slider.value) + 1
                if step_idx >= current_episode.num_steps:
                    episode_idx = int(episode_slider.value) + 1
                    if episode_idx >= replay.num_episodes:
                        episode_idx = 0
                    episode_slider.value = episode_idx
                    load_episode(episode_idx)
                    step_idx = 0
                step_slider.value = step_idx
                update_frame(step_idx)
        time.sleep(cfg.playback_dt)


if __name__ == "__main__":
    main()
