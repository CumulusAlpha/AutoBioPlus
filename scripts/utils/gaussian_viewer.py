"""Gaussian background overlays, particle display and frame capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mjlab.tasks.registry import load_env_cfg
from PIL import Image
from plyfile import PlyData

from scripts.utils.play_config import GaussianPlayConfig
from scripts.utils.play_viewer import DexBioLabViserPlayViewer
from src.tasks.config.gaussian_scenes import GaussianSceneCfg

REPO_ROOT = Path(__file__).resolve().parents[2]

SCREENSHOT_ROOT = REPO_ROOT / "logs" / "play_gaussian"

SH_C0 = 0.28209479177387814

PARTICLE_GEOM_MARKERS = ("fire_particle_", "fog_particle_")


class GaussianViserPlayViewer(DexBioLabViserPlayViewer):
    """Play viewer with a configured Gaussian background and capture controls."""

    def __init__(
        self, *args, task_id, scene_cfg, frame_body_name, splats, gaussian_cfg, **kwargs
    ) -> None:
        self._gaussian_task_id = task_id
        self._gaussian_scene_cfg = scene_cfg
        self._gaussian_frame_body_name = frame_body_name
        self._gaussian_splats = splats
        self._gaussian_cfg = gaussian_cfg
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        if self._gaussian_cfg.start_paused:
            self._is_paused = True
        super().setup()
        _install_gaussian_overlay(
            self,
            self._gaussian_task_id,
            self._gaussian_scene_cfg,
            self._gaussian_frame_body_name,
            self._gaussian_splats,
            self._gaussian_cfg,
        )


@dataclass(frozen=True)
class GaussianSplats:
    centers: np.ndarray
    covariances: np.ndarray
    rgbs: np.ndarray
    opacities: np.ndarray


@dataclass
class ViserParticleGaussianOverlay:
    """Mirror render-only Warp particles into a Viser Gaussian splat handle."""

    viewer: object
    geom_ids: np.ndarray
    body_ids: tuple[int, ...]
    handle: object

    @classmethod
    def create(cls, viewer) -> ViserParticleGaussianOverlay | None:
        model = viewer.env.unwrapped.sim.mj_model
        geom_ids = np.asarray(
            [
                geom_id
                for geom_id in range(model.ngeom)
                if any(
                    marker in (model.geom(geom_id).name or "")
                    for marker in PARTICLE_GEOM_MARKERS
                )
            ],
            dtype=np.int32,
        )
        if len(geom_ids) == 0:
            return None

        body_ids = tuple(
            sorted({int(model.geom_bodyid[geom_id]) for geom_id in geom_ids})
        )
        zeros = np.zeros((len(geom_ids), 3), dtype=np.float32)
        handle = viewer._server.scene.add_gaussian_splats(
            "/particles/gaussian_splats",
            centers=zeros,
            covariances=np.zeros((len(geom_ids), 3, 3), dtype=np.float32),
            rgbs=np.ones_like(zeros),
            opacities=np.zeros((len(geom_ids), 1), dtype=np.float32),
        )
        overlay = cls(viewer, geom_ids, body_ids, handle)
        overlay._hide_placeholder_meshes(model)
        return overlay

    def _hide_placeholder_meshes(self, model: mujoco.MjModel) -> None:
        particle_body_ids = set(self.body_ids)
        for (
            body_id,
            _group_id,
            _sub_idx,
        ), handle in self.viewer._scene.mesh_handles_by_group.items():
            if body_id in particle_body_ids:
                handle.visible = False
        for body_id in self.body_ids:
            body_name = model.body(body_id).name or f"body_{body_id}"
            self.viewer._server.scene.remove_by_name(f"/fixed_bodies/{body_name}")

    @staticmethod
    def _selected_world(values: np.ndarray, env_idx: int) -> np.ndarray:
        return values[env_idx] if values.ndim == 3 else values

    def update(self, wp_data, env_idx: int) -> None:
        wp_model = self.viewer.env.unwrapped.sim.wp_model
        positions = self._selected_world(wp_data.geom_xpos.cpu().numpy(), env_idx)
        sizes = self._selected_world(wp_model.geom_size.cpu().numpy(), env_idx)
        rgba = self._selected_world(wp_model.geom_rgba.cpu().numpy(), env_idx)

        centers = np.asarray(positions[self.geom_ids], dtype=np.float32)
        radii = np.asarray(sizes[self.geom_ids, 0], dtype=np.float32)
        particle_rgba = np.asarray(rgba[self.geom_ids], dtype=np.float32)
        covariances = np.zeros((len(self.geom_ids), 3, 3), dtype=np.float32)
        covariances[:, 0, 0] = radii * radii
        covariances[:, 1, 1] = radii * radii
        covariances[:, 2, 2] = radii * radii

        self.handle.centers = centers
        self.handle.covariances = covariances
        self.handle.rgbs = np.clip(particle_rgba[:, :3], 0.0, 1.0)
        self.handle.opacities = np.clip(particle_rgba[:, 3:4], 0.0, 1.0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat


def _load_gaussian_splats(
    path: Path,
    max_points: int,
    splat_scale: float,
) -> GaussianSplats:
    vertex = PlyData.read(path, mmap=True)["vertex"].data
    names = set(vertex.dtype.names or ())
    if max_points > 0 and len(vertex) > max_points:
        indices = np.linspace(0, len(vertex) - 1, max_points, dtype=np.int64)
        vertex = vertex[indices]

    centers = np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=1).astype(
        np.float32
    )
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh_dc = np.stack([vertex[f"f_dc_{channel}"] for channel in range(3)], axis=1)
        rgbs = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0).astype(np.float32)
    else:
        rgbs = (
            np.stack(
                [vertex[color] for color in ("red", "green", "blue")], axis=1
            ).astype(np.float32)
            / 255.0
        )

    opacities = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))[:, None]
    scales = np.exp(
        np.stack([vertex[f"scale_{axis}"] for axis in range(3)], axis=1)
    ).astype(np.float32)
    scales *= splat_scale
    quats = np.stack([vertex[f"rot_{axis}"] for axis in range(4)], axis=1).astype(
        np.float32
    )
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quats.T
    rotations = np.empty((len(centers), 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    scale_covariances = np.zeros((len(centers), 3, 3), dtype=np.float32)
    scale_covariances[:, 0, 0] = scales[:, 0] ** 2
    scale_covariances[:, 1, 1] = scales[:, 1] ** 2
    scale_covariances[:, 2, 2] = scales[:, 2] ** 2
    covariances = rotations @ scale_covariances @ np.swapaxes(rotations, 1, 2)
    return GaussianSplats(centers, covariances, rgbs, opacities)


def _find_background_frame(task_id: str, ply_path: Path) -> str | None:
    env_cfg = load_env_cfg(task_id, play=True)
    for sensor in env_cfg.scene.sensors:
        sensor_ply = getattr(sensor, "background_ply_path", None)
        if sensor_ply is not None and Path(sensor_ply).resolve() == ply_path.resolve():
            return getattr(sensor, "background_frame_body_name", None)
    raise ValueError(f"Task {task_id!r} has no sensor for {ply_path}")


def _next_capture_index(task_dir: Path) -> int:
    if not task_dir.is_dir():
        return 1
    indices = [
        int(path.stem.removeprefix("viser_"))
        for path in task_dir.glob("viser_*.png")
        if path.stem.removeprefix("viser_").isdigit()
    ]
    return max(indices, default=0) + 1


def _camera_pose(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.percentile(centers, (2.0, 98.0), axis=0)
    look_at = (lower + upper) / 2.0
    extent = np.maximum(upper - lower, 0.1)
    direction = np.asarray((1.0, -1.0, 0.9), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return look_at, look_at + 1.35 * float(np.linalg.norm(extent)) * direction


def _install_gaussian_overlay(
    viewer,
    task_id: str,
    scene_cfg: GaussianSceneCfg,
    frame_body_name: str | None,
    splats: GaussianSplats,
    cfg: GaussianPlayConfig,
) -> None:
    model = viewer.env.unwrapped.sim.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if frame_body_name is None:
        frame_position = np.zeros(3, dtype=np.float64)
        frame_rotation = np.eye(3, dtype=np.float64)
    else:
        frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_body_name)
        if frame_id < 0:
            raise ValueError(f"Background frame body not found: {frame_body_name}")
        frame_position = np.asarray(data.xpos[frame_id], dtype=np.float64)
        frame_rotation = np.asarray(data.xmat[frame_id], dtype=np.float64).reshape(3, 3)

    rotation = frame_rotation @ _quat_wxyz_to_matrix(np.asarray(scene_cfg.quat))
    position = frame_position + frame_rotation @ np.asarray(scene_cfg.position)
    wxyz = _matrix_to_quat_wxyz(rotation)
    viewer._server.scene.add_gaussian_splats(
        "/gaussian/splats",
        centers=splats.centers,
        covariances=splats.covariances,
        rgbs=splats.rgbs,
        opacities=splats.opacities,
        scale=scene_cfg.scale,
        position=position,
        wxyz=wxyz,
    )

    table_body_ids = {
        body_id
        for body_id in range(model.nbody)
        if (model.body(body_id).name or "").startswith("table/")
    }
    for (
        body_id,
        _group_id,
        _sub_idx,
    ), handle in viewer._scene.mesh_handles_by_group.items():
        if body_id in table_body_ids:
            handle.visible = False
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model.body(body_id).name or f"body_{body_id}"
        viewer._server.scene.remove_by_name(
            f"/fixed_bodies/{body_name}/{model.geom(geom_id).name}"
        )

    display_centers = scene_cfg.scale * (splats.centers @ rotation.T) + position
    camera_look_at, base_camera_position = _camera_pose(display_centers)

    def set_camera(client=None) -> None:
        camera_position = base_camera_position + camera_offset_control.value
        target = viewer._server.initial_camera if client is None else client.camera
        target.look_at = camera_look_at
        target.position = camera_position
        target.up = np.asarray((0.0, 0.0, 1.0))

    with viewer._server.gui.add_folder("Gaussian Overlay"):
        viewer._server.gui.add_markdown(
            f"**Task:** `{task_id}`  \n**PLY:** `{scene_cfg.ply_path.name}`"
        )
        camera_offset_control = viewer._server.gui.add_vector3(
            "Camera position offset (world XYZ)",
            initial_value=cfg.camera_offset,
            step=0.01,
        )
        reset_camera = viewer._server.gui.add_button("Reset Gaussian camera")

    with viewer._server.gui.add_folder("Capture"):
        save_button = viewer._server.gui.add_button("Save Viser PNG")
        save_status = viewer._server.gui.add_text(
            "status", initial_value="Ready", disabled=True
        )

    @camera_offset_control.on_update
    def _(_) -> None:
        for client in viewer._server.get_clients().values():
            set_camera(client)

    @reset_camera.on_click
    def _(event) -> None:
        if event.client is not None:
            set_camera(event.client)

    @save_button.on_click
    def _(event) -> None:
        if event.client is None:
            save_status.value = "No Viser client is associated with this click"
            return
        task_dir = SCREENSHOT_ROOT / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / f"viser_{_next_capture_index(task_dir)}.png"
        image = event.client.get_render(
            width=cfg.screenshot_width,
            height=cfg.screenshot_height,
            transport_format="png",
        )
        Image.fromarray(image).save(output_path)
        save_status.value = f"Saved {output_path.relative_to(REPO_ROOT)}"
        print(save_status.value)

    set_camera()
    print(
        f"Gaussian Viser overlay: {scene_cfg.ply_path.name}, "
        f"{len(splats.centers):,} splats, frame={frame_body_name}"
    )
