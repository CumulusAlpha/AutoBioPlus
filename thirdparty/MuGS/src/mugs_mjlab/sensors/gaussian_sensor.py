"""
GaussianSensorMjlab: Batch-first photorealistic rendering for mjlab

Implements mjlab.sensor.Sensor interface with:
- Batch rendering: (num_envs, H, W, 3) torch tensors
- 3DGS background + MuJoCo foreground hybrid rendering
- SensorContext integration for efficient GPU rendering
- Two-phase initialization: edit_spec() → initialize()

This module requires mjlab to be installed.

Author: MuGS Team
Date: 2026-05-02
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from plyfile import PlyData

try:
    from gsplat import rasterization

    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    raise ImportError("gsplat is required for mugs_mjlab. Install with: pip install gsplat")

try:
    import mujoco_warp as mjwarp
    import warp as wp
    from mjlab.entity import Entity
    from mjlab.sensor.camera_sensor import CameraSensor, CameraSensorCfg

    MJLAB_AVAILABLE = True
except ImportError as e:
    MJLAB_AVAILABLE = False
    raise ImportError(
        "mjlab is required for mugs_mjlab. " "Install from: https://github.com/YOUR_ORG/mjlab"
    ) from e

if TYPE_CHECKING:
    pass


RenderMode = Literal["hybrid", "3dgs_only", "mujoco_only"]


@dataclass
class GaussianSensorMjlabCfg(CameraSensorCfg):
    """Configuration for batch-first GaussianSensor.

    Inherits CameraSensorCfg so mjlab wires it into the SensorContext and
    mujoco_warp renders RGB+segmentation for all envs in one batched call.
    """

    intrinsics: tuple[float, float, float, float] | None = None
    """Pixel intrinsics ``(fx, fy, cx, cy)`` for calibrated rendering."""

    # 3DGS background
    background_ply_path: Path | None = None
    """Path to pretrained 3DGS PLY file."""

    background_frame_body_name: str | None = None
    """Body whose frame defines the canonical 3DGS scene coordinates.

    Multi-environment MJLab scenes translate each environment away from the
    origin. Set this to a fixed scene body that receives the same environment
    transform so Gaussian camera poses can be converted back to the canonical
    frame before rasterization.
    """

    background_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """3DGS scene translation in the background frame, in meters."""

    background_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """3DGS scene orientation in the background frame, as a wxyz quaternion."""

    background_scale: float = 1.0
    """Uniform scale applied to Gaussian positions and covariance scales."""

    # Rendering mode
    render_mode: RenderMode = "hybrid"
    """Rendering mode: hybrid (3DGS+MuJoCo), 3dgs_only, or mujoco_only."""

    foreground_mode: Literal["named", "all"] = "named"
    """Hybrid foreground mask mode: named geoms only, or all MuJoCo geoms."""

    foreground_exclude_geom_names: tuple[str, ...] = ("terrain",)
    """Geom names to keep out of the MuJoCo foreground mask in hybrid mode."""

    foreground_exclude_body_prefixes: tuple[str, ...] = ()
    """Body namespace prefixes whose descendant geoms stay out of the mask."""

    # Robot masking (for hybrid mode)
    robot_geom_names: list[str] = field(
        default_factory=lambda: [
            "base_link",
            "shoulder_link",
            "arm_link",
            "forearm_link",
            "palm",
            "left_finger_link",
            "right_finger_link",
        ]
    )
    """MuJoCo geom names to mask as foreground."""

    # Performance
    cache_background: bool = True
    """Cache static 3DGS background (faster for static scenes)."""

    # Output options
    return_components: bool = False
    """If True, include background/foreground/mask in data."""

    def __post_init__(self) -> None:
        if len(self.background_position) != 3:
            raise ValueError("background_position must contain 3 values")
        if len(self.background_quat) != 4:
            raise ValueError("background_quat must contain 4 values")
        if np.linalg.norm(self.background_quat) < 1e-8:
            raise ValueError("background_quat must be non-zero")
        if self.background_scale <= 0.0:
            raise ValueError("background_scale must be positive")

        # Hybrid mode needs warp to render rgb+segmentation; pure modes need rgb.
        required: tuple[str, ...]
        if self.render_mode == "hybrid":
            required = ("rgb", "segmentation")
        else:
            required = ("rgb",)
        merged = tuple(dict.fromkeys((*self.data_types, *required)))
        # Avoid mutating frozen-but-not-frozen dataclass via direct assignment.
        object.__setattr__(self, "data_types", merged)
        super().__post_init__()

    def build(self) -> GaussianSensorMjlab:
        return GaussianSensorMjlab(self)


@dataclass
class GaussianSensorData:
    """Batch sensor output data.

    All tensors have shape (num_envs, height, width, channels).
    Follows mjlab CameraSensorData pattern.
    """

    rgb: torch.Tensor
    """RGB images [num_envs, H, W, 3] (uint8)."""

    background: torch.Tensor | None = None
    """3DGS background [num_envs, H, W, 3] (uint8). None if not enabled."""

    foreground: torch.Tensor | None = None
    """MuJoCo foreground [num_envs, H, W, 3] (uint8). None if not enabled."""

    mask: torch.Tensor | None = None
    """MuJoCo foreground opacity [num_envs, H, W, 1]. None if not enabled."""


class GaussianSensorMjlab(CameraSensor):
    """Batch-first GaussianSensor compatible with mjlab.

    Inherits CameraSensor so mjlab's scene wires it into the SensorContext
    and mujoco_warp renders rgb+segmentation for all envs in one batched
    warp kernel (captured into the sense CUDA graph). MuGS layers a 3DGS
    background underneath and alpha-composites the robot foreground on top.

    Data format: (num_envs, height, width, channels) torch.Tensor.
    """

    def __init__(self, cfg: GaussianSensorMjlabCfg):
        if not GSPLAT_AVAILABLE:
            raise ImportError("gsplat is required for GaussianSensor")

        super().__init__(cfg)
        # Re-bind self.cfg with the concrete subclass type for attribute access.
        self.cfg: GaussianSensorMjlabCfg = cfg

        # Batch state (initialized in initialize()).
        self._num_envs: int = 0
        self._device: str = "cpu"
        self._mjwarp_data: mjwarp.Data | None = None

        # 3DGS background.
        self._gaussians: dict[str, torch.Tensor] | None = None
        self._cached_background: torch.Tensor | None = None
        self._cached_camera_poses: torch.Tensor | None = None
        self._background_frame_body_idx: int | None = None

        # Robot mask cache (populated in initialize).
        self._robot_geom_ids: list[int] = []
        self._foreground_exclude_geom_ids: list[int] = []

    def edit_spec(
        self,
        scene_spec: mujoco.MjSpec,
        entities: dict[str, Entity],
    ) -> None:
        super().edit_spec(scene_spec, entities)
        if self.cfg.intrinsics is None:
            return

        fx, fy, cx, cy = self.cfg.intrinsics
        cam = scene_spec.camera(self._camera_name)
        cam.resolution = (self.cfg.width, self.cfg.height)
        cam.sensor_size = (self.cfg.width, self.cfg.height)
        cam.focal_pixel = (fx, fy)
        cam.principal_pixel = (cx, cy)

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        """Initialize sensor after model compilation.

        Calls CameraSensor.initialize() to cache the MuJoCo camera index,
        then loads the 3DGS background and caches robot geom IDs for masking.
        """
        super().initialize(mj_model, model, data, device)

        self._mjwarp_data = data
        self._device = device
        self._num_envs = data.nworld

        if self.cfg.background_frame_body_name is not None:
            body_idx = mujoco.mj_name2id(
                mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.cfg.background_frame_body_name,
            )
            if body_idx < 0:
                available = [mj_model.body(i).name for i in range(mj_model.nbody)]
                raise ValueError(
                    f"Background frame body {self.cfg.background_frame_body_name!r} "
                    f"was not found. Available bodies: {available}"
                )
            self._background_frame_body_idx = body_idx

        print(
            f"[GaussianSensorMjlab] Initializing for {self._num_envs} envs"
            f" — camera '{self._camera_name}' idx={self._camera_idx}"
        )

        if self.cfg.background_ply_path is not None:
            print(f"  → Loading 3DGS background: {self.cfg.background_ply_path}")
            self._gaussians = self._load_official_ply(
                Path(self.cfg.background_ply_path), torch.device(device)
            )
            self._apply_background_transform(self._gaussians)
            print(f"  ✓ Loaded {len(self._gaussians['means']):,} Gaussians")

        if self.cfg.render_mode in ("hybrid", "mujoco_only"):
            self._robot_geom_ids = []
            for geom_name in self.cfg.robot_geom_names:
                geom_id = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                )
                if geom_id >= 0:
                    self._robot_geom_ids.append(geom_id)
            print(
                f"  ✓ Robot geoms: "
                f"{len(self._robot_geom_ids)}/{len(self.cfg.robot_geom_names)}"
            )
            foreground_exclude_geom_ids: set[int] = set()
            for geom_name in self.cfg.foreground_exclude_geom_names:
                geom_id = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                )
                if geom_id >= 0:
                    foreground_exclude_geom_ids.add(geom_id)
            foreground_exclude_geom_ids.update(
                self._geom_ids_under_body_prefixes(
                    mj_model,
                    self.cfg.foreground_exclude_body_prefixes,
                )
            )
            self._foreground_exclude_geom_ids = sorted(
                foreground_exclude_geom_ids
            )
            print(
                f"  ✓ Foreground exclusions: "
                f"{len(self._foreground_exclude_geom_ids)} geom(s)"
            )

        print("[GaussianSensorMjlab] Initialization complete")

    def _compute_data(self) -> GaussianSensorData:
        """Compute batch sensor data for all environments.

        Returns:
            GaussianSensorData with (num_envs, H, W, C) tensors
        """
        if self.cfg.render_mode == "hybrid":
            return self._compute_hybrid()
        elif self.cfg.render_mode == "3dgs_only":
            return self._compute_3dgs_only()
        else:  # mujoco_only
            return self._compute_mujoco_only()

    # ─────────────────────────────────────────────────────────────
    # Rendering Modes
    # ─────────────────────────────────────────────────────────────

    def _compute_hybrid(self) -> GaussianSensorData:
        """Hybrid mode: 3DGS background + MuJoCo foreground."""

        # 1. Render 3DGS backgrounds (batch)
        backgrounds = self._render_3dgs_batch()  # (N, H, W, 3) uint8

        # 2. Render MuJoCo foregrounds (batch via SensorContext)
        foregrounds, masks = self._render_mujoco_batch()  # (N, H, W, 3), (N, H, W, 1)

        # 3. Composite
        rgb = self._composite_batch(backgrounds, foregrounds, masks)

        # 4. Return data
        if self.cfg.return_components:
            return GaussianSensorData(
                rgb=rgb,
                background=backgrounds,
                foreground=foregrounds,
                mask=masks,
            )
        else:
            return GaussianSensorData(rgb=rgb)

    def _compute_3dgs_only(self) -> GaussianSensorData:
        """3DGS only mode: Just render backgrounds."""
        backgrounds = self._render_3dgs_batch()
        return GaussianSensorData(
            rgb=backgrounds,
            background=backgrounds if self.cfg.return_components else None,
        )

    def _compute_mujoco_only(self) -> GaussianSensorData:
        """MuJoCo only mode: Just render foregrounds."""
        if self._ctx is None:
            raise RuntimeError(
                "SensorContext not available. "
                "GaussianSensorMjlab must be attached to a mjlab Scene."
            )

        foregrounds = self._ctx.get_rgb(self._camera_idx)

        if self.cfg.return_components:
            return GaussianSensorData(
                rgb=foregrounds,
                foreground=foregrounds,
            )
        else:
            return GaussianSensorData(rgb=foregrounds)

    # ─────────────────────────────────────────────────────────────
    # Batch Rendering Implementation
    # ─────────────────────────────────────────────────────────────

    def _render_3dgs_batch(self) -> torch.Tensor:
        """Render 3DGS backgrounds for all environments.

        Returns:
            RGB batch (num_envs, H, W, 3) uint8 tensor
        """
        if self._gaussians is None:
            # No background loaded - return black
            return torch.zeros(
                (self._num_envs, self.cfg.height, self.cfg.width, 3),
                dtype=torch.uint8,
                device=self._device,
            )

        # Get camera poses for all environments
        camera_poses = self._get_camera_poses_batch()  # (N, 4, 4)

        # Check cache and verify camera poses
        if self.cfg.cache_background and self._cached_background is not None:
            if self._cached_camera_poses is not None:
                # Verify poses haven't changed
                if torch.allclose(camera_poses, self._cached_camera_poses, atol=1e-6):
                    return self._cached_background
            # Poses changed or not cached - invalidate

        # Batched rendering with gsplat
        rgb_batch = self._render_3dgs_batch_optimized(camera_poses)

        # Cache if enabled
        if self.cfg.cache_background:
            self._cached_background = rgb_batch
            self._cached_camera_poses = camera_poses.clone()

        return rgb_batch

    def _render_3dgs_batch_optimized(self, camera_poses: torch.Tensor) -> torch.Tensor:
        """Optimized batch 3DGS rendering with single gsplat call.

        Args:
            camera_poses: (num_envs, 4, 4) camera-to-world transforms

        Returns:
            RGB batch (num_envs, H, W, 3) uint8 tensor
        """
        # Build batch view matrices (world → camera)
        view_matrices = torch.inverse(camera_poses)  # (N, 4, 4)

        if self.cfg.intrinsics is not None:
            fx, fy, cx, cy = self.cfg.intrinsics
        else:
            # CameraSensorCfg.fovy is in degrees; None means the MuJoCo
            # default of 45°.
            fov_y_deg = self.cfg.fovy if self.cfg.fovy is not None else 45.0
            fov_y_rad = np.deg2rad(fov_y_deg)
            fx = (self.cfg.height / 2) / np.tan(fov_y_rad / 2)
            fy = (self.cfg.height / 2) / np.tan(fov_y_rad / 2)
            cx = self.cfg.width / 2
            cy = self.cfg.height / 2

        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self._device
        )
        Ks = K.unsqueeze(0).expand(self._num_envs, 3, 3)  # (N, 3, 3)

        # Single batched rasterization call
        render_colors, _, _ = rasterization(
            means=self._gaussians["means"],
            quats=self._gaussians["quats"],
            scales=self._gaussians["scales"],
            opacities=self._gaussians["opacities"],
            colors=self._gaussians["colors"],
            viewmats=view_matrices,  # (N, 4, 4) batch
            Ks=Ks,  # (N, 3, 3) batch
            width=self.cfg.width,
            height=self.cfg.height,
            packed=True,
        )

        # Convert to uint8
        rgb_batch = render_colors.clamp(0, 1)
        return (rgb_batch * 255).to(torch.uint8)

    def _render_mujoco_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pull batched foreground RGB, alpha, and segmentation from mjwarp.

        SensorContext writes per-camera rgb/segmentation into shared GPU
        buffers via a single batched warp kernel captured in the sense
        CUDA graph; here we just take views into them.

        Returns:
            foregrounds: (num_envs, H, W, 3) uint8
            masks: (num_envs, H, W, 1) float32 foreground opacity
        """
        if self._ctx is None:
            raise RuntimeError(
                "SensorContext not available. "
                "GaussianSensorMjlab must be attached to a mjlab Scene."
            )

        rgb_batch = self._ctx.get_foreground_rgb(self._camera_idx)
        alpha_batch = self._ctx.get_foreground_alpha(self._camera_idx)
        seg_batch = self._ctx.get_segmentation(self._camera_idx)
        masks = self._create_robot_masks_batch(seg_batch) * alpha_batch
        return rgb_batch, masks

    def _composite_batch(
        self, backgrounds: torch.Tensor, foregrounds: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Composite foregrounds onto backgrounds using masks (batched).

        Args:
            backgrounds: (N, H, W, 3) uint8
            foregrounds: (N, H, W, 3) uint8
            masks: (N, H, W, 1) float32 foreground opacity [0, 1]

        Returns:
            Composited RGB (N, H, W, 3) uint8
        """
        # Expand mask to 3 channels
        masks_3ch = masks.expand(-1, -1, -1, 3)  # (N, H, W, 3)

        # Convert to float for blending
        bg_f = backgrounds.float()
        fg_f = foregrounds.float()

        # Alpha composite: result = bg * (1 - mask) + fg * mask
        composite_f = bg_f * (1 - masks_3ch) + fg_f * masks_3ch

        return composite_f.to(torch.uint8)

    # ─────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────

    def _get_camera_poses_batch(self) -> torch.Tensor:
        """Get camera-to-world transforms for all environments.

        mjwarp stores per-env cam_xpos as (nworld, ncam, 3) and cam_xmat as
        (nworld, ncam, 3, 3), as warp arrays — convert to torch (zero-copy)
        before slicing.

        Returns:
            Camera poses (num_envs, 4, 4) on device.
        """
        assert self._mjwarp_data is not None
        cam_pos_all = self._as_torch(self._mjwarp_data.cam_xpos)  # (N, ncam, 3)
        cam_mat_all = self._as_torch(self._mjwarp_data.cam_xmat)  # (N, ncam, 3, 3)
        cam_pos = cam_pos_all[:, self._camera_idx, :]  # (N, 3)
        rot_matrices = cam_mat_all[:, self._camera_idx, :, :]  # (N, 3, 3)

        if self._background_frame_body_idx is not None:
            body_pos_all = self._as_torch(self._mjwarp_data.xpos)
            body_mat_all = self._as_torch(self._mjwarp_data.xmat)
            frame_pos = body_pos_all[:, self._background_frame_body_idx, :]
            frame_rot = body_mat_all[:, self._background_frame_body_idx, :, :]
            world_to_frame_rot = frame_rot.transpose(-1, -2)
            cam_pos = torch.bmm(
                world_to_frame_rot,
                (cam_pos - frame_pos).unsqueeze(-1),
            ).squeeze(-1)
            rot_matrices = torch.bmm(world_to_frame_rot, rot_matrices)

        poses = (
            torch.eye(4, device=self._device)
            .unsqueeze(0)
            .expand(self._num_envs, 4, 4)
            .clone()
        )
        # MuJoCo camera frames look along -Z (OpenGL convention), while gsplat
        # view matrices use +Z-forward camera coordinates. Rotate the camera
        # frame by 180 degrees around X before converting camera-to-world to
        # world-to-camera in _render_3dgs_batch_optimized().
        opengl_to_opencv = torch.diag(
            torch.tensor([1.0, -1.0, -1.0], device=self._device)
        )
        poses[:, :3, :3] = rot_matrices @ opengl_to_opencv
        poses[:, :3, 3] = cam_pos
        return poses

    def _as_torch(self, value) -> torch.Tensor:
        """Convert warp arrays to torch, while accepting newer torch-backed mjwarp fields."""
        if isinstance(value, torch.Tensor):
            return value.to(self._device)
        try:
            return wp.to_torch(value)
        except AttributeError:
            device = getattr(value, "device", None)
            if isinstance(device, torch.device):
                return torch.as_tensor(value, device=device).to(self._device)
            return torch.as_tensor(value).to(self._device)

    @staticmethod
    def _geom_ids_under_body_prefixes(
        model: mujoco.MjModel,
        prefixes: tuple[str, ...],
    ) -> set[int]:
        if not prefixes:
            return set()

        normalized_prefixes = tuple(prefix.rstrip("/") for prefix in prefixes)
        excluded: set[int] = set()
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            while body_id > 0:
                body_name = model.body(body_id).name or ""
                if any(
                    body_name == prefix or body_name.startswith(f"{prefix}/")
                    for prefix in normalized_prefixes
                ):
                    excluded.add(geom_id)
                    break
                body_id = int(model.body_parentid[body_id])
        return excluded

    def _create_robot_masks_batch(self, seg_batch: torch.Tensor) -> torch.Tensor:
        """Create robot masks from segmentation IDs (batched).

        Args:
            seg_batch: (num_envs, H, W, 2) int32 object ID/type pairs

        Returns:
            masks: (num_envs, H, W, 1) float32 [0, 1]
        """
        geom_ids = seg_batch[..., 0:1]
        masks = torch.zeros_like(geom_ids, dtype=torch.float32)

        if self.cfg.foreground_mode == "all":
            masks[geom_ids >= 0] = 1.0
            for geom_id in self._foreground_exclude_geom_ids:
                masks[geom_ids == geom_id] = 0.0
            return masks

        # Mark robot geoms
        for geom_id in self._robot_geom_ids:
            masks[geom_ids == geom_id] = 1.0

        return masks

    def _load_official_ply(self, ply_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
        """Load official 3DGS PLY format.

        Returns:
            Dictionary of Gaussian parameters as torch tensors on device
        """
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]

        # Positions
        positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

        # SH DC → RGB
        SH_C0 = 0.28209479177387814
        sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
        colors = 0.5 + SH_C0 * sh_dc
        colors = np.clip(colors, 0, 1)

        # Opacity (sigmoid)
        opacities = 1 / (1 + np.exp(-vertex["opacity"]))

        # Scales (exp)
        scales = np.exp(np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1))

        # Quaternions (normalize)
        quats = np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1
        )
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        return {
            "means": torch.from_numpy(positions).float().to(device),
            "colors": torch.from_numpy(colors).float().to(device),
            "opacities": torch.from_numpy(opacities).float().to(device),
            "scales": torch.from_numpy(scales).float().to(device),
            "quats": torch.from_numpy(quats).float().to(device),
        }

    def _apply_background_transform(
        self,
        gaussians: dict[str, torch.Tensor],
    ) -> None:
        """Transform loaded Gaussians from scan coordinates into the background frame."""
        device = gaussians["means"].device
        dtype = gaussians["means"].dtype
        position = torch.tensor(
            self.cfg.background_position,
            device=device,
            dtype=dtype,
        )
        quat = torch.tensor(
            self.cfg.background_quat,
            device=device,
            dtype=dtype,
        )
        quat = quat / torch.linalg.vector_norm(quat)
        rotation = self._quat_wxyz_to_matrix(quat)
        scale = float(self.cfg.background_scale)

        gaussians["means"] = scale * (gaussians["means"] @ rotation.T) + position
        gaussians["scales"] = gaussians["scales"] * scale
        scene_quat = quat.unsqueeze(0).expand_as(gaussians["quats"])
        gaussians["quats"] = self._quat_multiply_wxyz(
            scene_quat,
            gaussians["quats"],
        )
        gaussians["quats"] = gaussians["quats"] / torch.linalg.vector_norm(
            gaussians["quats"],
            dim=-1,
            keepdim=True,
        )

    @staticmethod
    def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind()
        return torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            )
        ).reshape(3, 3)

    @staticmethod
    def _quat_multiply_wxyz(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        lw, lx, ly, lz = lhs.unbind(dim=-1)
        rw, rx, ry, rz = rhs.unbind(dim=-1)
        return torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )
