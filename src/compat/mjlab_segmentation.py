"""Compatibility patch for MuGS hybrid cameras on older mjlab.

MuGS hybrid rendering asks mjlab's camera SensorContext for segmentation masks.
The mujoco_warp version used by this repo already supports segmentation buffers,
but mjlab 1.2's camera wrapper only exposes RGB/depth. This patch also exposes
the straight foreground RGB and alpha required to composite translucent MuJoCo
objects over Gaussian backgrounds without a backing-color halo.
"""

from __future__ import annotations

import inspect
import warnings

import mujoco_warp as mjwarp
import torch
import warp as wp


@wp.kernel
def _unpack_foreground_rgb_kernel(
    packed: wp.array2d(dtype=wp.uint32),
    rgb: wp.array3d(dtype=wp.uint8),
):
    world_idx, pixel_idx = wp.tid()
    val = packed[world_idx, pixel_idx]
    rgb[world_idx, pixel_idx, 0] = wp.uint8(
        (val >> wp.uint32(16)) & wp.uint32(0xFF)
    )
    rgb[world_idx, pixel_idx, 1] = wp.uint8(
        (val >> wp.uint32(8)) & wp.uint32(0xFF)
    )
    rgb[world_idx, pixel_idx, 2] = wp.uint8(val & wp.uint32(0xFF))


def apply_patch() -> None:
    """Enable segmentation and foreground layers for mjlab camera sensors."""
    from mjlab.sensor import camera_sensor
    from mjlab.sensor import sensor_context

    render_context_params = inspect.signature(mjwarp.create_render_context).parameters
    if not {"render_seg", "render_foreground"}.issubset(render_context_params):
        return

    _patch_camera_cfg_validation(camera_sensor.CameraSensorCfg)
    _patch_sensor_context(sensor_context.SensorContext)


def _patch_camera_cfg_validation(camera_sensor_cfg_cls: type) -> None:
    current = camera_sensor_cfg_cls.__post_init__
    if getattr(current, "_accepts_segmentation", False):
        return

    def __post_init__(self) -> None:
        valid = {"rgb", "depth", "segmentation"}
        invalid = {dt for dt in self.data_types if dt not in valid}
        if invalid:
            raise ValueError(f"Invalid camera data types: {invalid}. Valid types: {valid}")
        if not self.data_types:
            raise ValueError("At least one data type must be specified.")

    __post_init__._accepts_segmentation = True  # type: ignore[attr-defined]
    camera_sensor_cfg_cls.__post_init__ = __post_init__


def _patch_sensor_context(sensor_context_cls: type) -> None:
    if not hasattr(sensor_context_cls, "get_segmentation"):
        sensor_context_cls.get_segmentation = _get_segmentation
    if not hasattr(sensor_context_cls, "get_foreground_rgb"):
        sensor_context_cls.get_foreground_rgb = _get_foreground_rgb
    if not hasattr(sensor_context_cls, "get_foreground_alpha"):
        sensor_context_cls.get_foreground_alpha = _get_foreground_alpha

    unpack_rgb = sensor_context_cls.unpack_rgb
    if not getattr(unpack_rgb, "_unpacks_foreground", False):
        def unpack_rgb_with_foreground(self) -> None:
            unpack_rgb(self)
            foreground_unpacked = getattr(self, "_foreground_rgb_unpacked", None)
            if foreground_unpacked is not None:
                wp.launch(
                    _unpack_foreground_rgb_kernel,
                    dim=(self._data.nworld, self._ctx.foreground_rgb_data.shape[1]),
                    inputs=[self._ctx.foreground_rgb_data],
                    outputs=[foreground_unpacked],
                    device=self.wp_device,
                )

        unpack_rgb_with_foreground._unpacks_foreground = True  # type: ignore[attr-defined]
        sensor_context_cls.unpack_rgb = unpack_rgb_with_foreground

    current = sensor_context_cls._create_context
    if getattr(current, "_renders_segmentation", False):
        return

    _create_context._renders_segmentation = True  # type: ignore[attr-defined]
    sensor_context_cls._create_context = _create_context


def _get_segmentation(self, cam_idx: int) -> torch.Tensor:
    """Get segmentation object ID/type pairs for a camera."""
    if cam_idx not in self._cam_idx_to_list_idx:
        available = list(self._cam_idx_to_list_idx.keys())
        raise KeyError(
            f"Camera ID {cam_idx} not found in SensorContext. "
            f"Available camera IDs: {available}"
        )

    list_idx = self._cam_idx_to_list_idx[cam_idx]

    seg_adr_np = getattr(self, "_seg_adr_np", None)
    seg_torch = getattr(self, "_seg_torch", None)
    if seg_adr_np is None or seg_torch is None:
        raise RuntimeError(
            "Segmentation rendering is not enabled. Ensure at least one camera "
            "sensor has 'segmentation' in its data_types."
        )

    seg_adr = seg_adr_np[list_idx]
    if seg_adr < 0:
        raise RuntimeError(
            f"Camera ID {cam_idx} does not have segmentation rendering enabled."
        )

    sensor = self.camera_sensors[list_idx]
    w, h = sensor.cfg.width, sensor.cfg.height
    num_pixels = w * h
    nworld = self._data.nworld

    cam_data = seg_torch[:, seg_adr : seg_adr + num_pixels, :]
    return cam_data.view(nworld, h, w, 2)


def _get_foreground_rgb(self, cam_idx: int) -> torch.Tensor:
    """Get straight foreground RGB for a camera."""
    list_idx, rgb_adr, sensor = _resolve_rgb_camera(self, cam_idx)
    del list_idx
    foreground_rgb = getattr(self, "_foreground_rgb_torch", None)
    if foreground_rgb is None:
        raise RuntimeError("Foreground RGB rendering is not enabled.")

    w, h = sensor.cfg.width, sensor.cfg.height
    num_pixels = w * h
    cam_data = foreground_rgb[:, rgb_adr : rgb_adr + num_pixels, :]
    return cam_data.view(self._data.nworld, h, w, 3)


def _get_foreground_alpha(self, cam_idx: int) -> torch.Tensor:
    """Get foreground coverage for a camera."""
    list_idx, rgb_adr, sensor = _resolve_rgb_camera(self, cam_idx)
    del list_idx
    foreground_alpha = getattr(self, "_foreground_alpha_torch", None)
    if foreground_alpha is None:
        raise RuntimeError("Foreground alpha rendering is not enabled.")

    w, h = sensor.cfg.width, sensor.cfg.height
    num_pixels = w * h
    cam_data = foreground_alpha[:, rgb_adr : rgb_adr + num_pixels]
    return cam_data.view(self._data.nworld, h, w, 1)


def _resolve_rgb_camera(self, cam_idx: int):
    if cam_idx not in self._cam_idx_to_list_idx:
        available = list(self._cam_idx_to_list_idx.keys())
        raise KeyError(
            f"Camera ID {cam_idx} not found in SensorContext. "
            f"Available camera IDs: {available}"
        )
    list_idx = self._cam_idx_to_list_idx[cam_idx]
    rgb_adr_np = getattr(self, "_rgb_adr_np", None)
    if rgb_adr_np is None or rgb_adr_np[list_idx] < 0:
        raise RuntimeError(f"Camera ID {cam_idx} does not have RGB rendering enabled.")
    return list_idx, rgb_adr_np[list_idx], self.camera_sensors[list_idx]


def _create_context(self, mj_model) -> None:
    """Create a render context with optional segmentation/foreground output."""
    ncam = mj_model.ncam
    cam_res: list[tuple[int, int]] | None = None
    render_rgb: list[bool] | None = None
    render_depth: list[bool] | None = None
    render_seg: list[bool] | None = None
    render_foreground = any(
        getattr(sensor.cfg, "render_mode", None) == "hybrid"
        for sensor in self.camera_sensors
    )

    raycast_geom_groups = self._raycast_geom_groups()

    if self.camera_sensors:
        ref_cfg = self.camera_sensors[0].cfg
        use_textures = ref_cfg.use_textures
        use_shadows = ref_cfg.use_shadows
        cam_geom_groups = set(ref_cfg.enabled_geom_groups)

        if self.raycast_sensors and raycast_geom_groups != cam_geom_groups:
            merged = sorted(cam_geom_groups | raycast_geom_groups)
            warnings.warn(
                "Camera enabled_geom_groups "
                f"{sorted(cam_geom_groups)} and raycast "
                f"include_geom_groups {sorted(raycast_geom_groups)}"
                f" differ. Using union {merged}.",
                stacklevel=2,
            )
            enabled_geom_groups = merged
        else:
            enabled_geom_groups = sorted(cam_geom_groups)

        cam_active = [False] * ncam
        cam_res = []
        render_rgb = []
        render_depth = []
        render_seg = []

        for sensor in self.camera_sensors:
            cam_active[sensor.camera_idx] = True
            cam_res.append((sensor.cfg.width, sensor.cfg.height))
            render_rgb.append("rgb" in sensor.cfg.data_types)
            render_depth.append("depth" in sensor.cfg.data_types)
            render_seg.append("segmentation" in sensor.cfg.data_types)
    else:
        use_textures = False
        use_shadows = False
        enabled_geom_groups = sorted(raycast_geom_groups)
        cam_active = [False] * ncam

    with wp.ScopedDevice(self.wp_device):
        self._ctx = mjwarp.create_render_context(
            mjm=mj_model,
            nworld=self._data.nworld,
            cam_res=cam_res,
            render_rgb=render_rgb,
            render_depth=render_depth,
            render_seg=render_seg,
            render_foreground=render_foreground,
            use_textures=use_textures,
            use_shadows=use_shadows,
            enabled_geom_groups=enabled_geom_groups,
            cam_active=cam_active,
            use_precomputed_rays=not self._disable_precomputed_rays,
        )

    excluded_geom_ids = {
        geom_id
        for sensor in self.camera_sensors
        for geom_id in getattr(sensor, "_foreground_exclude_geom_ids", ())
    }
    mjwarp.configure_render_exclusions(
        self._ctx,
        tuple(sorted(excluded_geom_ids)),
    )

    if self.camera_sensors:
        self._rgb_adr_np = self._ctx.rgb_adr.numpy().tolist()
        self._depth_adr_np = self._ctx.depth_adr.numpy().tolist()
        self._seg_adr_np = self._ctx.seg_adr.numpy().tolist()

    has_any_rgb = self._rgb_adr_np is not None and any(a >= 0 for a in self._rgb_adr_np)
    has_any_depth = self._depth_adr_np is not None and any(
        a >= 0 for a in self._depth_adr_np
    )
    has_any_seg = getattr(self, "_seg_adr_np", None) is not None and any(
        a >= 0 for a in self._seg_adr_np
    )

    if has_any_rgb:
        total_rgb_pixels = self._ctx.rgb_data.shape[1]
        nworld = self._data.nworld
        self._rgb_unpacked = wp.zeros(
            (nworld, total_rgb_pixels, 3),
            dtype=wp.uint8,
            device=self.wp_device,
        )
        self._rgb_torch = wp.to_torch(self._rgb_unpacked)
    else:
        self._rgb_unpacked = None
        self._rgb_torch = None

    if has_any_rgb and render_foreground:
        self._foreground_rgb_unpacked = wp.zeros(
            (nworld, total_rgb_pixels, 3),
            dtype=wp.uint8,
            device=self.wp_device,
        )
        self._foreground_rgb_torch = wp.to_torch(self._foreground_rgb_unpacked)
        self._foreground_alpha_torch = wp.to_torch(self._ctx.foreground_alpha_data)
    else:
        self._foreground_rgb_unpacked = None
        self._foreground_rgb_torch = None
        self._foreground_alpha_torch = None

    if has_any_depth:
        self._depth_torch = wp.to_torch(self._ctx.depth_data)
    else:
        self._depth_torch = None

    if has_any_seg:
        self._seg_torch = wp.to_torch(self._ctx.seg_data)
    else:
        self._seg_torch = None
