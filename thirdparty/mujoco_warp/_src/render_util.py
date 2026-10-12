# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src.types import GaussianSpriteConfig
from mujoco_warp._src.types import ProjectionType
from mujoco_warp._src.types import RenderContext

wp.set_module_options({"enable_backward": False})


def configure_gaussian_sprites(
  rc: RenderContext,
  geom_ids: list[int] | tuple[int, ...],
  config: GaussianSpriteConfig,
) -> None:
  """Configure selected sphere geometries as camera-facing Gaussian sprites.

  This must be called before capturing a render CUDA graph.
  """
  if config.particle_sigma <= 0.0:
    raise ValueError("particle_sigma must be positive.")
  if config.particle_alpha < 0.0:
    raise ValueError("particle_alpha must be non-negative.")
  if config.particle_scale <= 0.0:
    raise ValueError("particle_scale must be positive.")
  if config.particle_alpha_cutoff < 0.0:
    raise ValueError("particle_alpha_cutoff must be non-negative.")
  if config.particle_alpha_cutoff > 1.0:
    raise ValueError("particle_alpha_cutoff must not exceed one.")
  if config.particle_velocity_scale < 0.0:
    raise ValueError("particle_velocity_scale must be non-negative.")
  if config.particle_max_anisotropy < 1.0:
    raise ValueError("particle_max_anisotropy must be at least one.")

  mask = np.zeros(rc.gaussian_sprite_mask.shape[0], dtype=bool)
  if geom_ids:
    ids = np.asarray(geom_ids, dtype=np.int32)
    if np.any(ids < 0) or np.any(ids >= mask.shape[0]):
      raise ValueError("Gaussian sprite geometry ID is out of range.")
    mask[ids] = True

  rc.gaussian_sprite_mask = wp.array(
    mask,
    dtype=bool,
    device=rc.gaussian_sprite_mask.device,
  )
  rc.particle_sigma = config.particle_sigma
  rc.particle_alpha = config.particle_alpha
  rc.particle_scale = config.particle_scale
  rc.particle_alpha_cutoff = config.particle_alpha_cutoff
  rc.particle_velocity_scale = config.particle_velocity_scale
  rc.particle_max_anisotropy = config.particle_max_anisotropy
  rc.update_gaussian_sprite_prev_xpos = bool(
    np.any(mask) and config.particle_velocity_scale > 0.0 and config.particle_max_anisotropy > 1.0
  )


def configure_render_exclusions(
  rc: RenderContext,
  geom_ids: list[int] | tuple[int, ...],
) -> None:
  """Exclude geometries from camera rays before transparency accumulation.

  Calls are additive so multiple camera sensors can contribute exclusions before
  the render graph is captured.
  """
  mask = rc.render_exclude_geom_mask.numpy()
  if geom_ids:
    ids = np.asarray(geom_ids, dtype=np.int32)
    if np.any(ids < 0) or np.any(ids >= mask.shape[0]):
      raise ValueError("Render exclusion geometry ID is out of range.")
    mask[ids] = True
  rc.render_exclude_geom_mask = wp.array(
    mask,
    dtype=bool,
    device=rc.render_exclude_geom_mask.device,
  )


@wp.kernel
def _convert_texture_data(
  # In:
  width: int,
  adr: int,
  nc: int,
  tex_data_in: wp.array(dtype=wp.uint8),
  # Out:
  tex_data_out: wp.array3d(dtype=float),
):
  """Convert uint8 texture data to vec4 format for efficient sampling."""
  x, y = wp.tid()
  offset = adr + (y * width + x) * nc
  r = tex_data_in[offset + 0] if nc > 0 else wp.uint8(0)
  g = tex_data_in[offset + 1] if nc > 1 else wp.uint8(0)
  b = tex_data_in[offset + 2] if nc > 2 else wp.uint8(0)
  a = wp.uint8(255)

  tex_data_out[y, x, 0] = float(r) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 1] = float(g) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 2] = float(b) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 3] = float(a) * wp.static(1.0 / 255.0)


def create_warp_texture(mjm: mujoco.MjModel, tex_id: int) -> wp.array:
  """Create a Warp texture from a MuJoCo model texture data."""
  tex_adr = mjm.tex_adr[tex_id]
  tex_width = mjm.tex_width[tex_id]
  tex_height = mjm.tex_height[tex_id]
  nchannel = mjm.tex_nchannel[tex_id]
  tex_data = wp.zeros((tex_height, tex_width, 4), dtype=float)

  wp.launch(
    _convert_texture_data,
    dim=(tex_width, tex_height),
    inputs=[tex_width, tex_adr, nchannel, wp.array(mjm.tex_data, dtype=wp.uint8)],
    outputs=[tex_data],
  )
  return wp.Texture2D(tex_data, filter_mode=wp.TextureFilterMode.LINEAR)


def update_warp_texture(mjm: mujoco.MjModel, rc: RenderContext, tex_id: int, image: np.ndarray) -> None:
  """Update an existing Warp texture in a render context.

  This is the runtime counterpart to ``create_warp_texture``.  The texture must
  already exist in ``rc.textures_registry`` and the incoming image must match
  the MuJoCo texture width and height.  The update copies RGBA float data into
  the existing Warp ``Texture2D`` object, so ``rc.textures`` does not need to be
  rebuilt for same-size updates.

  Args:
    mjm: MuJoCo model that owns the texture metadata.
    rc: Render context created from ``mjm``.
    tex_id: MuJoCo texture id to update.
    image: ``H x W x C`` NumPy image.  ``C`` may be 1, 3, or 4.  Integer images
      are normalized from ``[0, 255]``; floating images are assumed to be
      normalized and are clipped to ``[0, 1]``.
  """
  if not hasattr(wp, "Texture2D"):
    raise RuntimeError("Runtime texture updates require warp.Texture2D support.")
  if tex_id < 0 or tex_id >= mjm.ntex:
    raise ValueError(f"Invalid texture id {tex_id}; model has {mjm.ntex} textures.")
  if tex_id >= len(rc.textures_registry):
    raise ValueError(f"Texture id {tex_id} is not present in this render context.")
  if not rc.use_textures:
    raise ValueError("Render context was created with use_textures=False.")

  tex_height = int(mjm.tex_height[tex_id])
  tex_width = int(mjm.tex_width[tex_id])
  image = np.asarray(image)

  if image.ndim == 2:
    image = image[:, :, None]
  if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
    raise ValueError(f"Expected image shape HxWx1, HxWx3, or HxWx4; got {image.shape}.")
  if image.shape[:2] != (tex_height, tex_width):
    raise ValueError(
      f"Image size {image.shape[1]}x{image.shape[0]} does not match texture "
      f"{tex_id} size {tex_width}x{tex_height}."
    )

  if np.issubdtype(image.dtype, np.integer):
    rgba = image.astype(np.float32) * (1.0 / np.iinfo(image.dtype).max)
  else:
    rgba = image.astype(np.float32)
  rgba = np.clip(rgba, 0.0, 1.0)

  if rgba.shape[2] == 1:
    rgba = np.repeat(rgba, 3, axis=2)
  if rgba.shape[2] == 3:
    alpha = np.ones((tex_height, tex_width, 1), dtype=np.float32)
    rgba = np.concatenate((rgba, alpha), axis=2)

  rgba = np.ascontiguousarray(rgba)
  texture_data = wp.array(rgba, dtype=wp.vec4, device=rc.textures.device)
  rc.textures_registry[tex_id].copy_from_array(texture_data)


def _as_numpy_array(value) -> np.ndarray:
  if hasattr(value, "detach"):
    return value.detach().cpu().numpy()
  if hasattr(value, "numpy"):
    return value.numpy()
  return np.asarray(value)


def _resolve_batched_texture_ids(
  mat_texid,
  mat_id: int | None,
  tex_role: int,
) -> np.ndarray:
  mat_texid_np = _as_numpy_array(mat_texid)

  if mat_texid_np.ndim == 1:
    texture_ids = mat_texid_np
  elif mat_texid_np.ndim == 2:
    texture_ids = mat_texid_np[:, tex_role]
  elif mat_texid_np.ndim == 3:
    if mat_id is None:
      raise ValueError(
        "mat_id is required when mat_texid has shape [nworld, nmat, nrole]."
      )
    texture_ids = mat_texid_np[:, mat_id, tex_role]
  else:
    raise ValueError(
      "Expected mat_texid to be a 1D texture-id array, a 2D [nworld, nrole] "
      f"array, or a 3D [nworld, nmat, nrole] array; got shape {mat_texid_np.shape}."
    )

  return np.asarray(texture_ids, dtype=np.int32).reshape(-1)


def _normalize_batched_images(images) -> list[np.ndarray]:
  if isinstance(images, np.ndarray):
    if images.ndim < 3:
      raise ValueError(
        f"Expected batched images with shape [N, H, W, C], got {images.shape}."
      )
    return [images[i] for i in range(images.shape[0])]
  return [np.asarray(image) for image in images]


def update_batched_warp_texture(
  mjm: mujoco.MjModel,
  rc: RenderContext,
  mat_texid,
  images,
  *,
  mat_id: int | None = None,
  tex_role: int = 1,
) -> None:
  """Update per-world Warp textures selected by ``mat_texid``.

  This is a convenience wrapper around ``update_warp_texture`` for batched
  rendering.  It resolves one texture id per world from ``mat_texid`` and
  updates each referenced texture with the image at the same batch index.

  ``mat_texid`` may be:
    - ``[nworld]``: direct texture ids.
    - ``[nworld, nrole]``: texture ids for one material across worlds.
    - ``[nworld, nmat, nrole]``: full MJWarp ``model.mat_texid``; pass
      ``mat_id`` to select the material.

  The number of images must equal ``mat_texid.shape[0]`` after resolving the
  world dimension.  If multiple worlds point at the same texture id, those
  worlds cannot display different images because a Warp ``Texture2D`` is shared
  by texture id.  In that case this function accepts duplicate texture ids only
  when the corresponding images are identical.

  Args:
    mjm: MuJoCo model that owns the texture metadata.
    rc: Render context created from ``mjm``.
    mat_texid: Batched texture id source.
    images: Sequence or array of images with length ``nworld``. Each image uses
      the same format accepted by ``update_warp_texture``.
    mat_id: Material id to select when ``mat_texid`` is 3D.
    tex_role: MuJoCo texture role slot to select. MJWarp currently samples role
      1 for material textures.
  """
  texture_ids = _resolve_batched_texture_ids(mat_texid, mat_id, tex_role)
  batched_images = _normalize_batched_images(images)

  if len(batched_images) != len(texture_ids):
    raise ValueError(
      f"Expected {len(texture_ids)} images to match mat_texid batch dimension, "
      f"got {len(batched_images)}."
    )

  first_image_by_tex_id: dict[int, np.ndarray] = {}
  first_index_by_tex_id: dict[int, int] = {}
  for index, (tex_id_raw, image) in enumerate(
    zip(texture_ids, batched_images, strict=True)
  ):
    tex_id = int(tex_id_raw)
    if tex_id < 0:
      raise ValueError(
        f"mat_texid resolved to invalid texture id {tex_id} at batch index {index}."
      )

    first_image = first_image_by_tex_id.get(tex_id)
    if first_image is None:
      first_image_by_tex_id[tex_id] = np.asarray(image)
      first_index_by_tex_id[tex_id] = index
      continue

    if not np.array_equal(first_image, np.asarray(image)):
      first_index = first_index_by_tex_id[tex_id]
      raise ValueError(
        f"Texture id {tex_id} is shared by batch indices {first_index} and {index}, "
        "but their images differ. Give each world a unique texture id before using "
        "different per-world images."
      )

  for tex_id, image in first_image_by_tex_id.items():
    update_warp_texture(mjm, rc, tex_id, image)


@wp.func
def compute_ray(
  # In:
  projection: int,
  fovy: float,
  sensorsize: wp.vec2,
  intrinsic: wp.vec4,
  img_w: int,
  img_h: int,
  px: int,
  py: int,
  znear: float,
) -> wp.vec3:
  """Compute ray direction for a pixel with per-world camera parameters.

  This combines _camera_frustum_bounds and build_primary_rays logic for use
  inside a kernel when camera parameters are batched/randomized across worlds.
  """
  if projection == ProjectionType.ORTHOGRAPHIC:
    return wp.vec3(0.0, 0.0, -1.0)

  aspect = float(img_w) / float(img_h)
  sensor_h = sensorsize[1]

  # Check if we have intrinsics (sensorsize[1] != 0)
  if sensor_h != 0.0:
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
    sensor_w = sensorsize[0]

    target_aspect = float(img_w) / float(img_h)
    sensor_aspect = sensor_w / sensor_h
    if target_aspect > sensor_aspect:
      sensor_h = sensor_w / target_aspect
    elif target_aspect < sensor_aspect:
      sensor_w = sensor_h * target_aspect

    inv_fx_znear = znear / fx
    inv_fy_znear = znear / fy
    left = -inv_fx_znear * cx
    right = inv_fx_znear * (sensor_w - cx)
    top = inv_fy_znear * cy
    bottom = -inv_fy_znear * (sensor_h - cy)
  else:
    fovy_rad = fovy * wp.static(wp.pi / 180.0)
    half_height = znear * wp.tan(0.5 * fovy_rad)
    half_width = half_height * aspect
    left = -half_width
    right = half_width
    top = half_height
    bottom = -half_height

  u = (float(px) + 0.5) / float(img_w)
  v = (float(py) + 0.5) / float(img_h)
  x = left + (right - left) * u
  y = top + (bottom - top) * v

  return wp.normalize(wp.vec3(x, y, -znear))


@wp.func
def pack_rgba_to_uint32(r: float, g: float, b: float, a: float) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return wp.uint32((int(a) << int(24)) | (int(r) << int(16)) | (int(g) << int(8)) | int(b))


@wp.kernel
def unpack_rgb_kernel(
  # In:
  packed: wp.array2d(dtype=wp.uint32),
  rgb_adr: wp.array(dtype=int),
  camera_index: int,
  # Out:
  rgb_out: wp.array3d(dtype=wp.vec3),
):
  """Unpack ABGR uint32 packed pixel data into separate R, G, and B channels."""
  worldid, pixelid = wp.tid()

  xid = pixelid % rgb_out.shape[2]
  yid = pixelid // rgb_out.shape[2]

  rgb_adr_offset = rgb_adr[camera_index]
  val = packed[worldid, rgb_adr_offset + pixelid]
  b = wp.float32(val & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
  g = wp.float32((val >> wp.uint32(8)) & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
  r = wp.float32((val >> wp.uint32(16)) & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
  rgb_out[worldid, yid, xid] = wp.vec3(r, g, b)


@wp.kernel
def extract_depth_kernel(
  # In:
  depth_data: wp.array2d(dtype=float),
  depth_adr: wp.array(dtype=int),
  camera_index: int,
  depth_scale: float,
  # Out:
  depth_out: wp.array3d(dtype=float),
):
  """Extract the depth data from the render context buffers for a given camera index."""
  worldid, pixelid = wp.tid()
  xid = pixelid % depth_out.shape[2]
  yid = pixelid // depth_out.shape[2]

  depth_adr_offset = depth_adr[camera_index]
  val = depth_data[worldid, depth_adr_offset + pixelid]
  depth_out[worldid, yid, xid] = wp.clamp(val / depth_scale, 0.0, 1.0)


def get_rgb(rc: RenderContext, camera_index: int, rgb_out: wp.array3d(dtype=wp.vec3)):
  """Get the RGB data output from the render context buffers for a given camera index.

  Args:
    rc: The render context on device.
    camera_index: The index of the camera to get the RGB data for.
    rgb_out: The output array to store the RGB data in, with shape (nworld, height, width).
  """
  wp.launch(
    unpack_rgb_kernel,
    dim=(rgb_out.shape[0], rgb_out.shape[1] * rgb_out.shape[2]),
    inputs=[rc.rgb_data, rc.rgb_adr, camera_index],
    outputs=[rgb_out],
  )


def get_depth(rc: RenderContext, camera_index: int, depth_scale: float, depth_out: wp.array3d(dtype=float)):
  """Get the depth data output from the render context buffers for a given camera index.

  Args:
    rc: The render context on device.
    camera_index: The index of the camera to get the depth data for.
    depth_scale: The scale factor to apply to the depth data.
    depth_out: The output array to store the scaled and clamped depth data in
      with shape (nworld, height, width).
  """
  wp.launch(
    extract_depth_kernel,
    dim=(depth_out.shape[0], depth_out.shape[1] * depth_out.shape[2]),
    inputs=[rc.depth_data, rc.depth_adr, camera_index, depth_scale],
    outputs=[depth_out],
  )
