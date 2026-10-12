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
"""Tests for render functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data

try:
  mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"))
  _HAS_RENDERER = True
except Exception:
  _HAS_RENDERER = False


def _assert_eq(a, b, name):
  tol = 5e-4
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _render_gaussian_sprites(
  z_positions: tuple[float, ...],
  config: mjw.GaussianSpriteConfig,
  geom_alpha: float = 1.0,
  return_layers: bool = False,
  exclude: bool = False,
  backdrop: bool = False,
  exclude_backdrop: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
  geoms = "".join(
    f'<geom name="sprite_{i}" type="sphere" pos="0 0 {z}" size="0.5" '
    f'rgba="1 0 0 {geom_alpha}" contype="0" conaffinity="0"/>'
    for i, z in enumerate(z_positions)
  )
  backdrop_geom = (
    '<geom name="backdrop" type="plane" pos="0 0 -0.6" size="2 2 0.1" '
    'rgba="0 1 0 1" contype="0" conaffinity="0"/>'
    if backdrop
    else ""
  )
  mjm = mujoco.MjModel.from_xml_string(
    f"""
    <mujoco>
      <worldbody>
        <camera name="camera" pos="0 0 3"/>
        {geoms}
        {backdrop_geom}
      </worldbody>
    </mujoco>
    """
  )
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  rc = mjw.create_render_context(
    mjm,
    cam_res=(65, 65),
    render_rgb=True,
    render_depth=False,
    render_foreground=return_layers,
    use_textures=False,
  )
  rc.background_color = wp.uint32(0xFF000000)
  geom_ids = [
    mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, f"sprite_{i}")
    for i in range(len(z_positions))
  ]
  mjw.configure_gaussian_sprites(rc, geom_ids, config)
  if exclude:
    mjw.configure_render_exclusions(rc, geom_ids)
  if exclude_backdrop:
    backdrop_id = mujoco.mj_name2id(
      mjm, mujoco.mjtObj.mjOBJ_GEOM, "backdrop"
    )
    mjw.configure_render_exclusions(rc, [backdrop_id])
  mjw.refit_bvh(m, d, rc)
  mjw.render(m, d, rc)

  packed = rc.rgb_data.numpy()[0].reshape(65, 65)
  red = ((packed >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
  if not return_layers:
    return red

  foreground_packed = rc.foreground_rgb_data.numpy()[0].reshape(65, 65)
  foreground_red = (
    (foreground_packed >> np.uint32(16)) & np.uint32(0xFF)
  ).astype(np.uint8)
  foreground_alpha = rc.foreground_alpha_data.numpy()[0].reshape(65, 65)
  return red, foreground_red, foreground_alpha


def _render_moving_gaussian_sprite(delta: tuple[float, float, float]) -> np.ndarray:
  mjm = mujoco.MjModel.from_xml_string(
    """
    <mujoco>
      <worldbody>
        <camera name="camera" pos="0 0 3"/>
        <geom name="sprite" type="sphere" pos="0 0 0" size="0.5"
          rgba="1 0 0 1" contype="0" conaffinity="0"/>
      </worldbody>
    </mujoco>
    """
  )
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  rc = mjw.create_render_context(
    mjm,
    cam_res=(65, 65),
    render_rgb=True,
    render_depth=False,
    use_textures=False,
  )
  rc.background_color = wp.uint32(0xFF000000)
  geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "sprite")
  mjw.configure_gaussian_sprites(
    rc,
    [geom_id],
    mjw.GaussianSpriteConfig(
      particle_sigma=6.0,
      particle_scale=1.0,
      particle_alpha_cutoff=0.0,
      particle_velocity_scale=20.0,
      particle_max_anisotropy=3.0,
    ),
  )

  mjw.refit_bvh(m, d, rc)
  mjw.render(m, d, rc)

  geom_xpos = d.geom_xpos.numpy()
  geom_xpos[0, geom_id] += np.asarray(delta, dtype=np.float32)
  d.geom_xpos = wp.array(geom_xpos, dtype=wp.vec3)

  mjw.refit_bvh(m, d, rc)
  mjw.render(m, d, rc)

  packed = rc.rgb_data.numpy()[0].reshape(65, 65)
  return ((packed >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)


class RenderTest(parameterized.TestCase):
  def test_gaussian_sprite_has_soft_edge(self):
    red = _render_gaussian_sprites(
      (0.0,),
      mjw.GaussianSpriteConfig(
        particle_sigma=4.0,
        particle_alpha=1.0,
        particle_scale=1.0,
        particle_alpha_cutoff=0.0,
      ),
    )

    center = int(red[32, 32])
    shoulder = int(red[32, 38])
    edge = int(red[32, 44])
    outside = int(red[32, 47])
    self.assertGreater(center, shoulder)
    self.assertGreater(shoulder, edge)
    self.assertGreater(edge, outside)
    self.assertEqual(outside, 0)

  def test_gaussian_sprite_sigma_controls_softness(self):
    soft = _render_gaussian_sprites(
      (0.0,),
      mjw.GaussianSpriteConfig(
        particle_sigma=2.0,
        particle_scale=1.0,
        particle_alpha_cutoff=0.0,
      ),
    )
    sharp = _render_gaussian_sprites(
      (0.0,),
      mjw.GaussianSpriteConfig(
        particle_sigma=8.0,
        particle_scale=1.0,
        particle_alpha_cutoff=0.0,
      ),
    )

    self.assertGreater(int(soft[32, 40]), int(sharp[32, 40]))

  def test_gaussian_sprites_accumulate_alpha(self):
    config = mjw.GaussianSpriteConfig(
      particle_sigma=6.0,
      particle_alpha=1.0,
      particle_scale=1.0,
      particle_alpha_cutoff=0.0,
    )
    single = _render_gaussian_sprites((0.0,), config, geom_alpha=0.5)
    overlapping = _render_gaussian_sprites((0.0, -0.1), config, geom_alpha=0.5)

    self.assertGreater(int(overlapping[32, 32]), int(single[32, 32]))

  def test_gaussian_sprite_exports_straight_foreground_and_alpha(self):
    display_red, foreground_red, foreground_alpha = _render_gaussian_sprites(
      (0.0,),
      mjw.GaussianSpriteConfig(
        particle_sigma=4.0,
        particle_scale=1.0,
        particle_alpha_cutoff=0.0,
      ),
      geom_alpha=0.5,
      return_layers=True,
    )

    self.assertAlmostEqual(float(foreground_alpha[32, 32]), 0.5, places=5)
    self.assertEqual(int(foreground_red[32, 32]), 255)
    self.assertAlmostEqual(
      int(display_red[32, 32]),
      255.0 * float(foreground_alpha[32, 32]),
      delta=1.0,
    )

  def test_render_exclusion_removes_sprite_before_compositing(self):
    display_red, foreground_red, foreground_alpha = _render_gaussian_sprites(
      (0.0,),
      mjw.GaussianSpriteConfig(),
      return_layers=True,
      exclude=True,
    )

    self.assertEqual(int(display_red.max()), 0)
    self.assertEqual(int(foreground_red.max()), 0)
    self.assertEqual(float(foreground_alpha.max()), 0.0)

  def test_render_exclusion_prevents_backdrop_contaminating_transparency(self):
    _, backed_red, backed_alpha = _render_gaussian_sprites(
      (0.0,),
      mjw.GaussianSpriteConfig(particle_alpha_cutoff=0.0),
      geom_alpha=0.5,
      return_layers=True,
      backdrop=True,
    )
    _, isolated_red, isolated_alpha = _render_gaussian_sprites(
      (0.0,),
      mjw.GaussianSpriteConfig(particle_alpha_cutoff=0.0),
      geom_alpha=0.5,
      return_layers=True,
      backdrop=True,
      exclude_backdrop=True,
    )

    self.assertEqual(float(backed_alpha[32, 32]), 1.0)
    self.assertLess(int(backed_red[32, 32]), 255)
    self.assertAlmostEqual(float(isolated_alpha[32, 32]), 0.5, places=5)
    self.assertEqual(int(isolated_red[32, 32]), 255)

  def test_gaussian_sprite_velocity_orients_anisotropy(self):
    red = _render_moving_gaussian_sprite((0.15, 0.0, 0.0))
    yy, xx = np.indices(red.shape)
    weights = red.astype(np.float64)
    total = weights.sum()
    self.assertGreater(total, 0.0)

    cx = (xx * weights).sum() / total
    cy = (yy * weights).sum() / total
    cov_xx = (((xx - cx) ** 2) * weights).sum() / total
    cov_yy = (((yy - cy) ** 2) * weights).sum() / total

    self.assertGreater(cov_xx, cov_yy * 1.4)

  @parameterized.parameters(2, 512)
  def test_render(self, nworld: int):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)

    rc = mjw.create_render_context(
      mjm,
      nworld=nworld,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )

    mjw.render(m, d, rc)

    rgb = rc.rgb_data.numpy()
    depth = rc.depth_data.numpy()

    self.assertGreater(np.count_nonzero(rgb), 0)
    self.assertGreater(np.count_nonzero(depth), 0)

    self.assertNotEqual(np.unique(rgb).shape[0], 1)
    self.assertNotEqual(np.unique(depth).shape[0], 1)

  def test_render_humanoid(self):
    mjm, mjd, m, d = test_data.fixture("humanoid/humanoid.xml")
    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )
    mjw.render(m, d, rc)
    rgb = rc.rgb_data.numpy()

    self.assertNotEqual(np.unique(rgb).shape[0], 1)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires CUDA.")
  def test_render_graph_capture(self):
    mjm, mjd, m, d = test_data.fixture("humanoid/humanoid.xml")
    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )

    mjw.render(m, d, rc)
    rgb_np = rc.rgb_data.numpy()

    with wp.ScopedCapture() as capture:
      mjw.render(m, d, rc)

    wp.capture_launch(capture.graph)

    _assert_eq(rgb_np, rc.rgb_data.numpy(), "rgb_data")

  @absltest.skipIf(not _HAS_RENDERER, "MuJoCo rendering requires OpenGL")
  def test_depth_matches_mujoco(self):
    """Depth values should match native MuJoCo (planar depth, not Euclidean)."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=1)
    cam_w, cam_h = 32, 32

    # mjwarp depth
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_rgb=[False],
      render_depth=[True],
    )
    mjw.render(m, d, rc)
    warp_depth = rc.depth_data.numpy()[0]  # flat array for world 0

    # Native MuJoCo depth
    with mujoco.Renderer(mjm, height=cam_h, width=cam_w) as renderer:
      renderer.update_scene(mjd, camera=0)
      renderer.enable_depth_rendering()
      mj_depth = renderer.render().flatten()

    # Compare only pixels that hit geometry (non-zero in both)
    valid = (warp_depth > 0) & (mj_depth > 0)
    np.testing.assert_allclose(
      warp_depth[valid],
      mj_depth[valid],
      atol=1e-2,
      rtol=1e-2,
    )

  def test_rgba_alpha_zero_keeps_background(self):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=1)
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=False,
    )

    ngeom = int(m.geom_rgba.shape[1])
    nmat = int(m.mat_rgba.shape[1])
    m.geom_rgba = wp.array2d([[wp.vec4(1.0, 1.0, 1.0, 0.0)] * ngeom], dtype=wp.vec4)
    m.mat_rgba = wp.array2d([[wp.vec4(1.0, 1.0, 1.0, 0.0)] * nmat], dtype=wp.vec4)

    mjw.render(m, d, rc)
    rgb = rc.rgb_data.numpy()[0]
    bg = np.uint32(int(rc.background_color))
    self.assertTrue(np.all(rgb == bg))


if __name__ == "__main__":
  wp.init()
  absltest.main()
