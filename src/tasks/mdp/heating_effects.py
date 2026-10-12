from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.managers.scene_entity_config import SceneEntityCfg


@wp.kernel
def _render_bubbles_kernel(
    time: wp.array(dtype=float),
    site_xpos: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    geom_ids: wp.array(dtype=int),
    intensity: wp.array(dtype=float),
    stir_momentum: wp.array(dtype=float),
    stir_phase: wp.array(dtype=float),
    beaker_site_id: int,
    particle_count: float,
):
    world_id, particle_id = wp.tid()
    geom_id = geom_ids[particle_id]
    center = site_xpos[world_id, beaker_site_id]
    threshold = (float(particle_id) + 1.0) / (particle_count + 1.0)
    effect_strength = wp.max(intensity[world_id], stir_momentum[world_id])
    if effect_strength < threshold:
        geom_xpos[world_id, geom_id] = center
        geom_size[world_id, geom_id] = wp.vec3(0.0001, 0.0001, 0.0001)
        geom_rgba[world_id, geom_id] = wp.vec4(0.72, 0.90, 1.0, 0.0)
        return

    particle = float(particle_id)
    random_life = particle * 0.6180339
    random_life = random_life - wp.floor(random_life)
    lifetime = 1.8 + 0.9 * random_life
    progress = time[world_id] / lifetime + particle * 0.137
    progress = progress - wp.floor(progress)

    vortex = stir_momentum[world_id]
    angle = particle * 2.3999632 + stir_phase[world_id] - 3.0 * progress
    radial_layer = 0.024 * wp.sqrt((particle + 0.5) / particle_count)
    funnel_radius = radial_layer * (0.22 + 0.78 * progress)
    spawn_radius = funnel_radius * (0.35 + 0.65 * vortex)
    wobble = wp.sin(progress * 6.2831853 + particle) * 0.002
    eased_height = progress * progress * (3.0 - 2.0 * progress)
    geom_xpos[world_id, geom_id] = wp.vec3(
        center[0] + spawn_radius * wp.cos(angle) + wobble,
        center[1] + spawn_radius * wp.sin(angle) - wobble,
        center[2] - 0.041 + 0.061 * eased_height,
    )

    base_radius = 0.0012 + 0.00035 * float(particle_id % 5)
    radius = base_radius * (1.0 + 1.8 * progress)
    fade = wp.sin(3.1415927 * progress)
    geom_size[world_id, geom_id] = wp.vec3(radius, radius, radius)
    geom_rgba[world_id, geom_id] = wp.vec4(0.72, 0.90, 1.0, 0.52 * fade)


@wp.kernel
def _render_rotor_kernel(
    site_xpos: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    stir_phase: wp.array(dtype=float),
    rotor_geom_id: int,
    beaker_site_id: int,
):
    world_id = wp.tid()
    center = site_xpos[world_id, beaker_site_id]
    geom_xpos[world_id, rotor_geom_id] = wp.vec3(
        center[0],
        center[1],
        center[2] - 0.041,
    )

    angle = stir_phase[world_id]
    c = wp.cos(angle)
    s = wp.sin(angle)
    # Map the capsule's local Z axis into the horizontal stirring direction.
    geom_xmat[world_id, rotor_geom_id] = wp.mat33(
        0.0,
        s,
        c,
        0.0,
        -c,
        s,
        1.0,
        0.0,
        0.0,
    )


@dataclass
class RenderOnlyHeatingEffects:
    bubble_geom_names: tuple[str, ...]
    rotor_geom_name: str
    intensity_attr: str
    stir_momentum_attr: str
    stir_phase_attr: str

    def __post_init__(self) -> None:
        self._cache_device: wp.context.Devicelike | None = None
        self._bubble_geom_ids: wp.array | None = None
        self._rotor_geom_id: int | None = None
        self._beaker_site_id: int | None = None

    @staticmethod
    def _resolve_geom_ids(mj_model: mujoco.MjModel, names: tuple[str, ...]) -> list[int]:
        available = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in range(mj_model.ngeom)
        ]
        resolved = []
        for name in names:
            matches = [
                geom_id
                for geom_id, candidate in enumerate(available)
                if candidate == name or (candidate and candidate.endswith(f"/{name}"))
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Could not uniquely resolve heating effect geom {name!r}.")
            resolved.append(matches[0])
        return resolved

    def _ensure_cache(self, env, beaker_center_cfg: SceneEntityCfg) -> None:
        device = env.sim.wp_device
        if self._cache_device == device:
            return

        bubble_ids = self._resolve_geom_ids(env.sim.mj_model, self.bubble_geom_names)
        self._bubble_geom_ids = wp.array(bubble_ids, dtype=int, device=device)
        self._rotor_geom_id = self._resolve_geom_ids(
            env.sim.mj_model,
            (self.rotor_geom_name,),
        )[0]

        beaker = env.scene[beaker_center_cfg.name]
        site_ids = beaker_center_cfg.site_ids
        assert site_ids is not None and len(site_ids) == 1, "Please specify one beaker center site."
        self._beaker_site_id = int(beaker.indexing.site_ids[site_ids[0]].item())
        self._cache_device = device

    def launch(self, env, beaker_center_cfg: SceneEntityCfg) -> None:
        self._ensure_cache(env, beaker_center_cfg)
        assert self._bubble_geom_ids is not None
        assert self._rotor_geom_id is not None
        assert self._beaker_site_id is not None
        intensity = getattr(env, self.intensity_attr)
        stir_momentum = getattr(env, self.stir_momentum_attr)
        stir_phase = getattr(env, self.stir_phase_attr)
        common_inputs = [
            env.sim.wp_data.time,
            env.sim.wp_data.site_xpos,
            env.sim.wp_data.geom_xpos,
            env.sim.wp_model.geom_size,
            env.sim.wp_model.geom_rgba,
        ]
        wp.launch(
            _render_bubbles_kernel,
            dim=(env.num_envs, len(self.bubble_geom_names)),
            inputs=[
                *common_inputs,
                self._bubble_geom_ids,
                wp.from_torch(intensity, dtype=wp.float32),
                wp.from_torch(stir_momentum, dtype=wp.float32),
                wp.from_torch(stir_phase, dtype=wp.float32),
                self._beaker_site_id,
                float(len(self.bubble_geom_names)),
            ],
            device=env.sim.wp_device,
        )
        wp.launch(
            _render_rotor_kernel,
            dim=env.num_envs,
            inputs=[
                env.sim.wp_data.site_xpos,
                env.sim.wp_data.geom_xpos,
                env.sim.wp_data.geom_xmat,
                wp.from_torch(stir_phase, dtype=wp.float32),
                self._rotor_geom_id,
                self._beaker_site_id,
            ],
            device=env.sim.wp_device,
        )
        if env.sim._sensor_context is not None:
            mjwarp.refit_bvh(
                env.sim.wp_model,
                env.sim.wp_data,
                env.sim._sensor_context.render_context,
            )
