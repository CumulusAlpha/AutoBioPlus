from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv


FOG_CENTER = (0.0, 0.0, 0.92)
FOG_RADIUS = 0.30
FOG_HEIGHT = 0.90
FOG_TIME_SCALE = 0.42
@dataclass(frozen=True)
class FogParticleParams:
    xy0: tuple[tuple[float, float], ...]
    lifetime: tuple[float, ...]
    initial_age: tuple[float, ...]
    phase: tuple[float, ...]
    drift: tuple[tuple[float, float], ...]


@wp.kernel
def _render_only_fog_kernel(
    time: wp.array(dtype=float),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    geom_ids: wp.array(dtype=int),
    fog_center: wp.vec3,
    fog_radius: float,
    fog_height: float,
    fog_time_scale: float,
    xy0: wp.array2d(dtype=float),
    base_radius: wp.array(dtype=float),
    lifetime: wp.array(dtype=float),
    initial_age: wp.array(dtype=float),
    phase: wp.array(dtype=float),
    drift: wp.array2d(dtype=float),
    alpha: wp.array(dtype=float),
    color: wp.vec3,
):
    world_id, particle_id = wp.tid()
    fog_time = time[world_id] * fog_time_scale

    age = initial_age[particle_id] + fog_time
    life = lifetime[particle_id]
    age = age - life * wp.floor(age / life)
    progress = age / life

    p = phase[particle_id]
    spread = fog_radius * (0.14 + 0.96 * progress)
    wobble_x = wp.sin(2.0 * fog_time + p) * spread * 0.30
    wobble_y = wp.cos(1.6 * fog_time + p * 0.75) * spread * 0.30
    drift_x = drift[particle_id, 0] * fog_radius * 0.18 * progress
    drift_y = drift[particle_id, 1] * fog_radius * 0.12 * progress

    geom_id = geom_ids[particle_id]
    geom_xpos[world_id, geom_id] = wp.vec3(
        fog_center[0] + xy0[particle_id, 0] + wobble_x + drift_x,
        fog_center[1] + xy0[particle_id, 1] + wobble_y + drift_y,
        fog_center[2] + fog_height * progress,
    )

    fade = wp.sin(3.14159265359 * progress)
    radius = base_radius[particle_id] * (1.0 + 1.9 * progress)
    geom_size[world_id, geom_id] = wp.vec3(radius, radius, radius)
    geom_rgba[world_id, geom_id] = wp.vec4(
        color[0],
        color[1],
        color[2],
        alpha[particle_id] * fade,
    )


@dataclass
class RenderOnlyFogInstaller:
    """Install a sensor-pipeline hook that moves fog geoms only for rendering."""

    particle_params: FogParticleParams
    base_radius: tuple[float, ...]
    alpha: tuple[float, ...]
    geom_names: tuple[str, ...]
    fog_center: tuple[float, float, float] = FOG_CENTER
    fog_radius: float = FOG_RADIUS
    fog_height: float = FOG_HEIGHT
    fog_time_scale: float = FOG_TIME_SCALE
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gaussian_sprite_config: mjwarp.GaussianSpriteConfig = field(
        default_factory=mjwarp.GaussianSpriteConfig
    )

    def __post_init__(self) -> None:
        self._installed = False
        self._cache_device: wp.context.Devicelike | None = None
        self._geom_id_values: list[int] | None = None
        self._geom_ids: wp.array | None = None
        self._xy0: wp.array | None = None
        self._base_radius: wp.array | None = None
        self._lifetime: wp.array | None = None
        self._initial_age: wp.array | None = None
        self._phase: wp.array | None = None
        self._drift: wp.array | None = None
        self._alpha: wp.array | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        del env_ids
        if self._installed:
            return
        if env.sim._sensor_context is None:
            return

        self._ensure_cache(env)
        render_context = env.sim._sensor_context.render_context
        assert self._geom_id_values is not None
        mjwarp.configure_gaussian_sprites(
            render_context,
            self._geom_id_values,
            self.gaussian_sprite_config,
        )

        def sense_kernel(sim) -> None:
            assert sim._sensor_context is not None
            ctx = sim._sensor_context
            rc = ctx.render_context
            self.launch(sim)
            mjwarp.refit_bvh(sim.wp_model, sim.wp_data, rc)
            if ctx.has_cameras:
                mjwarp.render(sim.wp_model, sim.wp_data, rc)
                ctx.unpack_rgb()
            for sensor in ctx.raycast_sensors:
                sensor.raycast_kernel(rc=rc)

        env.sim._sense_kernel = MethodType(sense_kernel, env.sim)
        env.sim.create_graph()
        self._installed = True

    def _resolve_geom_ids(self, mj_model: mujoco.MjModel) -> list[int]:
        ids = []
        available = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            for i in range(mj_model.ngeom)
        ]
        for name in self.geom_names:
            geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                matches = [
                    i
                    for i, candidate in enumerate(available)
                    if candidate and candidate.endswith(f"/{name}")
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"Could not resolve fog geom '{name}'.")
                geom_id = matches[0]
            ids.append(geom_id)
        return ids

    def _ensure_cache(self, env: ManagerBasedRlEnv) -> None:
        device = env.sim.wp_device
        if self._cache_device == device:
            return

        geom_ids = self._resolve_geom_ids(env.sim.mj_model)
        self._geom_id_values = geom_ids
        self._geom_ids = wp.array(geom_ids, dtype=int, device=device)
        self._xy0 = wp.array(self.particle_params.xy0, dtype=float, device=device)
        self._base_radius = wp.array(self.base_radius, dtype=float, device=device)
        self._lifetime = wp.array(self.particle_params.lifetime, dtype=float, device=device)
        self._initial_age = wp.array(
            self.particle_params.initial_age, dtype=float, device=device
        )
        self._phase = wp.array(self.particle_params.phase, dtype=float, device=device)
        self._drift = wp.array(self.particle_params.drift, dtype=float, device=device)
        self._alpha = wp.array(self.alpha, dtype=float, device=device)
        self._cache_device = device

    def launch(self, sim) -> None:
        assert self._geom_ids is not None
        assert self._xy0 is not None
        assert self._base_radius is not None
        assert self._lifetime is not None
        assert self._initial_age is not None
        assert self._phase is not None
        assert self._drift is not None
        assert self._alpha is not None
        wp.launch(
            _render_only_fog_kernel,
            dim=(sim.num_envs, len(self.geom_names)),
            inputs=[
                sim.wp_data.time,
                sim.wp_data.geom_xpos,
                sim.wp_model.geom_size,
                sim.wp_model.geom_rgba,
                self._geom_ids,
                wp.vec3(*self.fog_center),
                self.fog_radius,
                self.fog_height,
                self.fog_time_scale,
                self._xy0,
                self._base_radius,
                self._lifetime,
                self._initial_age,
                self._phase,
                self._drift,
                self._alpha,
                wp.vec3(*self.color),
            ],
            device=sim.wp_device,
        )
