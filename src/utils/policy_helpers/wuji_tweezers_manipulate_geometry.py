"""WujiTweezersGeometry: extracted geometry behavior for wuji_tweezers_manipulate."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.observations import observation_tensor


class WujiTweezersGeometry:
    """Provide geometry methods used by the task policy."""

    def _get_tweezers_pose_w(self, obs) -> torch.Tensor:
        try:
            pose = observation_tensor(
                obs,
                "tweezers_body_pose_w",
                self.device,
                self.num_envs,
                select_first=True,
            )[:, :7]
        except KeyError:
            tweezers = self.env.scene["tweezers"]
            pose = getattr(tweezers.data, "root_link_pose_w", None)
            if pose is None:
                raise
            pose = pose.to(device=self.device, dtype=torch.float32)
            if pose.ndim == 1:
                pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
            pose = pose[:, :7]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _get_initial_tweezers_pose_w(self) -> torch.Tensor:
        tweezers = self.env.scene["tweezers"]
        pose = tweezers.data.default_root_state[:, :7].to(
            device=self.device,
            dtype=torch.float32,
        )
        pose = pose.clone()
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _get_tweezers_site_pos_w(self) -> torch.Tensor:
        return self._get_site_pos_w(
            self._resolve_tweezers_site_id(), self.tweezers_site_name
        )

    def _get_tube_site_pos_w(self, obs) -> torch.Tensor:
        if not self.tube_site_name:
            return self._get_tube_root_pos_w(obs)
        site_id = self._resolve_tube_site_id()
        if site_id is not None:
            return self._get_site_pos_w(site_id, self.tube_site_name)
        return self._get_tube_root_pos_w(obs)

    def _resolve_tweezers_site_id(self) -> int:
        if self._tweezers_site_id is None:
            self._tweezers_site_id = self._resolve_site_id(self.tweezers_site_name)
        return self._tweezers_site_id

    def _resolve_tube_site_id(self) -> int | None:
        if self._tube_site_id is None:
            self._tube_site_id = self._resolve_site_id(
                self.tube_site_name, required=False
            )
        return self._tube_site_id

    def _resolve_beaker_center_site_id(self) -> int | None:
        if self._beaker_center_site_id is None:
            self._beaker_center_site_id = self._resolve_site_id(
                self.beaker_center_site_name,
                required=False,
            )
        return self._beaker_center_site_id

    def _resolve_site_id(self, site_name: str, required: bool = True) -> int | None:
        try:
            model = self.env.sim.mj_model
        except Exception as exc:
            if required:
                raise RuntimeError(
                    "MuJoCo model is required to resolve site positions."
                ) from exc
            return None

        for site_id in range(model.nsite):
            name = model.site(site_id).name or ""
            if (
                name == site_name
                or name.endswith(f"/{site_name}")
                or site_name.endswith(f"/{name}")
            ):
                return site_id

        if required:
            raise KeyError(f"Could not find MuJoCo site {site_name!r}.")
        return None

    def _get_site_pos_w(self, site_id: int, site_name: str) -> torch.Tensor:
        try:
            xpos = self.env.sim.data.site_xpos
        except Exception as exc:
            raise RuntimeError(
                f"MuJoCo site position is unavailable for {site_name!r}."
            ) from exc

        xpos = torch.as_tensor(xpos, device=self.device, dtype=torch.float32)
        pos = xpos[:, site_id] if xpos.ndim == 3 else xpos[site_id].unsqueeze(0)
        if pos.shape[0] == 1 and self.num_envs > 1:
            pos = pos.repeat(self.num_envs, 1)
        return pos

    def _get_tube_root_pos_w(self, obs) -> torch.Tensor:
        return self._get_tube_pose_w(obs)[:, :3]

    def _get_beaker_center_pos_w(self) -> torch.Tensor:
        site_id = self._resolve_beaker_center_site_id()
        if site_id is not None:
            return self._get_site_pos_w(site_id, self.beaker_center_site_name)
        return self._get_entity_root_pos_w("beaker")

    def _get_tube_pose_w(self, obs) -> torch.Tensor:
        try:
            pose = observation_tensor(
                obs, "tube_body_pose_w", self.device, self.num_envs, select_first=True
            )
            pose = pose[:, :7]
            pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
            return pose
        except KeyError:
            pass

        try:
            tube = self.env.scene["tube"]
            pose = getattr(tube.data, "root_link_pose_w", None)
        except Exception:
            pose = None
        if pose is None:
            raise KeyError(
                f"Observation/model missing tube target site {self.tube_site_name!r} and tube root pose."
            )
        pose = pose.to(device=self.device, dtype=torch.float32)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
        pose = pose[:, :7]
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose

    def _get_entity_root_pos_w(self, entity_name: str) -> torch.Tensor:
        try:
            entity = self.env.scene[entity_name]
            pose = getattr(entity.data, "root_link_pose_w", None)
        except Exception:
            pose = None
        if pose is None:
            raise KeyError(f"Scene entity {entity_name!r} is missing root_link_pose_w.")
        pose = pose.to(device=self.device, dtype=torch.float32)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0).repeat(self.num_envs, 1)
        return pose[:, :3]
