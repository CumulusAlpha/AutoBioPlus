"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, CameraSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def site_pose_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    
    asset: Entity = env.scene[asset_cfg.name]

    site_ids = asset_cfg.site_ids
    assert site_ids is not None, "Please specify site_names in SceneEntityCfg"

    # (num_envs, num_sites, 7)
    site_pose_w = asset.data.site_pose_w

    return site_pose_w[:, site_ids]

def body_pose_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    body_ids = asset_cfg.body_ids
    assert body_ids is not None, "Please specify body_names in SceneEntityCfg"

    # (num_envs, num_bodies, 7)
    body_pose_w = asset.data.body_link_pose_w

    return body_pose_w[:, body_ids]

def root_pose_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pose_w

def joint_pos_abs(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, jnt_ids]

def camera_rgb(env: ManagerBasedRlEnv, sensor_name: str):
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, CameraSensor)
    return sensor.data.rgb

def builtin_sensor_data(env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
    values = []
    for sensor_name in sensor_names:
        sensor = env.scene[sensor_name]
        assert isinstance(sensor, BuiltinSensor)
        values.append(sensor.data)
    return torch.cat(values, dim=-1)
