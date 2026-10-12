from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

from src import MJCF_ROOT, MESH_ROOT


HEATING_PLATE_XML: Path = MJCF_ROOT / "instrument" / "heating_plate.xml"
HEATING_PLATE_ASSET_ROOT: Path = MESH_ROOT
assert HEATING_PLATE_XML.exists(), f"Missing MJCF: {HEATING_PLATE_XML}"


def get_heating_plate_spec() -> mujoco.MjSpec:
    """Load the heating plate MJCF with merge-safe embedded assets."""
    spec = mujoco.MjSpec.from_file(str(HEATING_PLATE_XML))
    assets: dict[str, bytes] = {}

    for asset in (*spec.meshes, *spec.textures):
        if not asset.file:
            continue
        asset_path = (HEATING_PLATE_ASSET_ROOT / asset.file).resolve()
        asset_key = f"heating_plate/{asset.file}"
        assets[asset_key] = asset_path.read_bytes()
        asset.file = asset_key

    spec.assets = assets
    return spec


def get_heating_plate_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(-0.4, 0.1, 0.824),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"knob_joint": 0.0},
            joint_vel={"knob_joint": 0.0},
        ),
        spec_fn=get_heating_plate_spec,
    )
