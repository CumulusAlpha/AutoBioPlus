from __future__ import annotations

from pathlib import Path

import mujoco
import copy

from src import MJCF_ROOT
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from assets.mjcf.robot.common import load_spec_with_embedded_assets, make_pd_actuator


# -----------------------------------------------------------------------------
# XML path
# -----------------------------------------------------------------------------
ALOHA_LEFT_XML: Path = (
    MJCF_ROOT
    / "robot"
    / "aloha_left.xml"
)
assert ALOHA_LEFT_XML.exists(), f"Missing MJCF: {ALOHA_LEFT_XML}"


ALOHA_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "left_finger",
)


ALOHA_ACTUATORS = (
    # base joints (values match XML defaults when possible)
    make_pd_actuator("waist", 43.0, 5.76, 35.0),
    make_pd_actuator("shoulder", 265.0, 20.0, 144.0),
    make_pd_actuator("elbow", 227.0, 18.49, 59.0),
    make_pd_actuator("forearm_roll", 78.0, 6.78, 22.0),
    make_pd_actuator("wrist_angle", 37.0, 6.28, 35.0),
    make_pd_actuator("wrist_rotate", 10.4, 1.2, 35.0),
    # gripper (left finger drives both via equality)
    make_pd_actuator("left_finger", 2000.0, 124.0, 35.0),
)


ARTICULATION = EntityArticulationInfoCfg(
    actuators=ALOHA_ACTUATORS,
)


# -----------------------------------------------------------------------------
# Spec
# -----------------------------------------------------------------------------

def get_spec() -> mujoco.MjSpec:
    spec = load_spec_with_embedded_assets(ALOHA_LEFT_XML, base_body_name="left/base_link")
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    return spec


# -----------------------------------------------------------------------------
# Home keyframe
# -----------------------------------------------------------------------------

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
    joint_pos={
        "waist": 0.0,
        "shoulder": -0.96,
        "elbow": 1.16,
        "forearm_roll": 0.0,
        "wrist_angle": -0.3,
        "wrist_rotate": 0.0,
        "left_finger": 0.0084,
        "right_finger": 0.0084,
    },
    joint_vel={".*": 0.0},
)


# -----------------------------------------------------------------------------
# EntityCfg
# -----------------------------------------------------------------------------

def get_aloha_left_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=copy.deepcopy(HOME_KEYFRAME),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# -----------------------------------------------------------------------------
# Debug viewer
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    aloha_left = Entity(get_aloha_left_cfg())
    mujoco.viewer.launch(aloha_left.spec.compile())
