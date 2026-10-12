"""
UR5E Gripper constants and EntityCfg for mjlab.

Minimal version:
- Use XML <position> actuators directly
- No current-domain PD
- No delay
- No hardware mapping
- Pure simulation usage
"""
from __future__ import annotations

import copy
from pathlib import Path
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

from src import MJCF_ROOT
from assets.mjcf.robot import common as robot_common
from assets.mjcf.robot.common import (
    load_spec_with_embedded_assets,
    make_pd_actuator,
)

# -----------------------------------------------------------------------------
# XML path
# -----------------------------------------------------------------------------
UR5E_XHAND_XML: Path = (
    MJCF_ROOT
    / "robot"
    / "ur5e_xhand.xml"
)
assert UR5E_XHAND_XML.exists(), f"Missing MJCF: {UR5E_XHAND_XML}"

UR5E_JOINT_ORDER = robot_common.UR5E_JOINT_ORDER
UR5E_JOINT_SCALE = robot_common.UR5E_JOINT_SCALE
UR5E_JOINT_OFFSET = robot_common.UR5E_JOINT_OFFSET

XHAND_JOINT_ORDER = (
    r".*thumb_bend_joint",
    r".*thumb_rota_joint1",
    r".*thumb_rota_joint2",

    r".*index_bend_joint",
    r".*index_joint1",
    r".*index_joint2",

    r".*mid_joint1",
    r".*mid_joint2",

    r".*ring_joint1",
    r".*ring_joint2",

    r".*pinky_joint1",
    r".*pinky_joint2",
)

UR5E_ACTUATORS = (

    # big joints
    make_pd_actuator("shoulder_pan", 2000.0, 100.0, 150.0),
    make_pd_actuator("shoulder_lift", 2000.0, 100.0, 150.0),
    make_pd_actuator("elbow", 2000.0, 100.0, 150.0),

    # small wrist joints
    make_pd_actuator("wrist_1", 500.0, 25.0, 28.0),
    make_pd_actuator("wrist_2", 500.0, 25.0, 28.0),
    make_pd_actuator("wrist_3", 500.0, 25.0, 28.0),
)

XHAND_ACTUATORS = tuple(
    make_pd_actuator(joint_name, 300, 35, 20, 1,)
    for joint_name in XHAND_JOINT_ORDER
)



ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        *UR5E_ACTUATORS,
        *XHAND_ACTUATORS,
    ),
)

# -----------------------------------------------------------------------------
# Spec
# -----------------------------------------------------------------------------

def get_spec() -> mujoco.MjSpec:
    return load_spec_with_embedded_assets(UR5E_XHAND_XML, base_body_name="base")

# -----------------------------------------------------------------------------
# Home keyframe
# -----------------------------------------------------------------------------

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, -0.45, 0.714),
    rot=(0.0, 0.0, 0.0, 1.0),

    joint_pos={
        # UR5e
        "shoulder_pan": 1.5708,
        "shoulder_lift": -1.3744,
        "elbow": 1.3744,
        "wrist_1": 0.3927,
        "wrist_2": 1.5708,
        "wrist_3": 0.7854,

        # XHand (prefix compatible)
        r".*thumb_bend_joint": 0.3491,
        r".*thumb_rota_joint1": 0.3491,
        r".*thumb_rota_joint2": 0.3491,

        r".*index_bend_joint": 0.0873,
        r".*index_joint1": 0.3491,
        r".*index_joint2": 0.3491,

        r".*mid_joint1": 0.3491,
        r".*mid_joint2": 0.3491,

        r".*ring_joint1": 0.3491,
        r".*ring_joint2": 0.3491,

        r".*pinky_joint1": 0.3491,
        r".*pinky_joint2": 0.3491,
    },

    joint_vel={".*": 0.0},
)

# -----------------------------------------------------------------------------
# EntityCfg
# -----------------------------------------------------------------------------

def get_ur5e_xhand_cfg() -> EntityCfg:
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
    ur5e_xhand = Entity(get_ur5e_xhand_cfg())
    mujoco.viewer.launch(ur5e_xhand.spec.compile())
