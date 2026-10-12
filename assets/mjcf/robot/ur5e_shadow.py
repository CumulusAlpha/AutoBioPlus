from __future__ import annotations

import copy
from pathlib import Path

import mujoco
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

from src import MJCF_ROOT
from assets.mjcf.robot.common import load_spec_with_embedded_assets, make_pd_actuator


UR5E_SHADOW_XML: Path = MJCF_ROOT / "robot" / "ur5e_shadow.xml"
assert UR5E_SHADOW_XML.exists(), f"Missing MJCF: {UR5E_SHADOW_XML}"


# UR5E arm joints (same actuator tuning as ur5e_gripper.py)
UR5E_SHADOW_ARM_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

# Shadow hand joints that are exposed to policy actions.
UR5E_SHADOW_HAND_JOINT_ORDER = (
    "hand:rh_THJ5",
    "hand:rh_THJ4",
    "hand:rh_THJ3",
    "hand:rh_THJ2",
    "hand:rh_THJ1",
    "hand:rh_FFJ4",
    "hand:rh_FFJ3",
    "hand:rh_MFJ4",
    "hand:rh_MFJ3",
    "hand:rh_RFJ4",
    "hand:rh_RFJ3",
    "hand:rh_LFJ5",
    "hand:rh_LFJ4",
    "hand:rh_LFJ3",
)

UR5E_SHADOW_HAND_TENDON = (
    "hand:rh_FFT1",
    "hand:rh_MFT1",
    "hand:rh_RFT1",
    "hand:rh_LFT1",
)

UR5E_SHADOW_HAND_TENDON_EFFORT_LIMITS = {
    "hand:rh_FFT1": 0.465773,
    "hand:rh_MFT1": 0.616938,
    "hand:rh_RFT1": 0.464343,
    "hand:rh_LFT1": 0.506173,
}


UR5E_SHADOW_ARM_ACTUATORS = (
    make_pd_actuator("shoulder_pan", 2000.0, 100.0, 150.0),
    make_pd_actuator("shoulder_lift", 2000.0, 100.0, 150.0),
    make_pd_actuator("elbow", 2000.0, 100.0, 150.0),
    make_pd_actuator("wrist_1", 500.0, 25.0, 28.0),
    make_pd_actuator("wrist_2", 500.0, 25.0, 28.0),
    make_pd_actuator("wrist_3", 500.0, 25.0, 28.0),
)

UR5E_SHADOW_HAND_JOINT_ACTUATORS = tuple(
    make_pd_actuator(joint_name, 5.0, 0.0, 3.0, 0.0002)
    for joint_name in UR5E_SHADOW_HAND_JOINT_ORDER
)

UR5E_SHADOW_HAND_TENDON_ACTUATORS = tuple(
    make_pd_actuator(
        tendon_name,
        5.0,
        0.0,
        UR5E_SHADOW_HAND_TENDON_EFFORT_LIMITS[tendon_name],
        0.0002,
        transmission_type=TransmissionType.TENDON,
    )
    for tendon_name in UR5E_SHADOW_HAND_TENDON
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        *UR5E_SHADOW_ARM_ACTUATORS,
        *UR5E_SHADOW_HAND_JOINT_ACTUATORS,
        *UR5E_SHADOW_HAND_TENDON_ACTUATORS,
    ),
)


def get_spec() -> mujoco.MjSpec:
    return load_spec_with_embedded_assets(UR5E_SHADOW_XML, base_body_name="base")


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.1, 0.0),
    rot=(0.0, 0.0, 0.0, -1.0),
    joint_pos={
        "shoulder_pan": 0.0,
        "shoulder_lift": -1.5708,
        "elbow": 1.5708,
        "wrist_1": -1.5708,
        "wrist_2": -1.5708,
        "wrist_3": -1.5708,
        **{joint_name: 0.0 for joint_name in UR5E_SHADOW_HAND_JOINT_ORDER},
    },
    joint_vel={".*": 0.0},
)


def get_ur5e_shadow_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=copy.deepcopy(HOME_KEYFRAME),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    ur5e_shadow = Entity(get_ur5e_shadow_cfg())
    mujoco.viewer.launch(ur5e_shadow.spec.compile())
