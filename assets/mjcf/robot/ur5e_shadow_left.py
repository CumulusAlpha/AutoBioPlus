from __future__ import annotations

import copy
from pathlib import Path

import mujoco
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from src import MJCF_ROOT
from assets.mjcf.robot.common import enable_gravity_compensation, make_pd_actuator


UR5E_XML: Path = MJCF_ROOT / "robot" / "ur5e.xml"
SHADOW_LEFT_XML: Path = MJCF_ROOT / "hand" / "shadow_left.xml"
assert UR5E_XML.exists(), f"Missing MJCF: {UR5E_XML}"
assert SHADOW_LEFT_XML.exists(), f"Missing MJCF: {SHADOW_LEFT_XML}"
UR5E_SHADOW_LEFT_ARM_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

UR5E_SHADOW_LEFT_HAND_JOINT_ORDER = (
    "hand:lh_THJ5",
    "hand:lh_THJ4",
    "hand:lh_THJ3",
    "hand:lh_THJ2",
    "hand:lh_THJ1",
    "hand:lh_FFJ4",
    "hand:lh_FFJ3",
    "hand:lh_MFJ4",
    "hand:lh_MFJ3",
    "hand:lh_RFJ4",
    "hand:lh_RFJ3",
    "hand:lh_LFJ5",
    "hand:lh_LFJ4",
    "hand:lh_LFJ3",
)

UR5E_SHADOW_LEFT_HAND_TENDON = (
    "hand:lh_FFJ0",
    "hand:lh_MFJ0",
    "hand:lh_RFJ0",
    "hand:lh_LFJ0",
)

UR5E_SHADOW_LEFT_HAND_TENDON_EFFORT_LIMITS = {
    "hand:lh_FFJ0": 1.0,
    "hand:lh_MFJ0": 1.0,
    "hand:lh_RFJ0": 1.0,
    "hand:lh_LFJ0": 1.0,
}


UR5E_SHADOW_LEFT_ARM_ACTUATORS = (
    make_pd_actuator("shoulder_pan", 2000.0, 100.0, 150.0),
    make_pd_actuator("shoulder_lift", 2000.0, 100.0, 150.0),
    make_pd_actuator("elbow", 2000.0, 100.0, 150.0),
    make_pd_actuator("wrist_1", 500.0, 25.0, 28.0),
    make_pd_actuator("wrist_2", 500.0, 25.0, 28.0),
    make_pd_actuator("wrist_3", 500.0, 25.0, 28.0),
)

UR5E_SHADOW_LEFT_HAND_JOINT_ACTUATORS = tuple(
    make_pd_actuator(joint_name, 5.0, 0.0, 3.0, 0.0002)
    for joint_name in UR5E_SHADOW_LEFT_HAND_JOINT_ORDER
)

UR5E_SHADOW_LEFT_HAND_TENDON_ACTUATORS = tuple(
    make_pd_actuator(
        tendon_name,
        5.0,
        0.0,
        UR5E_SHADOW_LEFT_HAND_TENDON_EFFORT_LIMITS[tendon_name],
        0.0002,
        transmission_type=TransmissionType.TENDON,
    )
    for tendon_name in UR5E_SHADOW_LEFT_HAND_TENDON
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        *UR5E_SHADOW_LEFT_ARM_ACTUATORS,
        *UR5E_SHADOW_LEFT_HAND_JOINT_ACTUATORS,
        *UR5E_SHADOW_LEFT_HAND_TENDON_ACTUATORS,
    ),
)


def get_spec() -> mujoco.MjSpec:
    ur5e_spec = mujoco.MjSpec.from_file(str(UR5E_XML))
    hand_spec = mujoco.MjSpec.from_file(str(SHADOW_LEFT_XML))

    for actuator in list(ur5e_spec.actuators):
        ur5e_spec.delete(actuator)
    for actuator in list(hand_spec.actuators):
        hand_spec.delete(actuator)
    for joint_name in ("lh_WRJ2", "lh_WRJ1"):
        joint = hand_spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"Missing Shadow left wrist joint in {SHADOW_LEFT_XML}: {joint_name}")
        # The Shadow wrist is rigidly mounted to the UR5e wrist. Removing these
        # joints fixes their child bodies at the zero pose and keeps them out of
        # the entity qpos consumed by learned policies.
        hand_spec.delete(joint)

    wrist_body = ur5e_spec.body("wrist_3_link")
    if wrist_body is None:
        raise ValueError(f"Missing UR5e wrist body in {UR5E_XML}: wrist_3_link")

    hand_body = hand_spec.body("lh_forearm")
    if hand_body is None:
        raise ValueError(f"Missing Shadow left base body in {SHADOW_LEFT_XML}: lh_forearm")

    hand_frame = wrist_body.add_frame(
        pos=(0.0, 0.1, 0.0),
        quat=(0.7071068, 0.0, 0.0, 0.7071068),
    )
    hand_frame.attach_body(hand_body, "hand:", "")
    enable_gravity_compensation(ur5e_spec.body("base"))
    return ur5e_spec


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
        **{joint_name: 0.0 for joint_name in UR5E_SHADOW_LEFT_HAND_JOINT_ORDER},
    },
    joint_vel={".*": 0.0},
)


def get_ur5e_shadow_left_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=copy.deepcopy(HOME_KEYFRAME),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    ur5e_shadow_left = Entity(get_ur5e_shadow_left_cfg())
    mujoco.viewer.launch(ur5e_shadow_left.spec.compile())
