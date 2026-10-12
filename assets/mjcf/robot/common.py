from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.actuator.actuator import TransmissionType

UR5E_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

UR5E_JOINT_SCALE = {
    "shoulder_pan": 6.2832,
    "shoulder_lift": 6.2832,
    "elbow": 3.1415,
    "wrist_1": 6.2832,
    "wrist_2": 6.2832,
    "wrist_3": 6.2832,
}

UR5E_JOINT_OFFSET = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow": 0.0,
    "wrist_1": 0.0,
    "wrist_2": 0.0,
    "wrist_3": 0.0,
}


def make_pd_actuator(
    name: str,
    kp: float,
    kd: float,
    effort: float,
    armature: float = 0.02,
    transmission_type: TransmissionType = TransmissionType.JOINT,
) -> IdealPdActuatorCfg:
    return IdealPdActuatorCfg(
        target_names_expr=(name,),
        transmission_type=transmission_type,
        stiffness=kp,
        damping=kd,
        effort_limit=effort,
        armature=armature,
    )


def enable_gravity_compensation(body: mujoco.MjsBody) -> None:
    body.gravcomp = 1
    for child in body.bodies:
        enable_gravity_compensation(child)


def load_spec_with_embedded_assets(xml_path: Path, base_body_name: str) -> mujoco.MjSpec:
    """Load MJCF, clear XML actuators, set gravcomp recursively and embed mesh assets."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    enable_gravity_compensation(spec.body(base_body_name))

    return spec
