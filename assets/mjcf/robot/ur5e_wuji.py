"""UR5e + Wuji hand constants and EntityCfg for mjlab."""
from __future__ import annotations

import copy
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from src import MJCF_ROOT
from assets.mjcf.robot import common as robot_common
from assets.mjcf.robot.common import load_spec_with_embedded_assets, make_pd_actuator

UR5E_WUJI_XML: Path = MJCF_ROOT / "robot" / "ur5e_wuji.xml"
assert UR5E_WUJI_XML.exists(), f"Missing MJCF: {UR5E_WUJI_XML}"

UR5E_JOINT_ORDER = robot_common.UR5E_JOINT_ORDER
UR5E_JOINT_SCALE = robot_common.UR5E_JOINT_SCALE
UR5E_JOINT_OFFSET = robot_common.UR5E_JOINT_OFFSET

WUJI_JOINT_ORDER = (
    r".*right_finger1_joint1",
    r".*right_finger1_joint2",
    r".*right_finger1_joint3",
    r".*right_finger1_joint4",
    r".*right_finger2_joint1",
    r".*right_finger2_joint2",
    r".*right_finger2_joint3",
    r".*right_finger2_joint4",
    r".*right_finger3_joint1",
    r".*right_finger3_joint2",
    r".*right_finger3_joint3",
    r".*right_finger3_joint4",
    r".*right_finger4_joint1",
    r".*right_finger4_joint2",
    r".*right_finger4_joint3",
    r".*right_finger4_joint4",
    r".*right_finger5_joint1",
    r".*right_finger5_joint2",
    r".*right_finger5_joint3",
    r".*right_finger5_joint4",
)

WUJI_HOME_JOINT_POS = {
    r".*right_finger1_joint1": 0.0475,
    r".*right_finger1_joint2": 0.0,
    r".*right_finger1_joint3": 0.0,
    r".*right_finger1_joint4": 0.0,
    r".*right_finger2_joint1": 0.0,
    r".*right_finger2_joint2": 0.0,
    r".*right_finger2_joint3": 0.0,
    r".*right_finger2_joint4": 0.0,
    r".*right_finger3_joint1": 0.0,
    r".*right_finger3_joint2": 0.0,
    r".*right_finger3_joint3": 0.0,
    r".*right_finger3_joint4": 0.0,
    r".*right_finger4_joint1": 0.0,
    r".*right_finger4_joint2": 0.0,
    r".*right_finger4_joint3": 0.0,
    r".*right_finger4_joint4": 0.0,
    r".*right_finger5_joint1": 0.0,
    r".*right_finger5_joint2": 0.0,
    r".*right_finger5_joint3": 0.0,
    r".*right_finger5_joint4": 0.0,
}

WUJI_JOINT_PD = {
    r".*right_finger1_joint1": {"kp": 0.40844710645048043, "kd": 0.020882010257063675, "effort": 0.4452, "armature": 0.0005},
    r".*right_finger1_joint2": {"kp": 0.6858601063643346, "kd": 0.030610939996373314, "effort": 0.4259, "armature": 0.0005},
    r".*right_finger1_joint3": {"kp": 0.2391112196099482, "kd": 0.010181475050560962, "effort": 0.1888, "armature": 0.0002},
    r".*right_finger1_joint4": {"kp": 0.20736128319711747, "kd": 0.00909698844675045, "effort": 0.1468, "armature": 0.0002},
    r".*right_finger2_joint1": {"kp": 0.37352218155860073, "kd": 0.01882274330029718, "effort": 0.6188, "armature": 0.0005},
    r".*right_finger2_joint2": {"kp": 0.45592448909027794, "kd": 0.019798167597016643, "effort": 0.1822, "armature": 0.0002},
    r".*right_finger2_joint3": {"kp": 0.24368366649522863, "kd": 0.010477031953162727, "effort": 0.2251, "armature": 0.0002},
    r".*right_finger2_joint4": {"kp": 0.18026971340925335, "kd": 0.008240212147903584, "effort": 0.2170, "armature": 0.0002},
    r".*right_finger3_joint1": {"kp": 0.3687093483646485, "kd": 0.01848622487024593, "effort": 0.6494, "armature": 0.0005},
    r".*right_finger3_joint2": {"kp": 0.4164253443634641, "kd": 0.018032947229953678, "effort": 0.1827, "armature": 0.0002},
    r".*right_finger3_joint3": {"kp": 0.22218607502059182, "kd": 0.009592200014076666, "effort": 0.2078, "armature": 0.0002},
    r".*right_finger3_joint4": {"kp": 0.19427606072023446, "kd": 0.009152994605972402, "effort": 0.2018, "armature": 0.0002},
    r".*right_finger4_joint1": {"kp": 0.35718151495111794, "kd": 0.018376606800780005, "effort": 0.6389, "armature": 0.0005},
    r".*right_finger4_joint2": {"kp": 0.42977315313086895, "kd": 0.01867700966212433, "effort": 0.1832, "armature": 0.0002},
    r".*right_finger4_joint3": {"kp": 0.24930151196247122, "kd": 0.01059512121009555, "effort": 0.2249, "armature": 0.0002},
    r".*right_finger4_joint4": {"kp": 0.2285032688178066, "kd": 0.009917602877441107, "effort": 0.2044, "armature": 0.0002},
    r".*right_finger5_joint1": {"kp": 0.3655325975433942, "kd": 0.018616960278988272, "effort": 0.6441, "armature": 0.0005},
    r".*right_finger5_joint2": {"kp": 0.41393113081120425, "kd": 0.018732177029667153, "effort": 0.1798, "armature": 0.0002},
    r".*right_finger5_joint3": {"kp": 0.22729367621965954, "kd": 0.00951005441616486, "effort": 0.2384, "armature": 0.0002},
    r".*right_finger5_joint4": {"kp": 0.1964723550341816, "kd": 0.009017295241756363, "effort": 0.1866, "armature": 0.0002},
}

UR5E_ACTUATORS = (
    make_pd_actuator("shoulder_pan", 2000.0, 100.0, 150.0),
    make_pd_actuator("shoulder_lift", 2000.0, 100.0, 150.0),
    make_pd_actuator("elbow", 2000.0, 100.0, 150.0),
    make_pd_actuator("wrist_1", 500.0, 25.0, 28.0),
    make_pd_actuator("wrist_2", 500.0, 25.0, 28.0),
    make_pd_actuator("wrist_3", 500.0, 25.0, 28.0),
)

WUJI_ACTUATORS = tuple(
    make_pd_actuator(
        joint_name,
        kp=WUJI_JOINT_PD[joint_name]["kp"],
        kd=WUJI_JOINT_PD[joint_name]["kd"],
        effort=WUJI_JOINT_PD[joint_name]["effort"],
        armature=WUJI_JOINT_PD[joint_name]["armature"],
    )
    for joint_name in WUJI_JOINT_ORDER
)

ARTICULATION = EntityArticulationInfoCfg(actuators=(*UR5E_ACTUATORS, *WUJI_ACTUATORS))


def get_spec() -> mujoco.MjSpec:
    return load_spec_with_embedded_assets(UR5E_WUJI_XML, base_body_name="base")


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, -0.45, 0.714),
    rot=(0.0, 0.0, 0.0, 1.0),
    joint_pos={
        "shoulder_pan": 1.5708,
        "shoulder_lift": -1.3744,
        "elbow": 1.3744,
        "wrist_1": 0.3927,
        "wrist_2": 1.5708,
        "wrist_3": 0.7854,
        **WUJI_HOME_JOINT_POS,
    },
    joint_vel={".*": 0.0},
)


def get_ur5e_wuji_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=copy.deepcopy(HOME_KEYFRAME),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    ur5e_wuji = Entity(get_ur5e_wuji_cfg())
    mujoco.viewer.launch(ur5e_wuji.spec.compile())
