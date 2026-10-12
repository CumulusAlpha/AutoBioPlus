import platform
import warnings

import mujoco
from pathlib import Path

from mjlab.entity import EntityCfg
import numpy as np

from src import DEXBIOLAB_SRC_ROOT, MJCF_ROOT


def _load_thread_plugin_if_available() -> None:
    plugin_path = DEXBIOLAB_SRC_ROOT / "libablab.so.3.6.0"
    if platform.system() == "Darwin":
        warnings.warn(
            f"Skipping MuJoCo thread plugin on macOS: {plugin_path}. "
            "This bundled library is not a macOS/arm64 binary; screw-thread "
            "objects may not run until a compatible plugin is provided.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    mujoco.mj_loadPluginLibrary(str(plugin_path))


_load_thread_plugin_if_available()

CENTRIFUGE_50ML_XML: Path = (
    MJCF_ROOT
    / "object"
    / "centrifuge_50ml.xml"
)

assert CENTRIFUGE_50ML_XML.exists(), f"Missing MJCF: {CENTRIFUGE_50ML_XML}"


def get_centrifuge_50ml_spec() -> mujoco.MjSpec:
    """Load MJCF and embed mesh assets."""
    spec = mujoco.MjSpec.from_file(str(CENTRIFUGE_50ML_XML))

    return spec


def get_centrifuge_50ml_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            # Pose of the wrap body in scene.xml
            pos=(0.0, 0.0, 0.83),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spec_fn=get_centrifuge_50ml_spec,
    )

CENTRIFUGE_50ML_XML_SCREW: Path = (
    MJCF_ROOT
    / "object"
    / "centrifuge_50ml_screw.xml"
)


assert CENTRIFUGE_50ML_XML_SCREW.exists(), f"Missing MJCF: {CENTRIFUGE_50ML_XML_SCREW}"

##########screw part
def get_centrifuge_50ml_cap_spec() -> mujoco.MjSpec:
    """Load MJCF and embed mesh assets."""
    spec = mujoco.MjSpec.from_file(str(CENTRIFUGE_50ML_XML_SCREW))

    # Remove body, keep only cap
    body = spec.body("centrifuge_50ml_screw_body")
    if body is not None:
        spec.delete(body)

    # Get cap
    cap = spec.body("centrifuge_50ml_screw_cap")


    cap.add_joint(
    name="centrifuge_50ml_screw_cap",
    type=mujoco.mjtJoint.mjJNT_FREE,
    )

    # Add site (origin3)
    cap.add_site(
        name="origin3",
        pos=[0, 0, +0.005],
        quat=[1, 0, 0, 0],
        size=[0.005],
        rgba=[1, 0, 0, 1],
        group=3,
    )
    return spec

def get_centrifuge_50ml_cap_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            # Pose of the wrap body in scene.xml
            pos=(0.1, -0.2, 0.83765),
            rot=(0, 1, 0.0, 0.0),
        ),
        spec_fn=get_centrifuge_50ml_cap_spec,
    )

def get_centrifuge_50ml_body_spec() -> mujoco.MjSpec:
    """Load MJCF and embed mesh assets."""
    spec = mujoco.MjSpec.from_file(str(CENTRIFUGE_50ML_XML_SCREW))

    # Remove cap, keep only body
    cap = spec.body("centrifuge_50ml_screw_cap")
    if cap is not None:
        spec.delete(cap)

    # Get body
    body = spec.body("centrifuge_50ml_screw_body")

    body.add_joint(
        name="centrifuge_50ml_screw_body",
        type=mujoco.mjtJoint.mjJNT_FREE,
    )
    # Add site (origin4)
    body.add_site(
        name="origin4",
        pos=[0.0, 0.0, 0.01234],
        size=[0.005],
        rgba=[1, 0, 0, 1],
        group=3,
    )
    return spec


def get_centrifuge_50ml_body_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            # Pose of the wrap body in scene.xml
            pos=(-0.018, 0.072, 0.83),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spec_fn=get_centrifuge_50ml_body_spec,
    )

# CENTRIFUGE_50ML_BODY_XML: Path = (
#     DEXBIOLAB_SRC_ROOT
#     / "model"
#     / "object"
#     / "centrifuge_50ml_body.xml"
# )

# def get_centrifuge_50ml_body_spec() -> mujoco.MjSpec:
#     """Load MJCF and embed mesh assets."""
#     spec = mujoco.MjSpec.from_file(str(CENTRIFUGE_50ML_BODY_XML))

#     mesh_root = (CENTRIFUGE_50ML_BODY_XML.parent / spec.meshdir).resolve()

#     if spec.meshdir:
#         assets: dict[str, bytes] = {}

#         update_assets(
#             assets,
#             mesh_root,
#             spec.meshdir,
#             recursive=True,
#         )

#         spec.assets = assets

#     return spec


# def get_centrifuge_50ml_body_cfg() -> EntityCfg:
#     return EntityCfg(
#         init_state=EntityCfg.InitialStateCfg(
#             # Pose of the wrap body in scene.xml
#             pos=(-0.072, -0.018, 0.83),
#             rot=(1.0, 0.0, 0.0, 0.0),
#         ),
#         spec_fn=get_centrifuge_50ml_body_spec,
#     )


if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    print("Select centrifuge type:")
    print("1: fixed (cap + body)")
    print("2: cap")
    print("3: body")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        obj = Entity(get_centrifuge_50ml_cfg())
    elif choice == "2":
        obj = Entity(get_centrifuge_50ml_cap_cfg())
    elif choice == "3":
        obj = Entity(get_centrifuge_50ml_body_cfg())
    else:
        print(f"Invalid choice: {choice}, using default (cfg)")
        obj = Entity(get_centrifuge_50ml_cfg())

    mujoco.viewer.launch(obj.spec.compile())  
    
