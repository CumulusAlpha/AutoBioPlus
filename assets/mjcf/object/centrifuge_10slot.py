import mujoco
from pathlib import Path

from mjlab.entity import EntityCfg

from src import MJCF_ROOT


CENTRIFUGE_10SLOT_XML: Path = (
    MJCF_ROOT
    / "object"
    / "centrifuge_10slot.gen.xml"
)

assert CENTRIFUGE_10SLOT_XML.exists(), f"Missing MJCF: {CENTRIFUGE_10SLOT_XML}"


def get_centrifuge_10slot_spec() -> mujoco.MjSpec:
    """Load MJCF and embed mesh assets."""
    spec = mujoco.MjSpec.from_file(str(CENTRIFUGE_10SLOT_XML))

    return spec


def get_centrifuge_10slot_1_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            # Position of slot1 in scene.xml
            pos=(-0.2, -0.3, 0.854),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spec_fn=get_centrifuge_10slot_spec,
    )

def get_centrifuge_10slot_2_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            # Position of slot1 in scene.xml
            pos=(0.2, -0.3, 0.854),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spec_fn=get_centrifuge_10slot_spec,
    )

if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    obj = Entity(get_centrifuge_10slot_1_cfg())

    mujoco.viewer.launch(obj.spec.compile())
    