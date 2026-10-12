import mujoco
from mjlab.entity import EntityCfg
from pathlib import Path
from src import MJCF_ROOT

TABLE_XML: Path = (
    MJCF_ROOT
    / "misc"
    / "simple_table.xml"
)
assert TABLE_XML.exists(), f"Missing MJCF: {TABLE_XML}"

def get_table_spec() -> mujoco.MjSpec:
    """Load MJCF and embed mesh assets."""
    spec = mujoco.MjSpec.from_file(str(TABLE_XML))
    return spec

def get_table_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spec_fn=get_table_spec,
    )

if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity
    table = Entity(get_table_cfg())
    mujoco.viewer.launch(table.spec.compile())
