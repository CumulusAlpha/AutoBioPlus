import mujoco
from mjlab.entity import EntityCfg
from pathlib import Path
from src import MJCF_ROOT
CYCLER_XML: Path = (
    MJCF_ROOT
    / "instrument"
    / "thermal_cycler_biorad_c1000.xml"
)

assert CYCLER_XML.exists(), f"Missing MJCF: {CYCLER_XML}"

def get_cycler_spec() -> mujoco.MjSpec:
    """Load MJCF and embed mesh + texture assets with namespace isolation."""

    spec = mujoco.MjSpec.from_file(str(CYCLER_XML))

    mesh_root = (CYCLER_XML.parent / spec.meshdir).resolve() if spec.meshdir else CYCLER_XML.parent

    prefix = "cycler"  # namespace

    assets = {}

    # -----------------------
    # rewrite meshes
    # -----------------------
    for mesh in spec.meshes:
        if not mesh.file:
            continue

        full_path = (mesh_root / mesh.file).resolve()
        if not full_path.exists():
            print("Mesh NOT FOUND:", full_path)
            continue

        new_key = f"{prefix}/{mesh.file}"

        # load binary
        with open(full_path, "rb") as f:
            assets[new_key] = f.read()

        # rewrite reference (CRITICAL)
        mesh.file = new_key

    # -----------------------
    # rewrite textures
    # -----------------------
    for tex in spec.textures:
        if not tex.file:
            continue  # skip builtin textures

        full_path = (mesh_root / tex.file).resolve()
        if not full_path.exists():
            print("Texture NOT FOUND:", full_path)
            continue

        new_key = f"{prefix}/{tex.file}"

        with open(full_path, "rb") as f:
            assets[new_key] = f.read()

        # rewrite reference (CRITICAL)
        tex.file = new_key

    # -----------------------
    # attach assets
    # -----------------------
    spec.assets = assets

    return spec

def get_cycler_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.05, 0.824),
            rot=(1.0, 0.0, 0.0, 0),
        ),
        spec_fn=get_cycler_spec,
    )

if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity
    cycler = Entity(get_cycler_cfg())
    mujoco.viewer.launch(cycler.spec.compile())
