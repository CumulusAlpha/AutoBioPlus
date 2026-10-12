from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assets import ASSET_ROOT


_SCENE_DIR = ASSET_ROOT / "scenes"


@dataclass(frozen=True)
class GaussianSceneCfg:
    task_id: str
    ply_path: Path
    position: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    scale: float

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError(f"{self.task_id}: position must contain 3 values")
        if len(self.quat) != 4:
            raise ValueError(f"{self.task_id}: quat must contain 4 values")
        if sum(value * value for value in self.quat) < 1e-16:
            raise ValueError(f"{self.task_id}: quat must be non-zero")
        if self.scale <= 0.0:
            raise ValueError(f"{self.task_id}: scale must be positive")


# Each task owns its complete background assignment. Repeated values are intentional:
# tasks may start from the same scan alignment and then be tuned independently.
TASK_GAUSSIAN_SCENE_CONFIGS = {
    "Shadow-Centrifuge-Insert": GaussianSceneCfg(
        task_id="Shadow-Centrifuge-Insert",
        ply_path=_SCENE_DIR / "lab1.ply",
        position=(0.0618209599, -0.220124545, 1.27405905),
        quat=(0.670184049, 9.10057282e-33, -2.02624687e-32, 0.742194948),
        scale=2.5,
    ),
    "Wuji-Centrifuge-Close": GaussianSceneCfg(
        task_id="Wuji-Centrifuge-Close",
        ply_path=_SCENE_DIR / "lab1.ply",
        position=(0.0618209599, -0.220124545, 1.27405905),
        quat=(0.670184049, 9.10057282e-33, -2.02624687e-32, 0.742194948),
        scale=2.5,
    ),
    "Shadow-Pipette-Insert": GaussianSceneCfg(
        task_id="Shadow-Pipette-Insert",
        ply_path=_SCENE_DIR / "lab2.ply",
        position=(-0.131612908, -0.183370923, 1.18346242),
        quat=(0.752664575, 1.82779682e-32, -1.41982268e-32, 0.658404159),
        scale=2.0,
    ),
    "Shadow-Pipette-Manipulate": GaussianSceneCfg(
        task_id="Shadow-Pipette-Manipulate",
        ply_path=_SCENE_DIR / "lab2.ply",
        position=(-0.131612908, -0.183370923, 1.18346242),
        quat=(0.752664575, 1.82779682e-32, -1.41982268e-32, 0.658404159),
        scale=2.0,
    ),
    "Shadow-Tube-Insert": GaussianSceneCfg(
        task_id="Shadow-Tube-Insert",
        ply_path=_SCENE_DIR / "lab3.ply",
        position=(0.145850456, -0.283748782, 1.09443533),
        quat=(0.654520196, 2.33459468e-32, -3.09449357e-32, 0.756044518),
        scale=1.5,
    ),
    "XHand-Tube-Insert": GaussianSceneCfg(
        task_id="XHand-Tube-Insert",
        ply_path=_SCENE_DIR / "lab2.ply",
        position=(-0.6543477793738188, -0.19627336739030246, 1.0959216439059145),
        quat=(
            0.748769031847845,
            3.1772814971044346e-32,
            -4.692020386128854e-32,
            0.6628310017988304,
        ),
        scale=1.5,
    ),
    "XHand-ThermalCycler-Close": GaussianSceneCfg(
        task_id="XHand-ThermalCycler-Close",
        ply_path=_SCENE_DIR / "lab4.ply",
        position=(0.160336372, -0.0289708826, 1.04302119),
        quat=(0.860333235, -1.25658643e-32, -7.44504923e-33, -0.509732013),
        scale=1.2,
    ),
    "XHand-ThermalCycler-Open": GaussianSceneCfg(
        task_id="XHand-ThermalCycler-Open",
        ply_path=_SCENE_DIR / "lab4.ply",
        position=(0.160336372, -0.0289708826, 1.04302119),
        quat=(0.860333235, -1.25658643e-32, -7.44504923e-33, -0.509732013),
        scale=1.2,
    ),
    "XHand-Bunsen-Manipulate": GaussianSceneCfg(
        task_id="XHand-Bunsen-Manipulate",
        ply_path=_SCENE_DIR / "lab5.ply",
        position=(0.00591871595, -0.300737791, 1.20741297),
        quat=(0.700115404, 1.92527162e-32, -1.58478625e-32, 0.714029706),
        scale=2.0,
    ),
    "XHand-MixerPanel-Manipulate": GaussianSceneCfg(
        task_id="XHand-MixerPanel-Manipulate",
        ply_path=_SCENE_DIR / "lab5.ply",
        position=(0.00286752126, -0.14570255, 1.01478477),
        quat=(0.700115404, 2.17285037e-32, -1.79520225e-32, 0.714029706),
        scale=1.0,
    ),
    "Wuji-HeatingPlate-Rotate": GaussianSceneCfg(
        task_id="Wuji-HeatingPlate-Rotate",
        ply_path=_SCENE_DIR / "lab6.ply",
        position=(-1.6061183832814134, 1.8933259701305867, 1.1779981011883285),
        quat=(
            0.6768610420701399,
            -2.0422840214839145e-32,
            -1.9735009154904718e-32,
            -0.7361108134837612,
        ),
        scale=1.0,
    ),
    "Wuji-Tweezers-Manipulate": GaussianSceneCfg(
        task_id="Wuji-Tweezers-Manipulate",
        ply_path=_SCENE_DIR / "lab6.ply",
        position=(-2.5019289260040294, 2.7302554807340536, 1.335921724960344),
        quat=(
            0.6869164311391185,
            -1.7915435814220474e-32,
            -1.8953978975351443e-32,
            -0.7267364148239008,
        ),
        scale=1.5,
    ),
    "Aloha-Screw-Tighten": GaussianSceneCfg(
        task_id="Aloha-Screw-Tighten",
        ply_path=_SCENE_DIR / "lab3.ply",
        position=(0.0899820493462948, -0.13352826881680824, 1.0150326137499173),
        quat=(
            0.6575272803203688,
            2.0795043221295613e-32,
            -1.086041497336576e-32,
            0.7534307371182166,
        ),
        scale=1.0,
    ),
    "Shadow-Tube-Rotate": GaussianSceneCfg(
        task_id="Shadow-Tube-Rotate",
        ply_path=_SCENE_DIR / "lab4.ply",
        position=(0.2748754397075149, -0.2114403349014624, 1.00598129),
        quat=(
            0.23589619850975588,
            -2.7903088372758175e-32,
            -1.8891184539956365e-32,
            -0.9717782584152858,
        ),
        scale=1.0,
    ),
}


def gaussian_scene_for_task(task_id: str) -> GaussianSceneCfg:
    try:
        return TASK_GAUSSIAN_SCENE_CONFIGS[task_id]
    except KeyError as exc:
        raise KeyError(f"Task {task_id!r} has no Gaussian scene assignment") from exc


def validate_gaussian_scene_registry() -> None:
    for task_id, scene in TASK_GAUSSIAN_SCENE_CONFIGS.items():
        if scene.task_id != task_id:
            raise ValueError(
                f"Gaussian scene key {task_id!r} does not match config task "
                f"{scene.task_id!r}"
            )
        if not scene.ply_path.is_file():
            raise FileNotFoundError(
                f"Gaussian scene for {task_id!r} is missing PLY: {scene.ply_path}"
            )
