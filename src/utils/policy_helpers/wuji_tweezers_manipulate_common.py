"""Shared constants, states and functions for wuji_tweezers_manipulate."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np

DEXONOMY_PHASE_KEYS = ("pregrasp_qpos", "grasp_qpos")


DEXONOMY_INTERP_STEPS = 5


BUILTIN_DEXONOMY_OBJ_POSE = (0.0, 0.0, 0.020140960812568665, 1.0, 0.0, 0.0, 0.0)


BUILTIN_DEXONOMY_PREGRASP_QPOS = """
-0.08180573582649231 0.10169616341590881 0.08674689382314682 0.5286833643913269 -0.03282772749662399 0.7031763792037964 -0.4742986261844635 0.14747650921344757 0.5426153540611267 1.4623357057571411 1.4568126201629639 0.2633645534515381 -0.2652240991592407 0.6310004591941833 0.3702129125595093 0.5163091421127319 -0.2700313627719879 0.5885187983512878 0.06365551054477692 0.8898558616638184 -0.2700349688529968 0.5787256360054016 -0.1855909526348114 1.096961259841919 -0.27002954483032227 0.4219732880592346 -0.26534923911094666
-0.08170239627361298 0.09908239543437958 0.08271459490060806 0.5272991061210632 -0.029505014419555664 0.7038437724113464 -0.4750674068927765 0.14734962582588196 0.5467917323112488 1.462578535079956 1.4568787813186646 0.27013009786605835 -0.2612806260585785 0.6597131490707397 0.3848925530910492 0.5307969450950623 -0.2701645493507385 0.6285224556922913 0.0788765475153923 0.9141515493392944 -0.2701787054538727 0.6202253103256226 -0.17308129370212555 1.1181987524032593 -0.27015987038612366 0.45644810795783997 -0.2531859278678894
-0.0821458175778389 0.09262464940547943 0.07387340813875198 0.5256265997886658 -0.026825690641999245 0.7059701681137085 -0.4739232063293457 0.14729978144168854 0.5655078291893005 1.4626216888427734 1.456905484199524 0.29177069664001465 -0.2529420256614685 0.74018794298172 0.4235842823982239 0.5741012692451477 -0.27082905173301697 0.741214394569397 0.11044636368751526 0.9761814475059509 -0.27078285813331604 0.7230005860328674 -0.147926464676857 1.1670255661010742 -0.27008455991744995 0.5335472226142883 -0.23019421100616455
-0.0834178552031517 0.08424531668424606 0.064735047519207 0.5240529179573059 -0.027811098843812943 0.7092062830924988 -0.47076696157455444 0.1470300406217575 0.6173688173294067 1.4624170064926147 1.4570951461791992 0.3343638777732849 -0.23506969213485718 0.8980520963668823 0.4912572205066681 0.6631813049316406 -0.27019035816192627 0.9183514714241028 0.11319541186094284 1.0629698038101196 -0.2721317708492279 0.8642444610595703 -0.12334509938955307 1.2447741031646729 -0.25163376331329346 0.6458719968795776 -0.204810231924057
-0.08760610967874527 0.07537833601236343 0.05693266913294792 0.52638840675354 -0.047587379813194275 0.7117598652839661 -0.4626538157463074 0.2318492829799652 0.6726650595664978 1.4638206958770752 1.4571877717971802 0.38699114322662354 -0.20033346116542816 1.1883774995803833 0.5902681946754456 0.8183251619338989 -0.2724083662033081 1.216176152229309 0.10618271678686142 1.1616941690444946 -0.14053678512573242 1.0119068622589111 -0.11952769011259079 1.3582367897033691 -0.13341279327869415 0.7921872735023499 -0.18909816443920135
-0.08960610628128052 0.07447510957717896 0.05558016896247864 0.5289040803909302 -0.05612051859498024 0.7131505012512207 -0.45664793252944946 0.3771464228630066 0.6730852723121643 1.2459392547607422 1.4572724103927612 0.40181782841682434 -0.15468306839466095 1.364096760749817 0.6525577902793884 0.8713130354881287 -0.17364093661308289 1.327007532119751 0.10428250581026077 1.2280339002609253 -0.05549604818224907 1.1074481010437012 -0.12465786188840866 1.4238998889923096 -0.07155319303274155 0.8782023787498474 -0.1792479306459427
"""


BUILTIN_DEXONOMY_GRASP_QPOS = """
-0.09023517370223999 0.07434646040201187 0.05550047382712364 0.529159665107727 -0.05775939300656319 0.7144511938095093 -0.4541071951389313 0.4498957395553589 0.6735613942146301 1.100840449333191 1.4569650888442993 0.4070376753807068 -0.1294235736131668 1.4489192962646484 0.6828171014785767 0.9036495685577393 -0.11672985553741455 1.3953311443328857 0.10388630628585815 1.2591959238052368 -0.018662508577108383 1.158259630203247 -0.11039918661117554 1.458682894706726 -0.04073943942785263 0.9253362417221069 -0.1695290356874466
"""


WUJI_SQUEEZE_QPOS = (
    0.6491445970258714,
    0.7689017287597656,
    1.410841,
    1.4672904507951736,
    0.657038,
    -0.00838884136581422,
    1.253968244037628,
    0.992817,
    1.0488570176601408,
    0.10884583451271057,
    1.6862856172618865,
    0.34431906469917295,
    1.509196,
    0.05966169791173935,
    1.3744611837959289,
    -0.29147435183429715,
    1.6236257115917205,
    0.17215611024951935,
    1.235336,
    -0.359529,
)


class WujiTweezersManipulateState(Enum):
    PREGRASP = 0
    GRASP = 1
    GRASP_HOLD = 2
    LIFT = 3
    ADJUST = 4
    ALIGN_XY = 5
    SQUEEZE = 6
    POST_ALIGN_LIFT = 7
    TRANSPORT = 8
    TRANSPORT_HOLD = 9
    RELEASE = 10


def _parse_pose_values(raw: str, expected_dim: int, name: str) -> tuple[float, ...]:
    text = raw.split("#", 1)[0].strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    values = tuple(float(value) for value in text.replace(",", " ").split())
    if len(values) != expected_dim:
        raise ValueError(
            f"{name} expects {expected_dim} values, got {len(values)}: {raw!r}"
        )
    return values


def _load_tweezers_rel_pose_from_file(
    path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rel_pos: tuple[float, ...] | None = None
    rel_quat: tuple[float, ...] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry.startswith("pos:"):
            rel_pos = _parse_pose_values(
                entry.split(":", 1)[1], expected_dim=3, name="pos"
            )
        elif entry.startswith("quat:"):
            rel_quat = _parse_pose_values(
                entry.split(":", 1)[1], expected_dim=4, name="quat"
            )

    if rel_pos is None or rel_quat is None:
        raise ValueError(
            f"Relative pose file must contain both 'pos:' and 'quat:' lines: {path}"
        )
    return rel_pos, rel_quat


def _load_wuji_joint_qpos_from_file(path: Path) -> dict[str, float]:
    joint_qpos: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry.startswith("joint:"):
            continue

        parts = entry.split(":")
        if len(parts) < 3:
            continue
        joint_name = parts[1].strip()
        joint_qpos[joint_name] = float(":".join(parts[2:]).strip())

    if not joint_qpos:
        raise ValueError(
            f"Wuji qpos file must contain 'joint:<name>: <value>' lines: {path}"
        )
    return joint_qpos


def _load_dexonomy_payload(
    path: Path, expected_dim: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dexonomy grasp file not found: {path}")

    arr = np.load(str(path), allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        payload = arr.item()
    elif isinstance(arr, dict):
        payload = arr
    else:
        raise ValueError(
            f"Unsupported Dexonomy grasp payload type: {type(arr)}. Expect dict-like data."
        )

    phases: dict[str, np.ndarray] = {}
    for key in DEXONOMY_PHASE_KEYS:
        if key not in payload:
            raise KeyError(f"Missing {key!r} in Dexonomy grasp file: {path}")
        qpos = np.asarray(payload[key], dtype=np.float32)
        if qpos.ndim == 1:
            qpos = qpos[None]
        if qpos.ndim != 2 or qpos.shape[1] != expected_dim:
            raise ValueError(
                f"{key} shape must be (N, {expected_dim}) (= 7 palm pose + {expected_dim - 7} Wuji joints), "
                f"got {qpos.shape}"
            )
        if qpos.shape[0] < 1:
            raise ValueError(f"{key} must contain at least one row: {path}")
        phases[key] = qpos.copy()

    obj_pose = np.asarray(
        payload.get("obj_pose", np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])),
        dtype=np.float32,
    )
    if obj_pose.shape != (7,):
        raise ValueError(f"obj_pose shape must be (7,), got {obj_pose.shape}")
    return phases, obj_pose.copy()


def _builtin_dexonomy_payload(
    expected_dim: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    phases = {
        "pregrasp_qpos": np.fromstring(BUILTIN_DEXONOMY_PREGRASP_QPOS, sep=" ").reshape(
            -1, expected_dim
        ),
        "grasp_qpos": np.fromstring(BUILTIN_DEXONOMY_GRASP_QPOS, sep=" ").reshape(
            -1, expected_dim
        ),
    }
    return phases, np.asarray(BUILTIN_DEXONOMY_OBJ_POSE, dtype=np.float32)


def _available_contact_count(data) -> int:
    try:
        ncon = int(data.nacon[0].detach().cpu().item())
    except Exception:
        return 0
    if ncon <= 0:
        return 0

    contact = data.contact
    capacities = []
    for field_name in ("geom", "dim", "frame", "efc_address"):
        value = getattr(contact, field_name, None)
        if hasattr(value, "shape") and value.shape:
            capacities.append(value.shape[0])
    if not capacities:
        return ncon
    return min(ncon, *capacities)
