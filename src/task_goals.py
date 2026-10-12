"""Pure language-goal contracts shared with isolated model environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


TASK_PROMPT_TEMPLATES: dict[str, str] = {
    "Aloha-Screw-Tighten": (
        "dual-Aloha arms screwing on centrifuge tube cap: one grasp the tube from "
        "the rack and then grips tube while the other twists cap"
    ),
    "Shadow-Centrifuge-Insert": (
        "Insert the tube into the centrifuge slot symmetrically opposite the already "
        "placed tube with the Shadow hand."
    ),
    "Shadow-Pipette-Insert": "Attach the pipette tip with the Shadow hand.",
    "Shadow-Pipette-Manipulate": (
        "Pick up the pipette and move it directly above the cryovial to perform the "
        "pipetting operation."
    ),
    "Shadow-Tube-Insert": (
        "Insert the tube into rack row {target_row}, column {target_col} with the Shadow hand."
    ),
    "Shadow-Tube-Rotate": (
        "dual-Shadow unscrewing centrifuge tube cap: one stabilizes tube while the "
        "other twists cap"
    ),
    "Wuji-Centrifuge-Close": "Close the centrifuge lid with the Wuji hand.",
    "Wuji-HeatingPlate-Rotate": (
        "Place the beaker on the heating plate, turn the knob on, then return the knob "
        "to its home position with the Wuji hand."
    ),
    "Wuji-Tweezers-Manipulate": (
        "Pick up the tweezers, move them to the cryovial, grasp it with the tweezers, "
        "and place it into the fogging liquid nitrogen flask."
    ),
    "XHand-Bunsen-Manipulate": (
        "Rotate the Bunsen burner's silver ring to turn it on, then rotate the yellow "
        "ring to increase the flame temperature."
    ),
    "XHand-MixerPanel-Manipulate": (
        "Set the mixer to {target_time_s:g} seconds, {target_temp_c:g} degrees Celsius, "
        "and {target_freq_rpm:g} RPM with the XHand."
    ),
    "XHand-ThermalCycler-Close": "Close the thermal cycler with the XHand.",
    "XHand-ThermalCycler-Open": "Open the thermal cycler with the XHand.",
    "XHand-Tube-Insert": (
        "pick up the centrifuge tube and move it to the other rack, row {end_row}, "
        "col {end_col}"
    ),
}

TASK_GOAL_NAMES: dict[str, tuple[str, ...]] = {
    "XHand-Tube-Insert": ("target_row", "target_col"),
    "Shadow-Tube-Insert": ("target_row", "target_col"),
    "XHand-MixerPanel-Manipulate": (
        "target_time_s",
        "target_temp_c",
        "target_freq_rpm",
    ),
}

# Raw episode goals are converted to [0, 1] before they are supplied to learned
# policies.  Keep this contract beside the prompt/goal definitions so dataset
# conversion and online evaluation cannot silently use different scales.
TASK_GOAL_NORMALIZATION_BOUNDS: dict[str, tuple[tuple[float, float], ...]] = {
    "XHand-Tube-Insert": ((0.0, 4.0), (0.0, 1.0)),
    "Shadow-Tube-Insert": ((0.0, 4.0), (0.0, 1.0)),
    "XHand-MixerPanel-Manipulate": (
        (10.0, 110.0),
        (10.0, 40.0),
        (300.0, 1500.0),
    ),
}

MIXER_TIME_VALUES = tuple(float(value) for value in (*range(10, 60, 5), *range(65, 115, 5)))
MIXER_TEMP_VALUES = tuple(float(value) for value in (*range(10, 25), *range(26, 41)))
MIXER_FREQ_VALUES = tuple(float(value) for value in (*range(300, 900, 50), *range(950, 1550, 50)))


def _mixer_balanced_variants() -> tuple[tuple[float, float, float], ...]:
    """Build a deterministic 100-point design with near-uniform axis marginals."""
    return tuple(
        (
            MIXER_TIME_VALUES[(index * 7) % len(MIXER_TIME_VALUES)],
            MIXER_TEMP_VALUES[(index * 11 + 7) % len(MIXER_TEMP_VALUES)],
            MIXER_FREQ_VALUES[(index * 13 + 13) % len(MIXER_FREQ_VALUES)],
        )
        for index in range(100)
    )


TASK_GOAL_VARIANTS: dict[str, tuple[tuple[float, ...], ...]] = {
    "XHand-Tube-Insert": tuple(
        (float(row), float(col))
        for row in range(5)
        for col in range(2)
    ),
    "Shadow-Tube-Insert": tuple(
        (float(row), float(col))
        for row in range(5)
        for col in range(2)
    ),
    "XHand-MixerPanel-Manipulate": _mixer_balanced_variants(),
}

TASK_GOAL_BALANCE_MODES = {
    "XHand-Tube-Insert": "joint",
    "Shadow-Tube-Insert": "joint",
    "XHand-MixerPanel-Manipulate": "marginal",
}

TASK_GOAL_DEFAULT_EPISODES_PER_VARIANT = {
    "XHand-Tube-Insert": 10,
    "Shadow-Tube-Insert": 10,
    "XHand-MixerPanel-Manipulate": 1,
}


def task_goal_names(task_id: str) -> tuple[str, ...] | None:
    """Return the ordered episode-goal fields for a variable-goal task."""
    return TASK_GOAL_NAMES.get(task_id)


def task_goal_values_from_prompt(task_id: str, prompt: str) -> tuple[float, ...] | None:
    """Recover a canonical structured goal from one episode prompt."""
    variants = task_goal_variants(task_id)
    if variants is None:
        if prompt.strip() != format_task_prompt(task_id):
            raise ValueError(f"Prompt is not canonical for {task_id!r}: {prompt!r}.")
        return None
    matches = [values for values in variants if format_task_prompt(task_id, values) == prompt.strip()]
    if len(matches) != 1:
        raise ValueError(
            f"Prompt must identify exactly one canonical goal for {task_id!r}, "
            f"got {len(matches)} matches: {prompt!r}."
        )
    return matches[0]


def normalize_task_goal_values(
    task_id: str, values: Sequence[float | int]
) -> tuple[float, ...]:
    """Normalize one raw structured goal to the learned-policy [0, 1] contract."""
    bounds = TASK_GOAL_NORMALIZATION_BOUNDS.get(task_id)
    if bounds is None:
        raise ValueError(f"Task {task_id!r} has no variable goal normalization contract.")
    if len(values) != len(bounds):
        raise ValueError(f"Goal for {task_id!r} has {len(values)} values, expected {len(bounds)}.")
    normalized: list[float] = []
    for name, value, (lower, upper) in zip(TASK_GOAL_NAMES[task_id], values, bounds, strict=True):
        numeric = float(value)
        if numeric < lower or numeric > upper:
            raise ValueError(f"Goal field {name!r}={numeric} is outside [{lower}, {upper}].")
        normalized.append((numeric - lower) / (upper - lower))
    return tuple(normalized)


def task_goal_variants(task_id: str) -> tuple[tuple[float, ...], ...] | None:
    """Return the canonical finite collection catalog for a variable-goal task."""
    return TASK_GOAL_VARIANTS.get(task_id)


def task_goal_balance_mode(task_id: str) -> str | None:
    """Return whether the catalog balances complete tuples or individual axes."""
    return TASK_GOAL_BALANCE_MODES.get(task_id)


def default_goal_episodes_per_variant(task_id: str) -> int | None:
    """Return the standard successful-demonstration quota for each catalog entry."""
    return TASK_GOAL_DEFAULT_EPISODES_PER_VARIANT.get(task_id)


def expected_collection_episodes(task_id: str, fixed_task_episodes: int = 100) -> int:
    """Return the standard automation target for one task."""
    variants = task_goal_variants(task_id)
    if variants is None:
        return int(fixed_task_episodes)
    quota = default_goal_episodes_per_variant(task_id)
    if quota is None:
        raise ValueError(f"No default balanced collection quota is registered for {task_id!r}.")
    return len(variants) * quota


def goal_variant_key(values: Sequence[float | int]) -> tuple[float, ...]:
    """Normalize one stored or sampled goal tuple for counting."""
    return tuple(float(value) for value in values)


def validate_balanced_goal_metadata(
    task_id: str,
    goal_values: Sequence[Sequence[float | int]],
    attrs: Any,
    *,
    expected_episodes_per_variant: int | None = None,
) -> dict[tuple[float, ...], int]:
    """Validate exact catalog coverage and return the per-variant histogram."""
    expected_catalog, quota = validate_goal_balance_contract(
        task_id,
        attrs,
        expected_episodes_per_variant=expected_episodes_per_variant,
    )

    counts = {variant: 0 for variant in expected_catalog}
    for values in goal_values:
        key = goal_variant_key(values)
        if key not in counts:
            raise ValueError(f"Collection contains an unknown goal variant: {key}.")
        counts[key] += 1
    wrong = {key: count for key, count in counts.items() if count != quota}
    if wrong:
        preview = list(wrong.items())[:5]
        raise ValueError(
            f"Collection is not balanced at {quota} episode(s) per variant; "
            f"first mismatches: {preview}."
        )
    return counts


def validate_goal_balance_contract(
    task_id: str,
    attrs: Any,
    *,
    expected_episodes_per_variant: int | None = None,
) -> tuple[tuple[tuple[float, ...], ...], int]:
    """Validate catalog and quota metadata without loading episode rows."""
    variants = task_goal_variants(task_id)
    if variants is None:
        raise ValueError(f"Fixed-goal task {task_id!r} has no balanced goal catalog.")

    expected_catalog = tuple(goal_variant_key(values) for values in variants)
    actual_catalog = tuple(
        goal_variant_key(values)
        for values in attrs.get("goal_variant_catalog", ())
    )
    if actual_catalog != expected_catalog:
        raise ValueError(f"Goal variant catalog does not match {task_id!r}.")
    if not bool(attrs.get("goal_balanced", False)):
        raise ValueError(f"Collection for {task_id!r} is not marked goal-balanced.")
    expected_mode = task_goal_balance_mode(task_id)
    if attrs.get("goal_balance_mode") != expected_mode:
        raise ValueError(
            f"Goal balance mode is {attrs.get('goal_balance_mode')!r}, "
            f"expected {expected_mode!r}."
        )

    quota = int(attrs.get("goal_episodes_per_variant", 0))
    if quota < 1:
        raise ValueError("goal_episodes_per_variant must be positive.")
    if expected_episodes_per_variant is not None and quota != expected_episodes_per_variant:
        raise ValueError(
            f"Goal quota is {quota}, expected {expected_episodes_per_variant}."
        )
    if int(attrs.get("goal_variant_count", -1)) != len(expected_catalog):
        raise ValueError("goal_variant_count does not match the canonical catalog.")
    expected_total = len(expected_catalog) * quota
    if int(attrs.get("goal_balance_target_episodes", -1)) != expected_total:
        raise ValueError("goal_balance_target_episodes does not match catalog * quota.")
    return expected_catalog, quota


def goal_balance_provenance(attrs: Any) -> dict[str, Any] | None:
    """Copy the balanced collection contract into a derived dataset."""
    if not bool(attrs.get("goal_balanced", False)):
        return None
    keys = (
        "goal_balanced",
        "goal_balance_mode",
        "goal_variant_count",
        "goal_episodes_per_variant",
        "goal_balance_target_episodes",
        "goal_balance_seed",
        "goal_variant_catalog",
    )
    return {key: attrs[key] for key in keys}


def validate_goal_balance_provenance(
    task_id: str,
    provenance: Any,
    *,
    expected_episodes_per_variant: int | None = None,
    total_episodes: int | None = None,
) -> dict[str, Any] | None:
    """Validate the balance contract copied into a derived dataset."""
    if not isinstance(provenance, Mapping):
        raise ValueError("DexBioLab dataset provenance must be a mapping.")
    balance = provenance.get("goal_balance")
    variants = task_goal_variants(task_id)
    if variants is None:
        if balance is not None:
            raise ValueError(
                f"Fixed-goal task {task_id!r} must not carry goal-balance metadata."
            )
        return None
    if not isinstance(balance, Mapping):
        raise ValueError(f"Variable-goal task {task_id!r} is missing goal-balance metadata.")

    _, quota = validate_goal_balance_contract(
        task_id,
        balance,
        expected_episodes_per_variant=expected_episodes_per_variant,
    )
    target = len(variants) * quota
    if total_episodes is not None and int(total_episodes) != target:
        raise ValueError(
            f"Dataset for {task_id!r} has {total_episodes} episodes, "
            f"but its goal-balance contract requires {target}."
        )
    return dict(balance)


def _integer_goal(value: float | int, name: str) -> int:
    number = float(value)
    integer = round(number)
    if abs(number - integer) > 1e-6:
        raise ValueError(f"Goal field {name!r} must be integer-valued, got {value!r}.")
    return int(integer)


def format_task_prompt(task_id: str, goal_values: Sequence[float | int] | None = None) -> str:
    """Format one training/inference prompt from the canonical episode goal."""
    try:
        template = TASK_PROMPT_TEMPLATES[task_id]
    except KeyError as exc:
        raise ValueError(f"No language prompt template is registered for task {task_id!r}.") from exc

    names = task_goal_names(task_id)
    if names is None:
        if goal_values is not None:
            raise ValueError(f"Fixed-goal task {task_id!r} must not provide goal values.")
        return template
    if goal_values is None:
        raise ValueError(f"Variable-goal task {task_id!r} requires {names}.")
    values = list(goal_values)
    if len(values) != len(names):
        raise ValueError(f"Task {task_id!r} requires {len(names)} goal values, got {len(values)}.")

    if task_id in ("XHand-Tube-Insert", "Shadow-Tube-Insert"):
        row = _integer_goal(values[0], names[0])
        col = _integer_goal(values[1], names[1])
        if not 0 <= row < 5 or not 0 <= col < 2:
            raise ValueError(
                f"Rack goal must satisfy row in [0, 4], col in [0, 1], got {(row, col)}."
            )
        if task_id == "XHand-Tube-Insert":
            return template.format(end_row=row + 1, end_col=col + 1)
        return template.format(target_row=row + 1, target_col=col + 1)

    if task_id == "XHand-MixerPanel-Manipulate":
        time_s, temp_c, freq_rpm = (float(value) for value in values)
        return template.format(
            target_time_s=time_s,
            target_temp_c=temp_c,
            target_freq_rpm=freq_rpm,
        )

    raise ValueError(f"No goal prompt formatter is registered for task {task_id!r}.")
