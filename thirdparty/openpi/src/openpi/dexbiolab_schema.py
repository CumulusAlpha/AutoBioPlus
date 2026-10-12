"""Shared DexBioLab dataset contracts for export, training, and inference."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CAMERA_SOURCES = (
    ("front", "gs_front_cam_rgb"),
    ("left", "gs_left_cam_rgb"),
    ("wrist1", "gs_wrist_cam_rgb"),
)

SINGLE_ARM_MODEL_IMAGE_KEYS = (
    "base_0_rgb",
    "base_1_rgb",
    "left_wrist_0_rgb",
)

DUAL_ARM_MODEL_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


@dataclass(frozen=True)
class DexBioLabTaskSchema:
    name: str
    prompt_template: str
    state_dim: int
    raw_action_dim: int
    num_arms: int = 1
    model_action_dim: int = 32
    max_token_len: int = 250
    position_slices: tuple[tuple[int, int], ...] = ((0, 3),)
    quaternion_slices: tuple[tuple[int, int], ...] = ((3, 7),)
    goal_names: tuple[str, ...] = ()
    state_entity_names: tuple[str, ...] = ()
    camera_sources: tuple[tuple[str, str], ...] = DEFAULT_CAMERA_SOURCES

    @property
    def model_image_keys(self) -> tuple[str, ...]:
        if self.num_arms == 1:
            return SINGLE_ARM_MODEL_IMAGE_KEYS
        if self.num_arms == 2:
            return DUAL_ARM_MODEL_IMAGE_KEYS
        raise ValueError(f"Unsupported num_arms={self.num_arms} for {self.name}.")


DEXBIOLAB_TASK_SCHEMAS: tuple[DexBioLabTaskSchema, ...] = (
    DexBioLabTaskSchema(
        "Aloha-Screw-Tighten",
        "dual-Aloha arms screwing on centrifuge tube cap: one grasp the tube from "
        "the rack and then grips tube while the other twists cap",
        state_dim=16,
        raw_action_dim=16,
        num_arms=2,
        position_slices=((0, 3), (8, 11)),
        quaternion_slices=((3, 7), (11, 15)),
        state_entity_names=("aloha_1", "aloha_2"),
        camera_sources=(
            ("front", "gs_front_cam_rgb"),
            ("wrist1", "gs_wrist_cam_rgb"),
            ("wrist2", "gs_aloha_2_wrist_cam_rgb"),
        ),
    ),
    DexBioLabTaskSchema(
        "Shadow-Centrifuge-Insert",
        "Insert the tube into the centrifuge slot symmetrically opposite the already "
        "placed tube with the Shadow hand.",
        state_dim=28,
        raw_action_dim=25,
    ),
    DexBioLabTaskSchema(
        "Shadow-Pipette-Insert",
        "Attach the pipette tip with the Shadow hand.",
        state_dim=28,
        raw_action_dim=24,
        position_slices=(),
        quaternion_slices=(),
    ),
    DexBioLabTaskSchema(
        "Shadow-Pipette-Manipulate",
        "Pick up the pipette and move it directly above the cryovial to perform the "
        "pipetting operation.",
        state_dim=28,
        raw_action_dim=25,
    ),
    DexBioLabTaskSchema(
        "Shadow-Tube-Insert",
        "Insert the tube into rack row {target_row}, column {target_col} with the Shadow hand.",
        state_dim=28,
        raw_action_dim=25,
        goal_names=("target_row", "target_col"),
    ),
    DexBioLabTaskSchema(
        "Shadow-Tube-Rotate",
        "dual-Shadow unscrewing centrifuge tube cap: one stabilizes tube while the "
        "other twists cap",
        state_dim=56,
        raw_action_dim=50,
        num_arms=2,
        model_action_dim=64,
        max_token_len=300,
        position_slices=((0, 3), (25, 28)),
        quaternion_slices=((3, 7), (28, 32)),
        state_entity_names=("robot", "tube_holder_robot"),
        camera_sources=(
            ("front", "gs_front_cam_rgb"),
            ("wrist1", "gs_wrist_cam_rgb"),
            ("wrist2", "gs_tube_holder_wrist_cam_rgb"),
        ),
    ),
    DexBioLabTaskSchema(
        "Wuji-Centrifuge-Close",
        "Close the centrifuge lid with the Wuji hand.",
        state_dim=26,
        raw_action_dim=27,
    ),
    DexBioLabTaskSchema(
        "Wuji-HeatingPlate-Rotate",
        "Place the beaker on the heating plate, turn the knob on, then return the knob "
        "to its home position with the Wuji hand.",
        state_dim=26,
        raw_action_dim=27,
    ),
    DexBioLabTaskSchema(
        "Wuji-Tweezers-Manipulate",
        "Pick up the tweezers, move them to the cryovial, grasp it with the tweezers, "
        "and place it into the fogging liquid nitrogen flask.",
        state_dim=26,
        raw_action_dim=27,
    ),
    DexBioLabTaskSchema(
        "XHand-Bunsen-Manipulate",
        "Rotate the Bunsen burner's silver ring to turn it on, then rotate the yellow "
        "ring to increase the flame temperature.",
        state_dim=18,
        raw_action_dim=19,
    ),
    DexBioLabTaskSchema(
        "XHand-MixerPanel-Manipulate",
        "Set the mixer to {target_time_s:g} seconds, {target_temp_c:g} degrees Celsius, "
        "and {target_freq_rpm:g} RPM with the XHand.",
        state_dim=18,
        raw_action_dim=19,
        goal_names=("target_time_s", "target_temp_c", "target_freq_rpm"),
    ),
    DexBioLabTaskSchema(
        "XHand-ThermalCycler-Close",
        "Close the thermal cycler with the XHand.",
        state_dim=18,
        raw_action_dim=19,
    ),
    DexBioLabTaskSchema(
        "XHand-ThermalCycler-Open",
        "Open the thermal cycler with the XHand.",
        state_dim=18,
        raw_action_dim=19,
    ),
    DexBioLabTaskSchema(
        "XHand-Tube-Insert",
        "pick up the centrifuge tube and move it to the other rack, row {end_row}, "
        "col {end_col}",
        state_dim=18,
        raw_action_dim=19,
        goal_names=("target_row", "target_col"),
        camera_sources=(
            ("front", "gs_front_cam_rgb"),
            ("wrist1", "gs_wrist_cam_rgb"),
        ),
    ),
)

DEXBIOLAB_TASK_SCHEMA_BY_NAME = {
    schema.name: schema for schema in DEXBIOLAB_TASK_SCHEMAS
}


def _integer_goal(value: float | int, name: str) -> int:
    number = float(value)
    integer = round(number)
    if abs(number - integer) > 1e-6:
        raise ValueError(f"Goal field {name!r} must be integer-valued, got {value!r}.")
    return int(integer)


def format_task_prompt(
    schema: DexBioLabTaskSchema,
    goal_values: tuple[float | int, ...] | list[float | int] | None = None,
) -> str:
    """Format the canonical DexBioLab prompt for one episode."""
    if not schema.goal_names:
        if goal_values is not None:
            raise ValueError(f"Fixed-goal task {schema.name!r} must not provide goal values.")
        return schema.prompt_template
    if goal_values is None or len(goal_values) != len(schema.goal_names):
        count = 0 if goal_values is None else len(goal_values)
        raise ValueError(
            f"Task {schema.name!r} requires {len(schema.goal_names)} goal values, got {count}."
        )

    if schema.name in ("XHand-Tube-Insert", "Shadow-Tube-Insert"):
        row = _integer_goal(goal_values[0], schema.goal_names[0])
        col = _integer_goal(goal_values[1], schema.goal_names[1])
        if not 0 <= row < 5 or not 0 <= col < 2:
            raise ValueError(f"Rack goal is outside row [0, 4], column [0, 1]: {(row, col)}.")
        if schema.name == "XHand-Tube-Insert":
            return schema.prompt_template.format(end_row=row + 1, end_col=col + 1)
        return schema.prompt_template.format(target_row=row + 1, target_col=col + 1)

    if schema.name == "XHand-MixerPanel-Manipulate":
        time_s, temp_c, freq_rpm = (float(value) for value in goal_values)
        return schema.prompt_template.format(
            target_time_s=time_s,
            target_temp_c=temp_c,
            target_freq_rpm=freq_rpm,
        )

    raise ValueError(f"No goal prompt formatter is registered for task {schema.name!r}.")
