from datetime import datetime
from pathlib import Path

import yaml

from openpi.dexbiolab_schema import DEXBIOLAB_TASK_SCHEMAS
from openpi.dexbiolab_schema import DexBioLabTaskSchema
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

OPENPI_ROOT = Path(__file__).resolve().parents[3]

with (OPENPI_ROOT / "config.yaml").open() as f:
    config = yaml.safe_load(f)

WANDB_ENABLED = bool(config["wandb_enabled"])
PROJECT_NAME = str(config.get("project_name", "dexbiolab-openpi"))
PRETRAINED_MODEL_PATH = str(config["pretrained_model_path"])
PRETRAINED_MODEL_ACTION_DIM_64_PATH = str(config.get("pretrained_model_action_dim_64_path", ""))
DATASET_ROOT = Path(config["dataset_root"])
CKPTS_ROOT = str(config["ckpts_root"])
BATCH_SIZE = int(config["batch_size"])
SINGLE_ARM_STEPS = int(config["single_arm_steps"])
DUAL_ARM_STEPS = int(config["dual_arm_steps"])
NUM_WORKERS = int(config.get("num_workers", 4))
SAVE_INTERVAL = int(config.get("save_interval", 10_000))
KEEP_PERIOD = config.get("keep_period")
if KEEP_PERIOD is not None:
    KEEP_PERIOD = int(KEEP_PERIOD)

DEFAULT_IMAGE_KEYS = {
    "front_img_name": "observation.images.front",
    "left_img_name": "observation.images.left",
    "wrist1_img_name": "observation.images.wrist1",
    "wrist2_img_name": "observation.images.wrist2",
}


def _checkpoint_path_for_action_dim(action_dim: int) -> str:
    path_by_dim = {
        32: PRETRAINED_MODEL_PATH,
        64: PRETRAINED_MODEL_ACTION_DIM_64_PATH,
    }
    path = path_by_dim.get(action_dim)
    if path:
        return path
    raise ValueError(
        f"Unsupported DexBioLab action_dim={action_dim}. "
        "Convert the pi0.5 checkpoint to that action dimension first and add it to config.yaml."
    )


def make_dexbiolab_config(cfg: DexBioLabTaskSchema):
    # Import in function to avoid the circular import with openpi.training.config.
    from .config import DataConfig
    from .config import DualArmDataConfig
    from .config import TrainConfig

    if cfg.state_dim > cfg.model_action_dim or cfg.raw_action_dim > cfg.model_action_dim:
        raise ValueError(
            f"{cfg.name} state_dim={cfg.state_dim} or raw_action_dim={cfg.raw_action_dim} "
            f"exceeds model_action_dim={cfg.model_action_dim}."
        )
    for start, stop in cfg.quaternion_slices:
        if stop - start != 4 or start < 0 or stop > cfg.raw_action_dim:
            raise ValueError(
                f"{cfg.name} has invalid wxyz quaternion slice ({start}, {stop}) "
                f"for raw_action_dim={cfg.raw_action_dim}."
            )
    for start, stop in cfg.position_slices:
        if stop - start != 3 or start < 0 or stop > cfg.raw_action_dim:
            raise ValueError(
                f"{cfg.name} has invalid Cartesian position slice ({start}, {stop}) "
                f"for raw_action_dim={cfg.raw_action_dim}."
            )

    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=cfg.model_action_dim,
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        max_token_len=cfg.max_token_len,
        image_keys=cfg.model_image_keys,
    )

    data_root = DATASET_ROOT / cfg.name

    return TrainConfig(
        name=cfg.name,
        exp_name=cfg.name + "-" + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        project_name=PROJECT_NAME,
        model=model_cfg,
        data=DualArmDataConfig(
            root=data_root,
            repo_id="local_repo",
            front_img_name=DEFAULT_IMAGE_KEYS["front_img_name"],
            left_img_name=(
                None if cfg.num_arms == 2 else DEFAULT_IMAGE_KEYS["left_img_name"]
            ),
            wrist1_img_name=DEFAULT_IMAGE_KEYS["wrist1_img_name"],
            wrist2_img_name=(
                DEFAULT_IMAGE_KEYS["wrist2_img_name"] if cfg.num_arms == 2 else None
            ),
            output_action_dim=cfg.raw_action_dim,
            quaternion_slices=cfg.quaternion_slices,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=model_cfg.get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            _checkpoint_path_for_action_dim(cfg.model_action_dim),
        ),
        num_train_steps=DUAL_ARM_STEPS if cfg.num_arms == 2 else SINGLE_ARM_STEPS,
        save_interval=SAVE_INTERVAL,
        keep_period=KEEP_PERIOD,
        wandb_enabled=WANDB_ENABLED,
        checkpoint_base_dir=CKPTS_ROOT,
    )


def get_dexbiolab_configs():
    return [make_dexbiolab_config(cfg) for cfg in DEXBIOLAB_TASK_SCHEMAS]
