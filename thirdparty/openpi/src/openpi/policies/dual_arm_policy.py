"""DexBioLab camera and action transforms for single- and dual-arm tasks."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Camera image contains non-finite values.")
        if image.size and float(np.nanmax(image)) <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DualArmInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    include_left: bool = True
    include_wrist2: bool = False

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Single-arm tasks use two fixed views and one wrist. Dual-arm tasks
        # use one fixed front view plus one wrist camera per arm.
        front_image = _parse_image(data["front"])
        wrist1_image = _parse_image(data["wrist1"])

        images = {"base_0_rgb": front_image}
        if self.include_left:
            images["base_1_rgb"] = _parse_image(data["left"])
        images["left_wrist_0_rgb"] = wrist1_image
        if self.include_wrist2:
            images["right_wrist_0_rgb"] = _parse_image(data["wrist2"])

        inputs = {
            "state": data["state"],
            "image": images,
            "image_mask": {key: np.True_ for key in images},
        }

        # Keep this for your own dataset.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Keep this for your own dataset
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DualArmOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_dim: int = 44
    quaternion_slices: tuple[tuple[int, int], ...] = ()

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # Return the unpadded action dimensions used by this adapter.
        actions = np.asarray(data["actions"][:, : self.action_dim]).copy()
        for start, stop in self.quaternion_slices:
            if stop - start != 4 or start < 0 or stop > self.action_dim:
                raise ValueError(
                    f"Invalid wxyz quaternion slice ({start}, {stop}) for action_dim={self.action_dim}."
                )
            quat = actions[:, start:stop]
            norm = np.linalg.norm(quat, axis=-1, keepdims=True)
            identity = np.zeros_like(quat)
            identity[:, 0] = 1.0
            actions[:, start:stop] = np.where(norm > 1e-8, quat / np.maximum(norm, 1e-8), identity)
        return {"actions": actions}
