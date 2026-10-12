"""Camera sensor output distortion matching RealSense calibration models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from mugs_mjlab.sensors.gaussian_sensor import (
    GaussianSensorData,
    GaussianSensorMjlab,
    GaussianSensorMjlabCfg,
)


@dataclass
class DistortedGaussianSensorMjlabCfg(GaussianSensorMjlabCfg):
    distortion_model: Literal["inverse_brown_conrady"] | None = None
    distortion: tuple[float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if (self.distortion_model is None) != (self.distortion is None):
            raise ValueError("distortion_model and distortion must be set together")
        if self.distortion is not None:
            if self.intrinsics is None:
                raise ValueError("intrinsics are required when distortion is configured")
            if len(self.distortion) != 5:
                raise ValueError("distortion must contain (k1, k2, p1, p2, k3)")
        super().__post_init__()

    def build(self) -> DistortedGaussianSensorMjlab:
        return DistortedGaussianSensorMjlab(self)


class DistortedGaussianSensorMjlab(GaussianSensorMjlab):
    def __init__(self, cfg: DistortedGaussianSensorMjlabCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._distortion_grid: torch.Tensor | None = None

    def _compute_data(self) -> GaussianSensorData:
        data = super()._compute_data()
        if self.cfg.distortion is None:
            return data

        data.rgb = self._apply_inverse_brown_conrady(data.rgb)
        if data.background is not None:
            data.background = self._apply_inverse_brown_conrady(data.background)
        if data.foreground is not None:
            data.foreground = self._apply_inverse_brown_conrady(data.foreground)
        if data.mask is not None:
            data.mask = self._apply_inverse_brown_conrady(data.mask)
        return data

    def _apply_inverse_brown_conrady(self, image: torch.Tensor) -> torch.Tensor:
        """Warp an ideal pinhole image to the configured RealSense raw image."""
        if self._distortion_grid is None or self._distortion_grid.device != image.device:
            assert self.cfg.intrinsics is not None
            assert self.cfg.distortion is not None
            fx, fy, cx, cy = self.cfg.intrinsics
            k1, k2, p1, p2, k3 = self.cfg.distortion
            height, width = image.shape[1:3]
            v, u = torch.meshgrid(
                torch.arange(height, device=image.device, dtype=torch.float32),
                torch.arange(width, device=image.device, dtype=torch.float32),
                indexing="ij",
            )
            x = (u - cx) / fx
            y = (v - cy) / fy
            r2 = x.square() + y.square()
            radial = 1.0 + k1 * r2 + k2 * r2.square() + k3 * r2.pow(3)
            source_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x.square())
            source_y = y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y.square())
            source_u = source_x * fx + cx
            source_v = source_y * fy + cy
            self._distortion_grid = torch.stack(
                (
                    2.0 * source_u / (width - 1) - 1.0,
                    2.0 * source_v / (height - 1) - 1.0,
                ),
                dim=-1,
            ).unsqueeze(0)

        dtype = image.dtype
        warped = F.grid_sample(
            image.permute(0, 3, 1, 2).float(),
            self._distortion_grid.expand(image.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        if dtype == torch.uint8:
            return warped.round().clamp_(0, 255).to(dtype)
        return warped.to(dtype)
