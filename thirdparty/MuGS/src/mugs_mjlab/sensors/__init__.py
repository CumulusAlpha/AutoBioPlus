"""
MuGS-MJLab Sensors

mjlab-compatible sensors for photorealistic hybrid rendering.
"""

from .gaussian_sensor import GaussianSensorMjlab, GaussianSensorMjlabCfg

__all__ = [
    "GaussianSensorMjlab",
    "GaussianSensorMjlabCfg",
]
