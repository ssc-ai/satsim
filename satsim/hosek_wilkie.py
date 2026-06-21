"""Compatibility wrapper for the Hosek-Wilkie background evaluator."""

from satsim.background.hosek_wilkie import (
    default_ground_albedo,
    default_turbidity,
    hosek_wilkie_luminance,
)


__all__ = [
    'default_ground_albedo',
    'default_turbidity',
    'hosek_wilkie_luminance',
]
