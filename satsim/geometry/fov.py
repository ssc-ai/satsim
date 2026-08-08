"""Shared analytical sensor field-of-view helpers."""

from __future__ import annotations

from typing import Optional, Tuple


def in_fov(
    azimuth: float,
    elevation: float,
    az_limits: Optional[Tuple[float, float]] = None,
    el_limits: Optional[Tuple[float, float]] = None,
) -> bool:
    """Return whether azimuth/elevation are inside configured degree limits."""
    if az_limits is not None:
        minimum, maximum = az_limits
        if not minimum <= azimuth <= maximum:
            return False
    if el_limits is not None:
        minimum, maximum = el_limits
        if not minimum <= elevation <= maximum:
            return False
    return True


__all__ = ['in_fov']
