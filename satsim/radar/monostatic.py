from __future__ import annotations

"""Radar sensor physics helpers for analytical simulations.

This module centralizes simple monostatic radar utilities used by the radar
simulator. Naming avoids unit suffixes per SatSim conventions — docstrings
state expected units explicitly.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any

from satsim import time
from satsim.geometry.astrometric import get_los


_C = 299_792_458.0  # m/s


@dataclass
class RadarParams:
    """Parameters for a (mono-static) radar sensor.

    Units and conventions:
    - tx_power: transmit power [W]
    - tx_frequency: carrier frequency [Hz]
    - antenna_diameter: effective antenna diameter [m]
    - efficiency: antenna efficiency [0–1]
    - min_detectable_power: receiver minimum detectable power [W] (optional)
    - snr_threshold: optional SNR threshold (dimensionless) for downstream use
    - angle_error: 1-sigma angle measurement error [deg]
    - range_error: 1-sigma range error [km]
    - range_rate_error: 1-sigma range-rate error [km/s]
    - false_alarm_rate: probability per dwell (not used here yet)
    - az_limits: azimuth FOV limits [deg] (min, max)
    - el_limits: elevation FOV limits [deg] (min, max)
    - range_limits: detection range bounds [km] (min, max)
    - dwell: dwell time per frame [s]
    - num_frames: number of frames to simulate
    - sensor_id: optional sensor identifier string
    """
    tx_power: float
    tx_frequency: float
    antenna_diameter: float
    efficiency: float
    min_detectable_power: Optional[float] = None
    snr_threshold: Optional[float] = None
    angle_error: float = 0.0
    range_error: float = 0.0
    range_rate_error: float = 0.0
    false_alarm_rate: float = 0.0
    az_limits: Optional[Tuple[float, float]] = None
    el_limits: Optional[Tuple[float, float]] = None
    range_limits: Optional[Tuple[float, float]] = None
    dwell: float = 1.0
    num_frames: int = 1
    sensor_id: Optional[str] = None


def wavelength(tx_frequency: float) -> float:
    """Return RF wavelength for a given frequency.

    Args:
        tx_frequency: frequency [Hz]

    Returns:
        Wavelength [m]. Returns 0.0 if input is non-positive.
    """
    return _C / tx_frequency if tx_frequency > 0 else 0.0


def gain_linear(diameter: float, wave_length: float, efficiency: float) -> float:
    """Approximate aperture antenna gain in linear units.

    Uses G = eta * (pi*D/lambda)^2.

    Args:
        diameter: effective diameter [m]
        wave_length: wavelength [m]
        efficiency: aperture efficiency [0–1]

    Returns:
        Dimensionless gain (linear). Returns 1.0 if inputs are invalid.
    """
    if diameter <= 0 or wave_length <= 0 or efficiency <= 0:
        return 1.0
    return efficiency * (math.pi * diameter / wave_length) ** 2


def max_detectable_range(p: RadarParams, sigma: float) -> Optional[float]:
    """Maximum detection range from the radar equation.

    Solves for R where received power equals the minimum detectable power:
    Rmax = ((Pt * G^2 * lambda^2 * sigma) / ((4*pi)^3 * Smin))^(1/4)

    Args:
        p: radar parameters
        sigma: target radar cross section [m^2]

    Returns:
        Maximum range [km] if ``min_detectable_power`` is provided; otherwise
        ``None``.
    """
    if p.min_detectable_power is None:
        return None
    wave_length = wavelength(p.tx_frequency)
    gain = gain_linear(p.antenna_diameter, wave_length, p.efficiency)
    numerator = p.tx_power * (gain ** 2) * (wave_length ** 2) * sigma
    denominator = ((4.0 * math.pi) ** 3) * p.min_detectable_power
    range_val = (numerator / denominator) ** 0.25
    return range_val / 1000.0


def in_fov(az: float, el: float, p: RadarParams) -> bool:
    """Check if (az, el) fall within configured FOV limits.

    Args:
        az: azimuth [deg]
        el: elevation [deg]
        p: radar parameters

    Returns:
        True if both azimuth and elevation are within limits (or if limits are
        not configured).
    """
    if p.az_limits is not None:
        mn, mx = p.az_limits
        if not (mn <= az <= mx):
            return False
    if p.el_limits is not None:
        mn, mx = p.el_limits
        if not (mn <= el <= mx):
            return False
    return True


def in_range_limits(rng: float, p: RadarParams) -> bool:
    """Check if range falls within configured limits.

    Args:
        rng: range [km]
        p: radar parameters

    Returns:
        True if within limits (or if limits are not configured).
    """
    if p.range_limits is None:
        return True
    mn, mx = p.range_limits
    return (rng >= mn) and (rng <= mx)


def range_rate(observer: Any, target: Any, t, dt: float = 1.0) -> float:
    """Estimate range-rate via finite difference over a short interval.

    Args:
        observer: observing site/body (Skyfield object)
        target: target object (Skyfield object)
        t: epoch (Skyfield time or compatible)
        dt: differencing interval [s]

    Returns:
        Range-rate [km/s].
    """
    offset_time = time.utc_from_list(time.to_utc_list(t), delta_sec=dt)
    _, _, range_start, _, _, _ = get_los(observer, target, t, deflection=False, aberration=False, stellar_aberration=False)
    _, _, range_end, _, _, _ = get_los(observer, target, offset_time, deflection=False, aberration=False, stellar_aberration=False)
    return (range_end - range_start) / dt


def detect(p: RadarParams, rcs: float, range_value: float) -> Tuple[bool, Optional[float]]:
    """Binary detection test using a power threshold, with optional SNR gating.

    Behavior:
    - If ``min_detectable_power`` is provided, an Rmax is computed and detection
      requires ``range_value <= Rmax``.
    - A simple SNR proxy is computed as ``(Rmax / range_value)^4`` when Rmax is
      available. If ``p.snr_threshold`` is provided, detection additionally
      requires ``snr_proxy >= p.snr_threshold``. If Rmax is not available, the
      SNR proxy is ``None`` and the SNR threshold is not enforced.

    Args:
        p: radar parameters
        rcs: radar cross section [m^2]
        range_value: range [km]

    Returns:
        Tuple of (detected, snr_proxy). ``snr_proxy`` is dimensionless and
        follows a simple (Rmax / R)^4 scaling when a power threshold is
        available; otherwise ``None``.
    """
    rmax = max_detectable_range(p, rcs)
    snr = None
    if rmax is not None:
        # Range must be within Rmax
        if range_value > rmax:
            return False, None
        # Compute SNR proxy and optionally enforce snr_threshold
        snr = (rmax / max(range_value, 1e-6)) ** 4
        if p.snr_threshold is not None and snr < p.snr_threshold:
            return False, snr
    return True, snr


def doppler(p: RadarParams, rr: float) -> float:
    """Compute monostatic Doppler frequency shift from range-rate.

    Sign convention: positive for closing (decreasing range), negative for
    receding (increasing range).

    Args:
        p: radar parameters (uses ``tx_frequency``)
        rr: range-rate [km/s]

    Returns:
        Doppler frequency shift [Hz].
    """
    # Convert rr from km/s to m/s, apply monostatic factor of 2 and sign.
    return -2.0 * (rr * 1000.0) / _C * p.tx_frequency


def range_unc(p: RadarParams) -> float:
    """Return range 1-sigma uncertainty [km] from parameters."""
    return abs(p.range_error)


def doppler_unc(p: RadarParams) -> float:
    """Return Doppler 1-sigma uncertainty [Hz] from range-rate uncertainty.

    Uses the monostatic mapping with standard deviation of range-rate.
    """
    return abs(-2.0 * (p.range_rate_error * 1000.0) / _C * p.tx_frequency)


__all__ = [
    'RadarParams',
    'wavelength',
    'gain_linear',
    'max_detectable_range',
    'in_fov',
    'in_range_limits',
    'range_rate',
    'detect',
    'doppler',
    'range_unc',
    'doppler_unc',
]
