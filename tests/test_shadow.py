import numpy as np

from astropy import units as u

from satsim.geometry.astrometric import load_earth, load_sun
from satsim.geometry.twobody import create_twobody
from satsim.geometry.shadow import earth_shadow_umbra_mask
from satsim import time


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_earth_shadow_umbra_mask_anti_sun():
    # Choose a fixed time
    t = time.utc(2021, 3, 18, 0, 0, 0)

    earth = load_earth()
    sun = load_sun()

    # Direction from Earth to Sun
    s_vec = (sun - earth).at(t).position.km
    s_hat = _unit(s_vec)

    # Place target on the anti-sun line near Earth (LEO altitude)
    alt_km = 700.0
    r_mag = 6378.137 + alt_km
    r0 = (-s_hat) * r_mag * u.km
    v0 = np.array([0.0, 0.0, 0.0]) * u.km / u.s

    target = create_twobody(r0, v0, t)

    mask = earth_shadow_umbra_mask(target, t)

    # Expect umbra (mask == 0)
    assert mask.shape[0] == 1
    assert mask[0] == 0.0


def test_earth_shadow_umbra_mask_sunward():
    # Choose a fixed time
    t = time.utc(2021, 3, 18, 0, 0, 0)

    earth = load_earth()
    sun = load_sun()

    # Direction from Earth to Sun
    s_vec = (sun - earth).at(t).position.km
    s_hat = _unit(s_vec)

    # Place target on the sunward line near Earth (LEO altitude)
    alt_km = 700.0
    r_mag = 6378.137 + alt_km
    r0 = (s_hat) * r_mag * u.km
    v0 = np.array([0.0, 0.0, 0.0]) * u.km / u.s

    target = create_twobody(r0, v0, t)

    mask = earth_shadow_umbra_mask(target, t)

    # Expect lit (mask == 1)
    assert mask.shape[0] == 1
    assert mask[0] == 1.0


def test_earth_shadow_umbra_mask_vectorized_time():
    # Vectorized time inputs should return a vector mask
    t = time.utc(2021, 3, 18, 0, 0, [0, 0, 0])

    earth = load_earth()
    sun = load_sun()

    s_vec = (sun - earth).at(t).position.km[:, 0]
    s_hat = _unit(s_vec)

    alt_km = 700.0
    r_mag = 6378.137 + alt_km
    r0 = (-s_hat) * r_mag * u.km
    v0 = np.array([0.0, 0.0, 0.0]) * u.km / u.s

    # Epoch at the first time entry (equivalent to all three here)
    t0 = time.utc(2021, 3, 18, 0, 0, 0)
    target = create_twobody(r0, v0, t0)

    mask = earth_shadow_umbra_mask(target, t)

    assert mask.shape[0] == 3
    # All three are anti-sun, so all umbra
    assert np.all(mask == 0.0)
