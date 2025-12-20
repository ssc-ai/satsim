import math

import numpy as np

import satsim.radar.monostatic as sensor


def test_wavelength_basic():
    f = 1.0e9  # Hz
    lam = sensor.wavelength(f)
    np.testing.assert_allclose(lam, 299_792_458.0 / f, rtol=1e-12, atol=0.0)

    # Non-positive frequency returns 0
    assert sensor.wavelength(0.0) == 0.0
    assert sensor.wavelength(-1.0) == 0.0


def test_gain_linear_basic():
    D = 1.0
    lam = 0.03
    eta = 0.6
    g = sensor.gain_linear(D, lam, eta)
    np.testing.assert_allclose(g, eta * (math.pi * D / lam) ** 2, rtol=1e-12)

    # Invalid inputs clamp to 1.0
    assert sensor.gain_linear(0.0, lam, eta) == 1.0
    assert sensor.gain_linear(D, 0.0, eta) == 1.0
    assert sensor.gain_linear(D, lam, 0.0) == 1.0


def _make_params():
    return sensor.RadarParams(
        tx_power=1.0e6,
        tx_frequency=1.0e9,
        antenna_diameter=10.0,
        efficiency=0.6,
        min_detectable_power=1.0e-13,
        snr_threshold=None,
        angle_error=0.0,
        range_error=0.0,
        range_rate_error=0.0,
        false_alarm_rate=0.0,
        az_limits=None,
        el_limits=None,
        range_limits=None,
        dwell=1.0,
        num_frames=1,
    )


def test_max_detectable_range_scaling():
    p = _make_params()

    r1 = sensor.max_detectable_range(p, sigma=1.0)
    r2 = sensor.max_detectable_range(p, sigma=16.0)
    # R ∝ sigma^(1/4)
    np.testing.assert_allclose(r2 / r1, 16.0 ** 0.25, rtol=1e-12)

    # R ∝ Pt^(1/4)
    p_power = _make_params()
    p_power.tx_power *= 16.0
    r3 = sensor.max_detectable_range(p_power, sigma=1.0)
    np.testing.assert_allclose(r3 / r1, 2.0, rtol=1e-12)

    # R ∝ 1 / sqrt(lambda) ⇒ doubling frequency halves lambda ⇒ R x sqrt(2)
    p_freq = _make_params()
    p_freq.tx_frequency *= 2.0
    r4 = sensor.max_detectable_range(p_freq, sigma=1.0)
    np.testing.assert_allclose(r4 / r1, math.sqrt(2.0), rtol=1e-12)


def test_max_detectable_range_no_threshold():
    p = _make_params()
    p.min_detectable_power = None
    assert sensor.max_detectable_range(p, sigma=1.0) is None


def test_in_fov_and_range_limits():
    p = _make_params()
    p.az_limits = (0.0, 180.0)
    p.el_limits = (10.0, 90.0)
    p.range_limits = (100.0, 2000.0)  # km

    assert sensor.in_fov(90.0, 45.0, p) is True
    assert sensor.in_fov(-10.0, 45.0, p) is False
    assert sensor.in_fov(90.0, 5.0, p) is False

    assert sensor.in_range_limits(150.0, p) is True
    assert sensor.in_range_limits(50.0, p) is False
    assert sensor.in_range_limits(3000.0, p) is False
    # No limits configured => always True
    p2 = _make_params()
    p2.range_limits = None
    assert sensor.in_range_limits(1e9, p2) is True


def test_range_rate_numeric(monkeypatch):
    # Monkeypatch time helpers to use floats for time
    monkeypatch.setattr(sensor.time, 'to_utc_list', lambda t: t)
    monkeypatch.setattr(sensor.time, 'utc_from_list', lambda t, delta_sec=0: t + delta_sec)

    # Radial velocity (m/s)
    v = 123.45

    def fake_get_los(observer, target, t, deflection=False, aberration=False, stellar_aberration=False):
        # Return range in km that grows linearly with time (seconds)
        return 0.0, 0.0, (v * t) / 1000.0, 0.0, 0.0, None

    monkeypatch.setattr(sensor, 'get_los', fake_get_los)

    rr = sensor.range_rate(None, None, t=0.0, dt=2.0)
    # Now in km/s
    np.testing.assert_allclose(rr, v / 1000.0, rtol=1e-12)


def test_doppler_shift_sign_and_value():
    p = _make_params()
    p.tx_frequency = 1.0e9  # Hz

    # Closing: negative rr (km/s) should yield positive Doppler
    rr_close = -1.0  # km/s
    df_close = sensor.doppler(p, rr_close)
    expected_close = -2.0 * (rr_close * 1000.0) / sensor._C * p.tx_frequency
    np.testing.assert_allclose(df_close, expected_close, rtol=1e-12)
    assert df_close > 0

    # Receding: positive rr -> negative Doppler
    rr_away = 1.0
    df_away = sensor.doppler(p, rr_away)
    expected_away = -2.0 * (rr_away * 1000.0) / sensor._C * p.tx_frequency
    np.testing.assert_allclose(df_away, expected_away, rtol=1e-12)
    assert df_away < 0


def test_detect_threshold_and_snr_proxy():
    p = _make_params()
    sigma = 1.0
    rmax = sensor.max_detectable_range(p, sigma)
    assert rmax is not None and rmax > 0

    # Just inside threshold
    detected, snr = sensor.detect(p, sigma, rmax * 0.9)
    assert detected is True
    assert snr is not None
    np.testing.assert_allclose(snr, (rmax / (rmax * 0.9)) ** 4, rtol=1e-12)

    # Beyond threshold
    detected2, snr2 = sensor.detect(p, sigma, rmax * 1.01)
    assert detected2 is False
    assert snr2 is None

    # No threshold configured: always detected, snr=None
    p2 = _make_params()
    p2.min_detectable_power = None
    det3, snr3 = sensor.detect(p2, sigma, 1.0)
    assert det3 is True
    assert snr3 is None


def test_detect_with_snr_threshold():
    p = _make_params()
    sigma = 1.0
    rmax = sensor.max_detectable_range(p, sigma)
    assert rmax is not None

    # Set an SNR threshold, snr = (rmax/R)^4
    p.snr_threshold = 16.0

    # Choose range that yields snr exactly threshold: (rmax / R)^4 = 16 => rmax/R = 2 => R = rmax/2
    rng_pass = rmax / 2.0
    det_ok, snr_ok = sensor.detect(p, sigma, rng_pass)
    assert det_ok is True
    assert snr_ok is not None and snr_ok >= p.snr_threshold

    # Range that yields snr below threshold but still within Rmax
    rng_fail = rmax / 1.5
    det_bad, snr_bad = sensor.detect(p, sigma, rng_fail)
    assert det_bad is False
    assert snr_bad is not None and snr_bad < p.snr_threshold
