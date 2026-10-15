"""Resource bounds and per-object failure isolation (GitHub #43)."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from satsim import time
from satsim.geometry import astrometric
from satsim.geometry.draw import TrackSamplingError, gen_curve_from_points, gen_line_from_endpoints
from satsim.geometry.photometric import model_to_mv
from satsim.geometry.sgp4 import create_sgp4
from satsim.image.fpa import mv_to_pe
from satsim.pipeline import _avg_t, _delta_t

sim = importlib.import_module('satsim.satsim')
START = [2026, 4, 7, 22, 0, 0.0]
TLE = (
    '1 00005U 58002B   26097.92152118  .00000210  00000-0  25961-3 0  9992',
    '2 00005  34.2431 171.1570 1836179 181.6781 177.6859 10.85968047435646',
)
BAD_TLE = (
    '1 52236U 82092BWU 22361.20122795 +.05406531 +97130-1 +90949-2 0 99998',
    '2 52236  82.5011  83.5469 0006714 106.8619 253.3401 16.09867303045483',
)
MODEL = {'mode': 'lambertian_sphere', 'albedo': 0.2, 'size': 1.0}


def run_objects(monkeypatch, obs, settings=None, tracks=None):
    """Exercise the complete object loop with controlled projection results."""
    settings = dict(settings or {})
    tracks = tracks or (([0.0], [1.0], [2.0]), ([0.0], [1.0], [2.0]))
    monkeypatch.setattr(sim, 'gen_track', lambda *a, **k: (*tracks, None, None, None))
    monkeypatch.setattr(sim, 'get_analytical_los', lambda *a, **k: (1., 2., 3.))
    monkeypatch.setattr(sim, 'get_los', lambda *a, **k: (1., 2., 3., 0., 0., None))
    ssp = {'sim': {'enable_deflection': False, 'enable_light_transit': True,
                   'enable_stellar_aberration': False, 'num_target_samples': 3,
                   'analytical_obs_frame': 'geocentric', **settings},
           'fpa': {'flip_up_down': False, 'flip_left_right': False}}
    return sim._gen_objects(
        ssp, 'none', obs, [None] * len(obs), {},
        time.utc_from_list(START), time.utc_from_list(START, 1.), 0., 1., START,
        None, None, None, None, None, None, 20., 1,
        1000, 1000, 0, 0, 0, 0, 1000, 1000, 1., 1., 1., 1., 0., 'rate')


@pytest.mark.parametrize('samples', [2, 3])
def test_huge_track_rejected_before_rasterization(monkeypatch, samples):
    def forbidden(*args, **kwargs):
        pytest.fail('Rasterization must not run before the budget check.')
    monkeypatch.setattr('satsim.geometry.draw.bezier_curve', forbidden)
    with pytest.raises(TrackSamplingError, match='max_track_samples'):
        if samples == 2:
            gen_line_from_endpoints(-1e23, 0., 1e23, 0., 1., 0., 1.)
        else:
            gen_curve_from_points(-1e23, 0., 0., 1e23, 1e23, 0., 1., 0., 1.)


def test_captured_curve_can_use_explicit_larger_budget():
    args = (46285, 307652, 5438, 13236, 570, -21477, 100., 0., 1.)
    with pytest.raises(TrackSamplingError):
        gen_curve_from_points(*args)
    rr, cc, counts, edges = gen_curve_from_points(*args, max_samples=2_000_000)
    assert len(rr) > 360_000
    assert len(rr) == len(cc) == len(counts) == len(edges) - 1
    np.testing.assert_allclose(np.sum(counts), 100.)


@pytest.mark.parametrize('curve', [False, True])
def test_legitimate_sixty_thousand_sample_track_is_preserved(curve):
    if curve:
        rr, _, counts, _ = gen_curve_from_points(0., 0., 30000., 0., 60000., 0., 100., 0., 1.)
    else:
        rr, _, counts, _ = gen_line_from_endpoints(0., 0., 60000., 0., 100., 0., 1.)
    assert len(rr) >= 60_000
    np.testing.assert_allclose(np.sum(counts), 100.)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_middle_control_is_rejected(value):
    with pytest.raises(TrackSamplingError, match='finite'):
        gen_curve_from_points(0., 0., value, 1., 2., 2., 1., 0., 1.)


@pytest.mark.parametrize('chunk_size', [1, 7, 64])
def test_chunked_photometry_matches_full_model_and_shadow(chunk_size):
    observer = astrometric.create_topocentric(20.7, -156.3, 3.)
    target = create_sgp4(*TLE)
    _, _, counts, edges = gen_curve_from_points(0., 0., 31., 2., 62., 0., 1., 0., 1.)
    t = time.utc_from_list(START, _avg_t(edges))
    expected = mv_to_pe(20., model_to_mv(observer, target, MODEL, t)) * _delta_t(edges)
    from satsim.geometry.shadow import earth_shadow_umbra_mask
    expected *= earth_shadow_umbra_mask(target, t)
    actual = sim._apply_photometry(counts, edges, observer, target, MODEL, START, 20.,
                                   chunk_size=chunk_size, earth_shadow=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_brightness_and_shadow_time_vectors_are_bounded(monkeypatch):
    seen = []

    def model(observer, target, config, t):
        seen.append(('model', len(t.tt)))
        return np.full(len(t.tt), 10.)

    def shadow(target, t):
        seen.append(('shadow', len(t.tt)))
        return np.ones(len(t.tt))
    monkeypatch.setattr(sim, 'model_to_mv', model)
    monkeypatch.setattr(sim, 'earth_shadow_umbra_mask', shadow)
    edges = np.linspace(0., 1., 1001)
    counts = sim._apply_photometry(np.zeros(1000), edges, None, object(), MODEL, START, 20.,
                                   chunk_size=64, earth_shadow=True)
    assert max(n for _, n in seen) <= 64
    assert sum(n for kind, n in seen if kind == 'model') == 1000
    np.testing.assert_allclose(counts.sum(), 10000.)


def test_explicit_brightness_series_are_sliced_with_chunks():
    edges = np.linspace(0., 1., 22)
    model = {**MODEL, 'distance': np.linspace(1000., 2000., 21),
             'phase_angle': np.linspace(20., 30., 21)}
    expected = mv_to_pe(20., model_to_mv(None, None, model, None)) * np.diff(edges)
    actual = sim._apply_photometry(np.zeros(21), edges, None, None, model, START, 20., chunk_size=4)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('stage', ['gen_track', 'model_to_mv', 'earth_shadow_umbra_mask', 'get_analytical_los'])
def test_failed_target_leaves_no_pixels_or_annotations(monkeypatch, caplog, stage):
    bad = object()
    monkeypatch.setattr(sim, 'create_satrec', lambda *a: None)
    monkeypatch.setattr(sim, 'create_sgp4_from_satrec', lambda *a: bad)
    original = {}
    # Install the stage failure after run_objects has installed its defaults.
    saved_gen = sim._gen_objects

    def gen(*args, **kwargs):
        original[stage] = getattr(sim, stage)
        original['cache'] = args[3]

        def fail(*a, **k):
            if stage == 'gen_track' or stage == 'get_analytical_los':
                target = a[6][0] if stage == 'gen_track' else a[1]
            else:
                target = a[1] if stage == 'model_to_mv' else a[0]
            if target is bad:
                raise ValueError('captured target failed at ' + stage)
            return original[stage](*a, **k)
        monkeypatch.setattr(sim, stage, fail)
        return saved_gen(*args, **kwargs)
    monkeypatch.setattr(sim, '_gen_objects', gen)
    monkeypatch.setattr(sim, 'model_to_mv', lambda o, t, m, ts: np.full(len(ts.tt), 10.))
    monkeypatch.setattr(sim, 'earth_shadow_umbra_mask', lambda t, ts: np.ones(len(ts.tt)))
    obs = [{'mode': 'tle', 'tle': BAD_TLE, 'model': MODEL, 'name': 'bad'},
           {'mode': 'line', 'origin': [.5, .5], 'velocity': [0., 0.], 'mv': 10., 'name': 'good'}]
    rr, cc, counts, entries, _ = run_objects(monkeypatch, obs, {'enable_earth_shadow': True})
    assert [entry['object_name'] for entry in entries] == ['good']
    assert len(rr) == len(cc) == len(counts) == 1
    np.testing.assert_allclose(counts, [10000.])
    assert 'Skipping target bad' in caplog.text
    assert original['cache'][0] is None


@pytest.mark.parametrize('samples', [2, 3])
def test_captured_tle_and_projection_budget_do_not_abort_other_objects(monkeypatch, samples, caplog):
    monkeypatch.setattr(sim, 'create_satrec', lambda *a: None)
    monkeypatch.setattr(sim, 'create_sgp4_from_satrec', lambda *a: object())
    obs = [{'mode': 'tle', 'tle': BAD_TLE, 'mv': 10., 'name': 'bad'},
           {'mode': 'line', 'origin': [.5, .5], 'velocity': [0., 0.], 'mv': 10., 'name': 'good'}]
    tracks = (([46285.], [5438.], [570.]), ([307652.], [13236.], [-21477.]))
    # A deliberately small explicit budget exercises both line and curve paths.
    _, _, counts, entries, _ = run_objects(
        monkeypatch, obs, {'num_target_samples': samples, 'max_track_samples': 100_000}, tracks)
    assert [entry['object_name'] for entry in entries] == ['good']
    np.testing.assert_allclose(counts, [10000.])
    assert 'max_track_samples' in caplog.text


def test_apparent_los_failure_is_not_replaced_with_geometric_los():
    def fail(target):
        raise ValueError('light-travel time failed to converge')
    observer = SimpleNamespace(at=lambda t: SimpleNamespace(observe=fail))
    with pytest.raises(ValueError, match='light-travel'):
        astrometric.get_los.__wrapped__(observer, object(), time.utc_from_list(START))


@pytest.mark.parametrize('name', ['max_track_samples', 'photometry_chunk_size'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_resource_settings_fail_explicitly(monkeypatch, name, value):
    with pytest.raises(ValueError, match=name):
        run_objects(monkeypatch, [], {name: value})


def test_captured_tle_photometry_failure_isolated_with_real_propagation(monkeypatch, caplog):
    obs = [{'mode': 'tle', 'tle': BAD_TLE, 'model': MODEL, 'name': 'bad'},
           {'mode': 'line', 'origin': [.5, .5], 'velocity': [0., 0.], 'mv': 10., 'name': 'good'}]
    _, _, counts, entries, _ = run_objects(monkeypatch, obs)
    assert [entry['object_name'] for entry in entries] == ['good']
    np.testing.assert_allclose(counts, [10000.])
    assert 'Skipping target bad' in caplog.text
