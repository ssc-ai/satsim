"""Real FFT grid parity and its effect on rendered source coordinates."""
import numpy as np
import pytest
from scipy import ndimage, signal
import tensorflow as tf

from satsim.image.epsf import build_epsf_lut
from satsim.image.psf import gen_gaussian
from satsim.math import fftconv2p
from satsim.util import configure_eager
from satsim.satsim import (
    _epsf_lut_cache_param, _gen_psf_shape, _load_epsf_lut_cache,
    _save_epsf_lut_cache, image_generator,
)


@pytest.fixture(params=['CPU:0', 'GPU:0'], ids=['cpu', 'gpu'])
def device(request):
    configure_eager()
    if request.param.startswith('GPU') and not tf.config.get_visible_devices('GPU'):
        pytest.skip('GPU required')
    return '/' + request.param


def _inputs(shape):
    image = np.zeros(shape, np.float32)
    image[7, 8], image[shape[0] // 2, shape[1] // 2], image[-9, -9] = 0.25, 1., 2.
    yy, xx = np.indices(shape)
    psf = np.exp(-((yy - (shape[0] - 1) / 2) ** 2 + (xx - (shape[1] - 1) / 2) ** 2) / (2 * 1.25 ** 2))
    psf = (psf / psf.sum()).astype(np.float32)
    return image, psf


@pytest.mark.parametrize('shape', [(57, 57), (57, 58), (58, 57), (58, 58)])
@pytest.mark.parametrize('pad', [1, 4, 32])
@pytest.mark.parametrize('cached', [False, True], ids=['uncached', 'cached'])
def test_real_fft_matches_direct_convolution_across_parities(device, shape, pad, cached):
    image, psf = _inputs(shape)
    with tf.device(device):
        kernel = tf.constant(psf)
        if cached:
            # Reuse the same kernel tensor after populating its forward-FFT cache.
            fftconv2p(tf.constant(image), kernel, pad=pad, cache_last_y=True)
            image = np.roll(image, 1, axis=1)
        result = fftconv2p(tf.constant(image), kernel, pad=pad, cache_last_y=cached)

    expected = signal.convolve2d(image.astype(np.float64), psf.astype(np.float64), mode='same')
    assert device[1:] in result.device
    assert tuple(result.shape) == shape
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.numpy().sum(dtype=np.float64), expected.sum(), rtol=1e-5)
    assert result.numpy().min() >= -1e-6


@pytest.mark.parametrize('dtype', [tf.complex64, tf.complex128])
@pytest.mark.parametrize('shape', [(57, 57), (58, 58)])
def test_complex_fft_retains_direct_convolution_result(device, dtype, shape):
    image, psf = _inputs(shape)
    with tf.device(device):
        result = fftconv2p(tf.constant(image), tf.constant(psf), pad=4, dtype=dtype)
    expected = signal.convolve2d(image.astype(np.float64), psf.astype(np.float64), mode='same')
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize('cached', [False, True], ids=['uncached', 'cached'])
def test_real_fft_dynamic_graph_accepts_both_parities(device, cached):
    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)] * 2)
    def convolve(image, psf):
        with tf.device(device):
            return fftconv2p(image, psf, pad=4, cache_last_y=cached)

    for shape in [(57, 57), (58, 58), (58, 57)]:
        image, psf = _inputs(shape)
        with tf.device(device):
            result = convolve(tf.constant(image), tf.constant(psf))
        expected = signal.convolve2d(image.astype(np.float64), psf.astype(np.float64), mode='same')
        assert tuple(result.shape) == shape
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)
    assert convolve.experimental_get_tracing_count() == 1


def test_legacy_complex_fallback_keeps_odd_width(device, monkeypatch):
    monkeypatch.setattr(tf, '__version__', '1.12.0')
    image, psf = _inputs((57, 57))
    with tf.device(device):
        result = fftconv2p(tf.constant(image), tf.constant(psf), pad=4)
    expected = signal.convolve2d(image.astype(np.float64), psf.astype(np.float64), mode='same')
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize('support', [57, 58])
def test_epsf_base_stamp_has_expected_registration(device, support):
    with tf.device(device):
        psf = gen_gaussian(support, support, 1.)
        stamp = build_epsf_lut(psf, 1, 31)[0, 0].numpy()
    # Even optical support retains its documented half-pixel offset; the
    # detector renderer compensates for it when positioning a source.
    center = 15. if support % 2 else 15.5
    np.testing.assert_allclose(ndimage.center_of_mass(stamp), [center, center], atol=1e-4)
    assert stamp.min() >= -stamp.max() * 1e-6


def _scene(tmp_path, mode, width, osf):
    ra, dec = 279.23410832, 38.78299311
    offsets = [(-12.3, -9.6), (0.25, 0.4), (8.7, 13.2), (14.1, -15.5)]
    path = tmp_path / 'stars.csv'
    path.write_text(''.join(f'{i},10.5,{ra + dc / 41 / np.cos(np.deg2rad(dec)):.10f},{dec + dr / 41:.10f}\n'
                            for i, (dr, dc) in enumerate(offsets)))
    return {
        'version': 1,
        'sim': {'mode': mode, 'spacial_osf': osf, 'temporal_osf': 20, 'padding': 8, 'samples': 1,
                'enable_shot_noise': False, 'calculate_snr': True, 'save_ground_truth': False,
                'star_render_mode': 'transform', 'star_catalog_query_mode': 'at_start',
                'star_annotation_threshold': 20,
                'epsf': {'kernel_size': 31, 'psf_generation_size': 57, 'crop': {'mode': 'off'}}},
        'fpa': {'height': 41, 'width': width, 'y_fov': 1., 'x_fov': width / 41,
                'dark_current': 0, 'gain': 1, 'bias': 0, 'zeropoint': 23.497,
                'a2d': {'response': 'linear', 'fwc': 1e9, 'gain': 1, 'bias': 0, 'dtype': 'float32'},
                'noise': {'read': 0, 'electronic': 0}, 'psf': {'mode': 'gaussian', 'eod': 0.5},
                'time': {'exposure': 1., 'gap': 0.}, 'num_frames': 1},
        'background': {'stray': {'mode': 'none'}, 'galactic': 22},
        'geometry': {'stars': {'mode': 'csv', 'path': str(path), 'ra': ra, 'dec': dec, 'rotation': 0., 'pad': 0.,
                               'motion': {'mode': 'affine', 'rotation': 0., 'translation': [0., 0.1]}},
                     'obs': {'mode': 'list', 'list': [{'mode': 'line', 'origin': [0.25, 0.75],
                                                      'velocity': [0., 0.], 'pe': 1000.}]}}}


def _window_centroid(image, expected):
    row, col = np.rint(expected).astype(int)
    top, left = max(row - 5, 0), max(col - 5, 0)
    window = image[top:row + 6, left:col + 6]
    return np.array(ndimage.center_of_mass(window)) + [top, left]


@pytest.mark.parametrize('mode', ['fftconv2p', 'epsf'])
@pytest.mark.parametrize('width', [41, 42])
@pytest.mark.parametrize('osf', [1, 2, 3])
def test_rendered_stars_and_target_match_pixel_truth(device, tmp_path, mode, width, osf):
    config = _scene(tmp_path, mode, width, osf)
    with tf.device(device):
        frame = next(image_generator(config, output_dir=str(tmp_path), with_meta=True, num_sets=1))
    stars, target, metadata = frame[4].numpy(), frame[5].numpy(), frame[12]
    midpoint = (float(metadata['t_start']) + float(metadata['t_end'])) / 2
    epoch_positions = np.column_stack([np.asarray(metadata['rr']), np.asarray(metadata['cc'])])
    expected = (epoch_positions + np.asarray(metadata['tran']) * midpoint + 0.5) / osf - 0.5 - 8
    assert len(expected) == 4
    measured = [_window_centroid(stars, position) for position in expected]
    # At spatial OSF > 1, pixel integration of the sampled narrow Gaussian
    # contributes a small phase-dependent centroid error on either parity.
    np.testing.assert_allclose(measured, expected, atol=1e-4 if osf == 1 else 0.005)
    target_position = np.array([0.25 * 41 - 0.5, 0.75 * width - 0.5])
    np.testing.assert_allclose(_window_centroid(target, target_position), target_position,
                               atol=1e-4 if osf == 1 else 0.005)
    for image in [stars, target]:
        assert np.isfinite(image).all()
        # Allow float32 FFT cancellation noise, well below the ringing defect.
        assert image.min() >= -image.max() * 5e-6


def test_obsolete_epsf_cache_does_not_reuse_biased_lut(device, tmp_path):
    config = _scene(tmp_path, 'epsf', 41, 1)
    config['sim']['epsf']['normalize'] = True
    config['sim']['epsf']['$cache'] = str(tmp_path / 'cache')
    h, w = _gen_psf_shape(config, 57, 57, 1)
    current = _epsf_lut_cache_param(config, h, w, 1 / 41, 1 / 41, 1)
    legacy = dict(current, cache_version=1)
    obsolete = np.zeros((1, 1, 31, 31), np.float32)
    obsolete[0, 0, 15, 16] = 1.  # A unit-flux LUT with an obsolete column bias.
    _save_epsf_lut_cache(legacy, obsolete)
    with tf.device(device):
        frame = next(image_generator(config, output_dir=str(tmp_path), with_meta=True, num_sets=1))
    assert np.isfinite(frame[5].numpy()).all()
    np.testing.assert_allclose(frame[5].numpy().sum(), 1000., rtol=1e-5)
    expected = np.array([0.25 * 41 - 0.5, 0.75 * 41 - 0.5])
    np.testing.assert_allclose(_window_centroid(frame[5].numpy(), expected), expected, atol=1e-4)
    regenerated = _load_epsf_lut_cache(current)
    assert regenerated is not None and not np.array_equal(regenerated, obsolete)
