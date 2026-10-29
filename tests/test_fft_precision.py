"""FFT accuracy, ADC background stability and cache isolation across runtimes."""
import numpy as np
import pytest
from scipy import signal
import tensorflow as tf

from satsim.math.fft import fftconv2, fftconv2p, _fft_compute_dtype
from satsim.util import configure_eager, get_semantic_version


@pytest.fixture(params=['CPU:0', 'GPU:0'], ids=['cpu', 'gpu'])
def device(request):
    configure_eager()
    if request.param.startswith('GPU') and not tf.config.get_visible_devices('GPU'):
        pytest.skip('GPU required')
    return '/' + request.param


def _scene(shape):
    image = np.zeros(shape, np.float32)
    image[shape[0] // 2, shape[1] // 2] = 1000
    yy, xx = np.indices(shape, dtype=np.float64)
    psf = np.exp(-((yy - (shape[0] - 1) / 2)**2 + (xx - (shape[1] - 1) / 2)**2) / (2 * 4.5**2))
    return image, (psf / psf.sum()).astype(np.float32)


@pytest.mark.parametrize('shape', [(191, 192), (192, 192), (520, 520)])
@pytest.mark.parametrize('dtype', [tf.float32, tf.complex64, tf.float64, tf.complex128])
@pytest.mark.parametrize('cached', [False, True], ids=['uncached', 'cached'])
def test_non_power_of_two_fft_preserves_photometry(device, shape, dtype, cached):
    image, psf = _scene(shape)
    with tf.device(device):
        kernel = tf.constant(psf)
        if cached:
            fftconv2p(tf.constant(image), kernel, pad=1, dtype=dtype)
        actual = fftconv2p(tf.constant(image), kernel, pad=1, dtype=dtype, cache_last_y=cached)
    expected = signal.fftconvolve(image.astype(np.float64), psf.astype(np.float64), mode='same')
    assert actual.dtype == dtype.real_dtype
    assert device[1:] in actual.device
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(actual.numpy().sum(dtype=np.float64), expected.sum(), rtol=1e-6)


@pytest.mark.parametrize('dtype', [tf.complex64, tf.complex128])
def test_linear_fft_preserves_photometry(device, dtype):
    image, psf = _scene((192, 192))
    with tf.device(device):
        actual = fftconv2(tf.constant(image), tf.constant(psf), dtype=dtype)
    expected = signal.fftconvolve(image.astype(np.float64), psf.astype(np.float64), mode='same')
    assert actual.dtype == dtype.real_dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=2e-5)
    # Match the renderer's 1e-5 flux contract. Native GPU complex64 FFTs at
    # prime transform lengths can exceed a 1e-6 relative bound.
    np.testing.assert_allclose(actual.numpy().sum(dtype=np.float64), expected.sum(), rtol=1e-5)


def test_fft_preserves_legitimate_negative_values(device):
    image, psf = _scene((192, 192))
    image[96, 96] = -1000
    with tf.device(device):
        actual = fftconv2p(tf.constant(image), tf.constant(psf), pad=1)
    expected = signal.fftconvolve(image.astype(np.float64), psf.astype(np.float64), mode='same')
    assert float(tf.reduce_min(actual)) < -1
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=2e-5)


def test_fft_cached_kernel_keeps_requested_input_precision(device):
    image, psf = _scene((57, 58))
    rng = np.random.default_rng(73)
    psf = psf.astype(np.float64) + rng.normal(0, 1e-10, psf.shape)
    with tf.device(device):
        kernel = tf.constant(psf)
        for dtype in [tf.float64, tf.float32, tf.float64, tf.complex64]:
            actual = fftconv2p(tf.constant(image), kernel, pad=4, dtype=dtype, cache_last_y=True)
            expected = fftconv2p(tf.constant(image), kernel, pad=4, dtype=dtype, cache_last_y=False)
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_fft_cached_eager_kernel_can_be_captured_by_distinct_graphs(device):
    image, psf = _scene((57, 58))
    with tf.device(device):
        kernel = tf.constant(psf)

    def build():
        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
        def convolve(values):
            with tf.device(device):
                return fftconv2p(values, kernel, pad=4)
        return convolve

    first, second = build(), build()
    expected = signal.fftconvolve(image.astype(np.float64), psf.astype(np.float64), mode='same')
    for function in [first, second, first]:
        actual = function(tf.constant(image))
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=2e-5)


def test_fft_cache_does_not_mix_cpu_gpu_spectra():
    if not tf.config.get_visible_devices('GPU'):
        pytest.skip('GPU required')
    configure_eager()
    image, psf = _scene((192, 192))
    with tf.device('/CPU:0'):
        kernel = tf.constant(psf)
    for device in ['/CPU:0', '/GPU:0', '/CPU:0']:
        with tf.device(device):
            actual = fftconv2p(tf.constant(image), kernel, pad=1)
            expected = fftconv2p(tf.constant(image), kernel, pad=1, cache_last_y=False)
        assert device[1:] in actual.device
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_unaffected_fft_keeps_native_single_precision_result(device):
    if device == '/CPU:0' and get_semantic_version(tf) < [2, 15]:
        pytest.skip('The legacy CPU calculation intentionally changes')
    image, psf = _scene((192, 192))
    with tf.device(device):
        values, kernel = tf.constant(image), tf.constant(psf)
        padded = tf.pad(values, [[1, 1], [1, 1]])
        spectrum = tf.signal.rfft2d(padded) * tf.signal.rfft2d(tf.pad(kernel, [[1, 1], [1, 1]]))
        expected = tf.signal.irfft2d(spectrum, fft_length=[194, 194])
        expected = tf.roll(expected, [97, 97], [0, 1])[:192, :192]
        actual = fftconv2p(values, kernel, pad=1)
    assert actual.dtype == tf.float32
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize('graph', [False, True], ids=['eager', 'graph'])
def test_fft_gradient_remains_accurate(device, graph):
    rng = np.random.default_rng(73)
    image = rng.uniform(.1, 1, (31, 31)).astype(np.float32)
    psf = np.zeros_like(image)
    psf[15, 15] = 1

    def differentiate(values):
        with tf.GradientTape() as tape:
            tape.watch(values)
            actual = fftconv2p(values, tf.constant(psf), pad=4, cache_last_y=False)
            loss = tf.reduce_sum(actual**2)
        return tape.gradient(loss, values)

    with tf.device(device):
        function = tf.function(differentiate) if graph else differentiate
        gradient = function(tf.constant(image))
    assert gradient is not None and gradient.dtype == tf.float32
    np.testing.assert_allclose(gradient.numpy(), 2 * image, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('version, device, dtype, expected', [
    ('2.13.1', '/CPU:0', tf.float32, tf.float64),
    ('2.14.1', '/CPU:0', tf.float32, tf.float64),
    ('2.15.0', '/CPU:0', tf.float32, tf.float32),
    ('2.19.0', '/CPU:0', tf.float32, tf.float32),
    ('2.20.0', '/CPU:0', tf.float32, tf.float32),
    ('2.13.1', '/GPU:0', tf.float32, tf.float32),
    ('2.13.1', '/CPU:0', tf.complex64, tf.complex128),
    ('2.13.1', '/CPU:0', tf.float64, tf.float64),
    ('2.13.1', '/CPU:0', tf.complex128, tf.complex128),
])
def test_fft_precision_policy_respects_backend_and_device(monkeypatch, version, device, dtype, expected):
    monkeypatch.setattr(tf, '__version__', version)
    assert _fft_compute_dtype(dtype, device) == expected


@pytest.mark.parametrize('dtype', [tf.float32, tf.complex64])
def test_one_fft_graph_runs_accurately_on_cpu_and_gpu(dtype):
    configure_eager()
    image, psf = _scene((520, 520))

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)] * 2)
    def convolve(values, kernel):
        return fftconv2p(values, kernel, pad=1, dtype=dtype)

    devices = ['/CPU:0']
    if tf.config.get_visible_devices('GPU'):
        devices += ['/GPU:0', '/CPU:0']
    expected = signal.fftconvolve(image.astype(np.float64), psf.astype(np.float64), mode='same')
    for device in devices:
        with tf.device(device):
            actual = convolve(tf.constant(image), tf.constant(psf))
            eager = fftconv2p(tf.constant(image), tf.constant(psf), pad=1, dtype=dtype)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=2e-5)
        np.testing.assert_allclose(actual.numpy().sum(dtype=np.float64), expected.sum(), rtol=1e-6)
        np.testing.assert_array_equal(actual.numpy(), eager.numpy())
    assert convolve.experimental_get_tracing_count() == 1


def test_fft_graph_soft_placement_remains_accurate_on_cpu():
    if tf.config.get_visible_devices('GPU'):
        pytest.skip('CPU-only soft-placement case')
    configure_eager()
    image, psf = _scene((520, 520))
    previous = tf.config.get_soft_device_placement()

    @tf.function
    def convolve(values, kernel):
        with tf.device('/GPU:0'):
            return fftconv2p(values, kernel, pad=1)

    try:
        tf.config.set_soft_device_placement(True)
        actual = convolve(tf.constant(image), tf.constant(psf))
    finally:
        tf.config.set_soft_device_placement(previous)
    expected = signal.fftconvolve(image.astype(np.float64), psf.astype(np.float64), mode='same')
    assert 'CPU' in actual.device
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(actual.numpy().sum(dtype=np.float64), expected.sum(), rtol=1e-6)
