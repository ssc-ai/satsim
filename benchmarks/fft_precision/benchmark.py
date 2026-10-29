"""Compare FFT accuracy and warm execution time in two SatSim checkouts.

Run with each interpreter being qualified; --device selects CPU or GPU.
Timing excludes import, tracing and initial kernel FFT creation. Run while idle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def worker(args):
    sys.path.insert(0, str(Path(args.tree).resolve()))
    import numpy as np
    import scipy
    import tensorflow as tf
    from scipy.signal import fftconvolve
    from satsim.math.fft import fftconv2p
    from satsim.util import configure_eager

    configure_eager()
    if args.device == 'GPU':
        assert tf.config.get_visible_devices('GPU'), 'GPU required'
    rows = []
    for size in (192, 520, 2136):
        yy, xx = np.indices((size, size), dtype=np.float64)
        psf = np.exp(-((yy - (size - 1) / 2)**2 + (xx - (size - 1) / 2)**2) / (2 * 4.5**2))
        psf = (psf / psf.sum()).astype(np.float32)
        image = np.zeros((size, size), np.float32)
        image[size // 2, size // 2] = 1000
        expected = fftconvolve(image.astype(np.float64), psf.astype(np.float64), mode='same')
        for graph in (False, True):
            with tf.device('/' + args.device + ':0'):
                values, kernel = tf.constant(image), tf.constant(psf)

                def convolve(values, kernel):
                    return fftconv2p(values, kernel, pad=1)

                function = tf.function(convolve) if graph else convolve
                for _ in range(3):
                    function(values, kernel).numpy()
                elapsed = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        actual = function(values, kernel).numpy()
                    elapsed.append((time.perf_counter() - start) / args.iterations)
            rows.append(dict(size=size, graph=graph, seconds=elapsed,
                             median_seconds=statistics.median(elapsed),
                             flux=float(actual.sum(dtype=np.float64)),
                             expected_flux=float(expected.sum()),
                             max_absolute_error=float(np.max(np.abs(actual - expected))),
                             sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                             output_dtype=str(actual.dtype)))
    print(json.dumps(dict(tensorflow=tf.__version__, numpy=np.__version__, scipy=scipy.__version__,
                          python=platform.python_version(), platform=platform.platform(), device=args.device,
                          source=str(Path(args.tree).resolve()), results=rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before')
    parser.add_argument('--after')
    parser.add_argument('--output')
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='CPU')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--tree', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tree:
        worker(args)
        return
    if not (args.before and args.after and args.output):
        parser.error('--before, --after and --output are required')
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       NUMBA_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='1',
                       TF_NUM_INTEROP_THREADS='1', TF_CPP_MIN_LOG_LEVEL='3')
    if args.device == 'CPU':
        environment['CUDA_VISIBLE_DEVICES'] = ''
    results = {}
    for label, tree in [('before', args.before), ('after', args.after)]:
        command = [sys.executable, str(Path(__file__).resolve()), '--tree', tree,
                   '--device', args.device, '--repeats', str(args.repeats),
                   '--iterations', str(args.iterations)]
        completed = subprocess.run(command, env=environment, check=True, text=True,
                                   stdout=subprocess.PIPE)
        results[label] = json.loads(completed.stdout)
    results['source_sha256'] = {
        label: {name: hashlib.sha256(Path(tree, name).read_bytes()).hexdigest()
                for name in ('satsim/math/fft.py', 'satsim/satsim.py')}
        for label, tree in [('before', args.before), ('after', args.after)]}
    results['iterations_per_sample'] = args.iterations
    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
