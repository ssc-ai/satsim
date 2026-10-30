"""Compare two SatSim trees on one GPU without importing either in the driver.

Run from any directory; results and generated synthetic data go under --output.
The trees must have their dependencies installed in the invoking interpreter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


HEIGHT, WIDTH, OSF, PAD, KERNEL = 1595, 2394, 5, 32, 31
CASES = [('sparse_static', 520, False), ('sparse_moving', 520, True),
         ('dense_static', 3600, False), ('dense_moving', 3600, True)]


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def worker(args):
    started = time.perf_counter()
    sys.path.insert(0, str(Path(args.tree).resolve()))
    import logging
    import numpy as np
    import tensorflow as tf
    import satsim
    import satsim.satsim as sim
    import satsim.image.epsf as epsf
    from satsim.image.psf import gen_gaussian
    from satsim.image.render import render_epsf
    from satsim.util import configure_eager, MultithreadedTaskQueue

    logging.basicConfig(level=logging.ERROR)
    configure_eager()
    assert tf.config.get_visible_devices('GPU'), 'No GPU visible'
    imported = time.perf_counter()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    captured = []
    active_luts = []
    original_generator = sim.image_generator
    original_render = sim.render_epsf

    def capture(*a, **kw):
        for frame in original_generator(*a, **kw):
            assert 'GPU' in frame[4].device and 'GPU' in frame[5].device
            captured.append((frame[4], frame[5]))
            yield frame

    def capture_lut(*a, **kw):
        active_luts.append(a[7])
        return original_render(*a, **kw)

    sim.image_generator = capture
    sim.render_epsf = capture_lut
    if hasattr(epsf, '_accumulate_stars_cpu'):
        def reject_cpu(*a, **kw):
            raise AssertionError('GPU benchmark entered the CPU accumulator')
        epsf._accumulate_stars_cpu = reject_cpu

    queue = MultithreadedTaskQueue(num_threads=1)
    with tf.device('/GPU:0'):
        psf = gen_gaussian(315, 315, 6.25)
        lut = epsf.build_epsf_lut(psf, OSF, KERNEL)
        # Synchronize CUDA initialization and optical table generation separately.
        lut_hash = digest(lut.numpy())
        scenes = {}
        for case, count, moving in CASES:
            rng = np.random.default_rng(445 + count)
            rows = rng.uniform(100, HEIGHT - 100, count)
            cols = rng.uniform(100, WIDTH - 100, count)
            magnitudes = rng.uniform(10, 17, count)
            scenes[case] = (tf.constant(((rows + PAD + 0.5) * OSF - 0.5).astype('float32')),
                            tf.constant(((cols + PAD + 0.5) * OSF - 0.5).astype('float32')),
                            tf.constant((3 * 10 ** ((23.497 - magnitudes) / 2.5)).astype('float32')))
    setup_done = time.perf_counter()
    print('RESULT ' + json.dumps(dict(ready=True, version=satsim.__version__,
          source=satsim.__file__, import_seconds=imported - started,
          gpu_setup_seconds=setup_done - imported, lut_sha256=lut_hash,
          tf_version=tf.__version__, numpy_version=np.__version__,
          tf_build=tf.sysconfig.get_build_info(),
          deterministic=os.environ.get('TF_DETERMINISTIC_OPS', '0'))), flush=True)

    for line in sys.stdin:
        task = json.loads(line)
        if task.get('stop'):
            queue.stop()
            queue.waitUntilEmpty()
            break
        case = task['case']
        count, moving = next((n, m) for c, n, m in CASES if c == case)
        mode = task.get('mode', 'transform')
        kind = task['kind']
        captured.clear()
        active_luts.clear()
        tf.random.set_seed(42)
        np.random.seed(42)
        tf.config.experimental.reset_memory_stats('GPU:0')
        before_memory = tf.config.experimental.get_memory_info('GPU:0')['current']
        trace = task.get('trace')
        if trace:
            tf.profiler.experimental.start(trace, options=tf.profiler.experimental.ProfilerOptions(
                host_tracer_level=2, python_tracer_level=0, device_tracer_level=1))
        stamp = time.perf_counter()
        if kind == 'stage':
            with tf.device('/GPU:0'):
                rows, cols, counts = scenes[case]
                kwargs = {}
                if task.get('force_tf'):
                    kwargs['star_accumulator'] = 'tensorflow'
                result = render_epsf(
                    HEIGHT * OSF, WIDTH * OSF, (HEIGHT + 2 * PAD) * OSF, (WIDTH + 2 * PAD) * OSF,
                    PAD * OSF, PAD * OSF, OSF, lut,
                    [HEIGHT / 2 * OSF], [WIDTH / 2 * OSF], [1000.], rows, cols, counts,
                    0., 3., 54 if count == 3600 else 38, 0.0002 if moving else 0.,
                    [3.5, -5.5] if moving else [0., 0.],
                    star_render_mode=mode, psf_os=psf,
                    epsf_crop={'mode': 'auto', 'noise_fraction': 0.1,
                               'sigma_pix': 10., 'sizes': [5, 9, 17]}, **kwargs)
                assert 'GPU' in result[0].device and 'GPU' in result[1].device
                float(tf.reduce_sum(result[0] + result[1]).numpy())
                captured.append(result[:2])
                active_luts.append(lut)
            directory = None
        else:
            from astropy.io import fits
            config = json.loads((root / 'inputs' / (case + '.json')).read_text())
            config['sim']['star_render_mode'] = mode
            config['sim']['epsf']['$cache'] = str(root / 'cache' / args.label / case)
            directory = root / 'runs' / args.label / task['tag']
            with tf.device('/GPU:0'):
                sim.gen_images(config, eager=True, output_dir=str(directory.parent),
                               set_name=directory.name, queue=queue, output_debug=False)
            queue.waitUntilEmpty()
            # Generation and all output writes have completed before the timer stops.
            assert len(list((directory / 'ImageFiles').glob('*.fits'))) == 3
            assert len(list((directory / 'Annotations').glob('*.json'))) == 3
        elapsed = time.perf_counter() - stamp
        if trace:
            tf.profiler.experimental.stop()
        memory = tf.config.experimental.get_memory_info('GPU:0')
        answer = dict(seconds=elapsed, memory_before=before_memory, memory=memory,
                      directory=str(directory) if directory else None,
                      frames=len(captured), lut_sha256=digest(active_luts[-1].numpy()),
                      cpu_accumulator_calls=0, host_load=os.getloadavg())
        if directory is not None:
            pixels = fits.getdata(next((directory / 'ImageFiles').glob('*.fits')))
            assert np.ptp(pixels) > 4, 'Invalid fixture: constant or saturated output'
            answer['pixel_range'] = [int(pixels.min()), int(pixels.max())]
        if task.get('save'):
            prefix = root / 'arrays' / args.label / task['tag']
            prefix.parent.mkdir(parents=True, exist_ok=True)
            for index, (stars, targets) in enumerate(captured):
                for name, image in [('stars', stars), ('targets', targets)]:
                    array = image.numpy()
                    assert np.isfinite(array).all(), (case, name)
                    np.save(str(prefix) + f'_{index}_{name}.npy', array)
            answer['array_prefix'] = str(prefix)
        print('RESULT ' + json.dumps(answer), flush=True)


def prepare(output, tree):
    import numpy as np
    path = output / 'inputs'
    path.mkdir(parents=True, exist_ok=True)
    for case, count, moving in CASES:
        rng = np.random.default_rng(445 + count)
        rows = rng.uniform(100, HEIGHT - 100, count)
        cols = rng.uniform(100, WIDTH - 100, count)
        magnitudes = rng.uniform(10, 17, count)
        catalog = path / f'{case}.csv'
        ra, dec = 279.23410832, 38.78299311
        ra_values = ra + (cols - (WIDTH - 1) / 2) / WIDTH / np.cos(np.deg2rad(dec))
        dec_values = dec + (rows - (HEIGHT - 1) / 2) / WIDTH
        np.savetxt(catalog, np.column_stack([np.arange(count), magnitudes, ra_values, dec_values]),
                   delimiter=',', fmt=['%d', '%.10f', '%.12f', '%.12f'])
        config = json.loads((tree / 'tests/config_static.json').read_text())
        config['sim'].update(mode='epsf', spacial_osf=OSF, temporal_osf=54 if count == 3600 else 38,
                             padding=PAD, enable_shot_noise=False, calculate_snr=True,
                             save_ground_truth=False, save_jpeg=False, save_movie=False, save_czml=False,
                             save_pickle=False, star_annotation_threshold=18,
                             star_catalog_query_mode='at_start', apply_star_wrap_around=False,
                             point_rendering='bilinear')
        config['sim']['epsf'] = dict(kernel_size=KERNEL, psf_generation_size=63, normalize=False,
                                     crop=dict(mode='auto', noise_fraction=0.1, sizes=[5, 9, 17]))
        config['fpa'].update(height=HEIGHT, width=WIDTH, y_fov=HEIGHT / WIDTH, x_fov=1.,
                             num_frames=3, dark_current=0, psf=dict(mode='gaussian', eod=0.5),
                             noise=dict(read=10, electronic=0))
        config['fpa']['time'].update(exposure=3., gap=0.1)
        config['background']['galactic'] = 22
        config['geometry'] = dict(
            stars=dict(mode='csv', path=str(catalog), ra=ra, dec=dec, rotation=0., pad=0.,
                       motion=dict(mode='affine', rotation=0.0002 if moving else 0.,
                                   translation=[0.7, -1.1] if moving else [0., 0.])),
            obs=dict(mode='list', list=[dict(mode='line', origin=[0.5, 0.5], velocity=[0., 0.], pe=1000.)]))
        (path / f'{case}.json').write_text(json.dumps(config, indent=2) + '\n')


class Worker:
    def __init__(self, script, tree, label, args):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   NUMBA_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
                   TF_CPP_MIN_LOG_LEVEL='2', TF_DETERMINISTIC_OPS='1' if args.deterministic else '0')
        if args.cupti_libdir:
            env['LD_LIBRARY_PATH'] = args.cupti_libdir + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        self.error = open(Path(args.output) / f'{label}-stderr.log', 'w')
        self.process = subprocess.Popen([sys.executable, str(script), '--worker', '--tree', str(tree),
                                        '--label', label, '--output', str(args.output)],
                                        cwd=tree, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.error, text=True, bufsize=1)
        self.ready = self.read()

    def read(self):
        for line in self.process.stdout:
            if line.startswith('RESULT '):
                return json.loads(line[7:])
        raise RuntimeError(f'Worker exited {self.process.poll()}; inspect {self.error.name}')

    def run(self, task):
        self.process.stdin.write(json.dumps(task) + '\n')
        self.process.stdin.flush()
        return self.read()

    def stop(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"stop": true}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=60)
        self.error.close()


def main(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.output = str(output)
    prepare(output, Path(args.candidate))
    script = Path(__file__).resolve()
    workers = {}
    result = dict(arguments=vars(args), hardware=subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader'], text=True),
        commits={label: subprocess.check_output(['git', '-C', tree, 'rev-parse', 'HEAD'], text=True).strip()
                 for label, tree in [('baseline', args.baseline), ('candidate', args.candidate)]},
        setup={}, cold={}, measurements=[])
    path = output / 'results.json'

    def record(task, label, section):
        data = workers[label].run(task)
        data.update(task, version=label)
        section.append(data)
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(f"{label} {task['kind']} {task['case']} {task['tag']}: {data['seconds']:.3f}s", flush=True)
        return data

    try:
        for label, tree in [('baseline', args.baseline), ('candidate', args.candidate)]:
            workers[label] = Worker(script, Path(tree), label, args)
            result['setup'][label] = workers[label].ready
        assert result['setup']['baseline']['lut_sha256'] == result['setup']['candidate']['lut_sha256']
        kinds = [] if args.extra_only else ['stage', 'collect']
        for kind in kinds:
            for case, _, _ in CASES:
                cold = result['cold'].setdefault(kind + '_' + case, [])
                for label in workers:
                    record(dict(kind=kind, case=case, tag='cold_' + kind + '_' + case, save=True), label, cold)
                for round_number in range(args.rounds):
                    order = ['baseline', 'candidate'] if round_number % 2 == 0 else ['candidate', 'baseline']
                    for label in order:
                        record(dict(kind=kind, case=case, round=round_number,
                                    tag=f'{kind}_{case}_{round_number}', save=round_number == 0),
                               label, result['measurements'])
        # Separate auto-dispatch overhead from the shared rendering changes.
        result['dispatch'] = []
        for i in range(0 if args.extra_only else args.rounds):
            for force_tf in ([False, True] if i % 2 == 0 else [True, False]):
                record(dict(kind='stage', case='dense_moving', force_tf=force_tf,
                            tag=f'dispatch_{i}_{force_tf}'), 'candidate', result['dispatch'])
        # The legacy shared trail computes a different image. Measure its cost separately.
        result['streak'] = []
        for i in range(0 if args.extra_only else args.rounds):
            order = ['baseline', 'candidate'] if i % 2 == 0 else ['candidate', 'baseline']
            for label in order:
                record(dict(kind='stage', case='dense_moving', mode='streak',
                            tag=f'streak_{i}', save=i == 0), label, result['streak'])
        if args.streak_collect:
            result['streak_collect'] = []
            for label in workers:
                record(dict(kind='collect', case='dense_moving', mode='streak',
                            tag='cold_streak_collect', save=True), label,
                       result['cold'].setdefault('streak_collect', []))
            for i in range(args.rounds):
                order = ['baseline', 'candidate'] if i % 2 == 0 else ['candidate', 'baseline']
                for label in order:
                    record(dict(kind='collect', case='dense_moving', mode='streak',
                                tag=f'streak_collect_{i}', save=i == 0), label, result['streak_collect'])
        if args.profile:
            result['profiles'] = []
            for kind in ['stage', 'collect']:
                for label in workers:
                    if args.extra_only:
                        record(dict(kind=kind, case='dense_moving', tag=f'warm_profile_{kind}'), label,
                               result['cold'].setdefault('profile_warmup', []))
                    record(dict(kind=kind, case='dense_moving', tag=f'profile_{kind}',
                                trace=str(output / 'profiles' / label / kind)), label, result['profiles'])
    finally:
        for process in workers.values():
            process.stop()
        path.write_text(json.dumps(result, indent=2) + '\n')


def image_metrics(actual, reference):
    import numpy as np
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)

    def moments(image):
        flux = image.sum()
        yy, xx = np.arange(image.shape[0]), np.arange(image.shape[1])
        row, col = image.sum(axis=1), image.sum(axis=0)
        cy, cx = row @ yy / flux, col @ xx / flux
        values = np.array([row @ (yy - cy) ** 2, np.einsum('i,ij,j->', yy - cy, image, xx - cx),
                           col @ (xx - cx) ** 2]) / flux
        return np.array([cy, cx]), values

    ca, ma = moments(actual)
    cr, mr = moments(reference)
    result = dict(l2=float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)),
                  flux=float(abs(actual.sum() - reference.sum()) / abs(reference.sum())),
                  centroid_px=float(np.linalg.norm(ca - cr)),
                  moment=float(np.linalg.norm(ma - mr) / np.linalg.norm(mr)))
    return result


def profile_summary(root):
    from tensorflow.core.profiler.protobuf import xplane_pb2
    from collections import Counter
    summary = {}
    for version in ['baseline', 'candidate']:
        summary[version] = {}
        for kind in ['stage', 'collect']:
            path = next((root / 'profiles' / version / kind).rglob('*.xplane.pb'))
            space = xplane_pb2.XSpace()
            space.ParseFromString(path.read_bytes())
            copies = Counter()
            for plane in space.planes:
                for line in plane.lines:
                    for event in line.events:
                        name = plane.event_metadata[event.metadata_id].name
                        if not name.startswith('Memcpy'):
                            continue
                        for stat in event.stats:
                            if plane.stat_metadata[stat.metadata_id].name == 'memcpy_details':
                                details = (plane.stat_metadata[stat.ref_value].name
                                           if stat.HasField('ref_value') else stat.str_value)
                                copies[(name, int(re.search(r'size:(\d+)', details)[1]))] += 1
            summary[version][kind] = dict(
                counts=[dict(kind=name, bytes=size, count=count) for (name, size), count in sorted(copies.items())],
                totals={name: sum(size * count for (kind_name, size), count in copies.items() if kind_name == name)
                        for name in sorted(set(name for name, _ in copies))},
                exported_device_timeline_events=sum(len(line.events) for plane in space.planes
                                                    if plane.name.startswith('/device:') for line in plane.lines))
            assert copies, 'No CUDA memcpy records; check that CUPTI was loaded'
    return summary


def analyze(args):
    import numpy as np
    from astropy.io import fits
    from scipy.ndimage import maximum_filter
    root = Path(args.output)
    normal = json.loads((root / 'results.json').read_text())
    extra = json.loads((Path(args.extra_output) / 'results.json').read_text())
    deterministic = json.loads((Path(args.deterministic_output) / 'results.json').read_text())
    streak_key = 'streak' if args.corrected_baseline else 'legacy_streak'
    report = dict(commits=normal['commits'], setup=normal['setup'], timing=[],
                  fidelity={}, exports={}, profiles=profile_summary(Path(args.extra_output)),
                  comparison='corrected' if args.corrected_baseline else 'legacy')
    report[streak_key] = {}
    for kind in ['stage', 'collect']:
        for case, _, _ in CASES:
            item = dict(kind=kind, case=case)
            for version in ['baseline', 'candidate']:
                values = [m for m in normal['measurements'] if m['kind'] == kind
                          and m['case'] == case and m['version'] == version]
                assert len(values) == normal['arguments']['rounds'] and len(values) >= 5
                seconds = [m['seconds'] for m in values]
                item[version] = dict(median=statistics.median(seconds), minimum=min(seconds), maximum=max(seconds),
                                     peak_mib=statistics.median(m['memory']['peak'] / 2**20 for m in values))
            item['speedup'] = item['baseline']['median'] / item['candidate']['median']
            report['timing'].append(item)

    cohorts = [('normal', normal), ('deterministic', deterministic)]
    if args.corrected_baseline:
        cohorts.append(('extra', extra))
    for label, results in cohorts:
        maxima = dict(l2=0., flux=0., centroid_px=0., moment=0., aperture_tolerance_ratio=0.)
        images, targets_bitwise = 0, True
        sections = list(results['cold'].values()) + [results['measurements']]
        if args.corrected_baseline:
            sections += [results.get(key, []) for key in ['streak', 'streak_collect', 'profiles']]
        for section in sections:
            for base in section:
                if base['version'] != 'baseline' or 'array_prefix' not in base:
                    continue
                candidate = next(v for v in section if v['version'] == 'candidate' and v['tag'] == base['tag'])
                assert base['lut_sha256'] == candidate['lut_sha256'], 'Different optical LUTs'
                for index in range(base['frames']):
                    for component in ['stars', 'targets']:
                        b = np.load(base['array_prefix'] + f'_{index}_{component}.npy')
                        c = np.load(candidate['array_prefix'] + f'_{index}_{component}.npy')
                        assert np.isfinite(b).all() and np.isfinite(c).all()
                        metrics = image_metrics(c, b)
                        assert metrics['l2'] <= 1e-5 and metrics['flux'] <= 1e-5, (label, base['tag'], metrics)
                        assert metrics['centroid_px'] <= 1e-4 and metrics['moment'] <= 1e-4, metrics
                        for key, value in metrics.items():
                            maxima[key] = max(maxima[key], value)
                        if component == 'targets':
                            targets_bitwise &= np.array_equal(b, c)
                        else:
                            # Bright-source aperture checks are independent of the global field centroid.
                            peaks = np.argwhere((b == maximum_filter(b, size=31)) & (b > b.max() * 0.01))
                            peaks = sorted(peaks, key=lambda rc: b[tuple(rc)], reverse=True)[:25]
                            for row, col in peaks:
                                if row < 16 or col < 16 or row >= b.shape[0] - 16 or col >= b.shape[1] - 16:
                                    continue
                                rb = b[row - 15:row + 16, col - 15:col + 16].astype(np.float64)
                                rc = c[row - 15:row + 16, col - 15:col + 16].astype(np.float64)
                                yy, xx = np.indices(rb.shape)
                                for radius in [1.5, 3., 6.]:
                                    mask = (yy - 15) ** 2 + (xx - 15) ** 2 <= radius ** 2
                                    tolerance = max(1e-5 * abs(rb[mask].sum()), rb.sum() * 1e-8)
                                    ratio = float(abs(rc[mask].sum() - rb[mask].sum()) / tolerance)
                                    assert ratio <= 1, (label, base['tag'], ratio)
                                    maxima['aperture_tolerance_ratio'] = max(maxima['aperture_tolerance_ratio'], ratio)
                        images += 1
        if label == 'deterministic':
            assert targets_bitwise, 'Target path changed under deterministic execution'
        report['fidelity'][label] = dict(images=images, targets_bitwise=bool(targets_bitwise), maxima=maxima)

        frames, objects, maximum_adu, changed_pixels = 0, 0, 0, 0
        for section in sections:
            for base in section:
                if base['version'] != 'baseline' or base['kind'] != 'collect':
                    continue
                candidate = next(v for v in section if v['version'] == 'candidate' and v['tag'] == base['tag'])
                bd, cd = Path(base['directory']), Path(candidate['directory'])
                for bp in sorted((bd / 'Annotations').glob('*.json')):
                    cp = cd / 'Annotations' / bp.name
                    bj, cj = [json.loads(p.read_text()) for p in [bp, cp]]
                    if args.corrected_baseline:
                        assert bj == cj, ('Annotation/SNR/metadata difference', bp)
                    bj, cj = bj['data'], cj['data']
                    assert bj['objects'] == cj['objects'], ('Annotation/SNR difference', bp, cp)
                    bm, cm = bj['metadata'], cj['metadata']
                    if not args.corrected_baseline:
                        bm['epsf']['crop'].pop('trail_det', None)
                    assert bm == cm, ('Unexpected frame metadata difference', bp)
                    objects += len(bj['objects'])
                    bf, cf = [directory / 'ImageFiles' / (bp.stem + '.fits') for directory in [bd, cd]]
                    bh, ch = dict(fits.getheader(bf)), dict(fits.getheader(cf))
                    if not args.corrected_baseline:
                        bh.pop('SWCREATE')
                        ch.pop('SWCREATE')
                    assert bh == ch, ('Unexpected FITS header difference', bp)
                    b, c = [fits.getdata(p).astype(np.int32) for p in [bf, cf]]
                    assert b.shape == c.shape == (HEIGHT, WIDTH) and np.ptp(b) > 4 and np.ptp(c) > 4
                    delta = np.abs(b - c)
                    maximum_adu = max(maximum_adu, int(delta.max()))
                    changed_pixels += int(np.count_nonzero(delta))
                    assert delta.max() <= 1, ('Exported image difference', bp, delta.max())
                    frames += 1
        report['exports'][label] = dict(paired_frames=frames, annotation_objects_compared=objects,
                                        maximum_adu_difference=maximum_adu, differing_pixels=changed_pixels,
                                        header_exclusions=[] if args.corrected_baseline else ['SWCREATE'],
                                        metadata_exclusions=[] if args.corrected_baseline else ['epsf.crop.trail_det'])

    for key, values in [('stage', normal['streak']), ('collect', extra['streak_collect'])]:
        entry = {}
        for version in ['baseline', 'candidate']:
            seconds = [v['seconds'] for v in values if v['version'] == version]
            entry[version] = dict(median=statistics.median(seconds), minimum=min(seconds), maximum=max(seconds))
        entry['runtime_ratio'] = entry['candidate']['median'] / entry['baseline']['median']
        base = next(v for v in values if v['version'] == 'baseline' and 'array_prefix' in v)
        candidate = next(v for v in values if v['version'] == 'candidate' and 'array_prefix' in v)
        entry['star_image_differences'] = [image_metrics(
            np.load(candidate['array_prefix'] + f'_{i}_stars.npy'),
            np.load(base['array_prefix'] + f'_{i}_stars.npy')) for i in range(base['frames'])]
        report[streak_key][key] = entry
    report['dispatch'] = {mode: statistics.median(m['seconds'] for m in normal['dispatch'] if m['force_tf'] == forced)
                          for mode, forced in [('auto', False), ('tensorflow', True)]}
    report['source_sha256'] = {
        version: {str(p): hashlib.sha256((Path(tree) / p).read_bytes()).hexdigest()
                  for p in ['satsim/image/epsf.py', 'satsim/image/render.py', 'satsim/satsim.py']}
        for version, tree in [('baseline', normal['arguments']['baseline']),
                              ('candidate', normal['arguments']['candidate'])]}
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ['profiles', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline')
    parser.add_argument('--candidate')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--cupti-libdir', help='Optional directory containing the matching CUPTI shared library')
    parser.add_argument('--extra-only', action='store_true', help='Only run requested streak-collect/profile checks')
    parser.add_argument('--streak-collect', action='store_true', help='Also compare legacy/corrected streak collects')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--corrected-baseline', action='store_true',
                        help='Analyze corrected renderers with exact metadata and additional streak fidelity checks')
    parser.add_argument('--extra-output')
    parser.add_argument('--deterministic-output')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--tree', help=argparse.SUPPRESS)
    parser.add_argument('--label', help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
    elif arguments.analyze:
        analyze(arguments)
    else:
        main(arguments)
