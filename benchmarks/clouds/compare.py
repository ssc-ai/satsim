"""Reproduce cloud variation diagnostics against the pre-feature release commit.

Run from the repository root with PYTHONPATH=. and OMP_NUM_THREADS=1.
Uses only the pinned repository sources, not an installed older SatSim package.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import tracemalloc
import types
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from satsim.clouds import CloudGeometry, crop_cloud_field, generate_cloud_layer, parse_cloud_layers
from satsim.clouds.field import combine_cloud_layers, optical_fields
from satsim.clouds.models import cloud_density_floor


BASE = '2143a20'
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/workspace/git/satsim-utils/datasets/clouds_v1/runs/fig_patchy2/configs/lmnt03/lmnt03_patchy_low_00.json')


def baseline():
    """Load the frozen comparison implementation without adding a runtime mode."""
    modules = {}
    for name in ('field', 'generation'):
        source = subprocess.check_output(
            ['git', 'show', f'{BASE}:satsim/clouds/{name}.py'], cwd=ROOT, text=True)
        module = types.ModuleType(f'baseline_{name}')
        exec(compile(source, f'{BASE}/{name}.py', 'exec'), module.__dict__)
        modules[name] = module
    modules['generation'].cloud_layer_from_density = modules['field'].cloud_layer_from_density
    return modules


def stats(layer, old):
    t = layer.transmission
    cloudy = layer.mask
    old_cloudy = old.mask
    old_values, counts = np.unique(old.transmission[old_cloudy], return_counts=True)
    plateau = old_cloudy & (old.transmission == old_values[np.argmax(counts)])
    interior = layer.density >= 1 - 1e-7
    # Adjacent cloudy pixels isolate local texture from the clear/cloud boundary.
    pairs = cloudy[:, 1:] & cloudy[:, :-1]
    gradient = np.abs(np.diff(t, axis=1))[pairs]
    return {
        'mean_transmission': float(t.mean()),
        'min_transmission': float(t.min()),
        'max_transmission': float(t.max()),
        'mask_coverage': float(cloudy.mean()),
        'interior_fraction': float(interior.mean()),
        'interior_transmission_min': float(t[interior].min()) if interior.any() else None,
        'interior_transmission_max': float(t[interior].max()) if interior.any() else None,
        'flat_cloudy_neighbor_fraction': float(np.mean(gradient < 1e-7)) if gradient.size else None,
        'old_largest_cloudy_plateau_fraction': float(plateau.sum() / old_cloudy.sum()),
        'range_on_old_plateau': float(np.ptp(t[plateau])),
        'std_on_old_plateau': float(np.std(t[plateau])),
    }


def measure(func, *args):
    func(*args)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        func(*args)
        samples.append(time.perf_counter() - start)
    tracemalloc.start()
    func(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'median_seconds': float(np.median(samples)), 'peak_traced_mb': peak / 1024**2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/clouds'))
    parser.add_argument('--junit', type=Path, help='Full-suite JUnit evidence to record with this run.')
    parser.add_argument('--targeted-junit', type=Path, help='Focused cloud-suite JUnit evidence.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old = baseline()
    # Cloud settings and angular geometry from the archived AMOS scene. Preview
    # resolution is reduced explicitly; this is not a full detector rerender.
    archive_config = {'type': 'patchy', 'seed': 827990538, 'coverage': 0.3435,
                      'optical': {'tau_max': 3.5625, 'tau_gamma': 1.1661},
                      'illumination': {'source_gains': {'lunar': 0.227}}, 'wind_speed': 0.0}
    if ARCHIVE.exists():
        assert json.loads(ARCHIVE.read_text())['clouds'][0] == archive_config
    cases = [
        ('AMOS patchy example', archive_config),
        ('Haze with a density floor', {'type': 'fog', 'seed': 42, 'coverage': 0.5}),
        ('Full cloud coverage', {'type': 'custom', 'seed': 42, 'coverage': 1.0,
                                 'feature_scales_m': [10, 20, 40, 80],
                                 'optical': {'tau_min': 0.05, 'tau_max': 1.5}}),
    ]
    geometry = CloudGeometry(384, 288, 3.193994030187939, 2.400005216608189)
    results = {'baseline_commit': BASE, 'archive_path': str(ARCHIVE),
               'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() if ARCHIVE.exists() else None,
               'preview_geometry': vars(geometry), 'cases': {},
               'notes': ['Field-only diagnostics at reduced detector resolution, same angular footprint.',
                         'Old and new defaults share identical original textures and tau settings.',
                         'The explicit bounded example replaces tau settings with min=0.2, max=0.8.',
                         'Bounds are allowed intervals, not a guarantee every view reaches both endpoints.',
                         'Mean attenuation changes are reported, not fitted away.']}
    results['environment'] = {
        'python': platform.python_version(), 'numpy': np.__version__,
        'system': platform.platform(),
        'threads': {key: os.environ.get(key) for key in
                    ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS',
                     'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
    }
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
    detail, detail_axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    colors = ['#7d8790', '#0676bb', '#c65a19']
    names = ['Before', 'New default', 'Explicit range 0.2–0.8']
    for row, (title, raw) in enumerate(cases):
        cfg = parse_cloud_layers([raw], sim_seed=1)[0]
        bounded_raw = copy.deepcopy(raw)
        bounded_raw.pop('optical', None)
        bounded_raw['transmission'] = {'min': 0.2, 'max': 0.8}
        bounded_cfg = parse_cloud_layers([bounded_raw], sim_seed=1)[0]
        layers = [old['generation'].generate_cloud_layer(cfg, geometry),
                  generate_cloud_layer(cfg, geometry), generate_cloud_layer(bounded_cfg, geometry)]
        values, counts = np.unique(layers[0].transmission[layers[0].mask], return_counts=True)
        plateau = layers[0].mask & (layers[0].transmission == values[np.argmax(counts)])
        cross_row = int(np.argmax(plateau.sum(axis=1)))
        results['cases'][title] = {'config': raw, 'bounded_config': bounded_raw,
                                   'cross_section_row': cross_row,
                                   'measurements': {name: stats(layer, layers[0]) for name, layer in zip(names, layers)}}
        for col, (name, layer, color) in enumerate(zip(names, layers, colors)):
            im = axes[row, col].imshow(layer.transmission, vmin=0, vmax=1, cmap='viridis')
            axes[row, col].set_title(name if row == 0 else '')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].axhline(cross_row, color='white', alpha=0.6, linewidth=0.7)
            if col == 0:
                axes[row, col].set_ylabel(title)
            axes[row, col].set_xlabel(f'Mean transmission {layer.transmission.mean():.3f}')
            detail_axes[row, 0].plot(layer.transmission[cross_row], label=name, color=color)
            detail_axes[row, 1].hist(layer.transmission[layer.mask], bins=np.linspace(0, 1, 61),
                                     histtype='step', density=True, label=name, color=color)
        detail_axes[row, 0].set(title=title, ylim=(-0.02, 1.02),
                                xlabel=f'Column (row {cross_row}: most old plateau pixels)', ylabel='Transmission')
        detail_axes[row, 1].set(xlabel='Cloud-mask transmission', ylabel='Probability density')
    fig.colorbar(im, ax=axes, shrink=0.65, label='Transmission: 0 = opaque, 1 = clear')
    fig.suptitle('Cloud interiors retain texture\nSame seed and angular field; attenuation changes shown explicitly', fontsize=15)
    detail_axes[0, 0].legend(fontsize=8)
    detail.suptitle('Cloud transmission cross-sections and distributions', fontsize=15)
    fig.savefig(args.output / 'comparison.png', dpi=150)
    detail.savefig(args.output / 'diagnostics.png', dpi=150)
    plt.close('all')
    texture = np.linspace(0, 1, 1024**2, dtype=np.float32).reshape(1024, 1024)
    envelope = old['generation'].cloud_density_from_noise(texture, 0.5, 0.08, None)
    cfg = parse_cloud_layers([archive_config], sim_seed=1)[0]
    bounded = parse_cloud_layers([{'type': 'custom', 'transmission': {'min': 0.2, 'max': 0.8}}], sim_seed=1)[0]
    results['mapping_benchmark'] = {
        'pixels': texture.size, 'repeats': 5,
        'old': measure(old['field'].optical_fields, envelope, cfg),
        'default': measure(optical_fields, envelope, cfg, texture),
        'bounded': measure(optical_fields, envelope, bounded, texture),
        'scope': 'Mapping only, same arrays; excludes generation, rendering, export and persistent texture storage.',
    }
    performance_geometry = CloudGeometry(1024, 1024, 3, 3)
    perf_cfg = parse_cloud_layers([{'type': 'patchy', 'seed': 42}], sim_seed=1)[0]
    perf_bounded = parse_cloud_layers([{'type': 'patchy', 'seed': 42,
                                       'transmission': {'min': 0.2, 'max': 0.8}}], sim_seed=1)[0]
    old_layer = old['generation'].generate_cloud_layer(perf_cfg, performance_geometry)
    new_layer = generate_cloud_layer(perf_cfg, performance_geometry)
    bounded_layer = generate_cloud_layer(perf_bounded, performance_geometry)
    results['generation_benchmark'] = {
        'shape': [1024, 1024], 'repeats': 5,
        'old': measure(old['generation'].generate_cloud_layer, perf_cfg, performance_geometry),
        'default': measure(generate_cloud_layer, perf_cfg, performance_geometry),
        'bounded': measure(generate_cloud_layer, perf_bounded, performance_geometry),
        'scope': 'One patchy layer, noise + coverage + optics + statistics, after JIT warmup. Excludes rendering.',
    }
    results['crop_benchmark'] = {
        'shape': [512, 512], 'repeats': 5,
        'old': measure(old['field'].crop_cloud_field, combine_cloud_layers((old_layer,)), (-0.25, -0.5), 512, 512),
        'default': measure(crop_cloud_field, combine_cloud_layers((new_layer,)), (-0.25, -0.5), 512, 512),
        'bounded': measure(crop_cloud_field, combine_cloud_layers((bounded_layer,)), (-0.25, -0.5), 512, 512),
        'scope': 'Bilinear moving crop, optics + field composition + statistics; excludes rendering.',
    }
    results['parameter_sweep'] = parameter_sweep()
    results['persistent_texture_bytes_per_pixel_per_layer'] = 4
    results['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / 'satsim/clouds').glob('*.py'))}
    results['qualification'] = {}
    for label, path in (('full_suite', args.junit), ('cloud_suite', args.targeted_junit)):
        if path:
            suites = list(ET.parse(path).getroot().iter('testsuite'))
            counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                      for key in ('tests', 'failures', 'errors', 'skipped')}
            counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
            counts['junit_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            results['qualification'][label] = counts
    results['diagnostic_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    write_report(results, args.output)
    print(json.dumps({name: case['measurements'] for name, case in results['cases'].items()}, indent=2))


def parameter_sweep():
    """Check all presets over two seeds and five coverage values."""
    geometry = CloudGeometry(64, 64, 1, 1)
    rows = []
    for cloud_type in ('custom', 'patchy', 'cellular', 'veil', 'sheet', 'fog'):
        for seed in (1, 42):
            for coverage in (0, 0.1, 0.5, 0.9, 1):
                for bounded in (False, True):
                    raw = {'type': cloud_type, 'seed': seed, 'coverage': coverage}
                    if bounded:
                        raw['transmission'] = {'min': 0.2, 'max': 0.8}
                    cfg = parse_cloud_layers([raw], sim_seed=1)[0]
                    layer = generate_cloud_layer(cfg, geometry)
                    t = layer.transmission
                    interior = layer.density == 1
                    assert np.isfinite(t).all() and t.min() >= 0 and t.max() <= 1
                    if coverage == 0:
                        assert (t == 1).all()
                    if coverage == 1:
                        assert layer.mask.all() and np.ptp(t) > 0.01
                    if bounded and cloud_density_floor(coverage, cfg.density_floor) > 0 and coverage > 0:
                        assert t.min() >= 0.2 - 1e-7 and t.max() <= 0.8 + 1e-7
                    if bounded and interior.any():
                        assert t[interior].min() >= 0.2 - 1e-7 and t[interior].max() <= 0.8 + 1e-7
                    rows.append({'config': raw, 'mean_transmission': float(t.mean()),
                                 'transmission_range': float(np.ptp(t)),
                                 'mask_coverage': float(layer.mask.mean())})
    return {'checks': 'pass', 'cases': rows}


def write_report(results, output):
    lines = [
        '# Cloud transmission variation — v0.27.0', '',
        'The new default retains the original procedural texture separately from the',
        'coverage envelope. Saturated cloud interiors, density floors and full coverage',
        'therefore no longer flatten the optical field. Clear sky remains exactly 1.', '',
        '![Before/after transmission maps](comparison.png)', '',
        '![Cross-sections and histograms](diagnostics.png)', '',
        'The white line marks the row with the most pixels at the most common old cloudy',
        'transmission level. That plateau set may contain disconnected patches.',
        'Statistics below use the entire field, not just that row.', '',
        '## Method and behavior', '',
        f'Baseline: `{BASE}` (merged release branch before this feature). The AMOS row uses',
        'the archived LMNT03 patchy configuration and angular footprint recorded in',
        '`results.json`, with a reduced 384 × 288 field for this diagnostic. It is not a',
        'full-resolution rerender of the AMOS dataset. The other rows isolate a fog floor',
        'and full coverage with custom 10–80 m feature scales. The original texture and',
        'seed are identical across each row. No per-frame normalization is introduced.', '',
        '“New default” retains the same tau settings. “Explicit range” replaces them with',
        '`transmission: {"min": 0.2, "max": 0.8}`; it is a separate control example, not an',
        'attenuation-matched comparison. Coverage envelopes reach one in interiors,',
        'independently of coverage fraction. Mean attenuation changes are intentional',
        'and shown below; this is a behavior change for existing configurations.', '',
        '## Measurements', '',
        'Flat-neighbor fraction counts horizontally adjacent mask pixels with absolute',
        'transmission difference < 1e-7. It excludes clear-sky pairs. Plateau range uses',
        'all locations at the most common exact transmission value within the old cloud',
        'mask, without connected-component ranking. These tests detect lost texture,',
        'not meteorological realism.', '',
        '| Case / mode | Mean T | Mask coverage | Flat neighbors | Range on old plateau |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name, case in results['cases'].items():
        for mode, m in case['measurements'].items():
            lines.append(f"| {name} / {mode} | {m['mean_transmission']:.4f} | {m['mask_coverage']:.2%} | "
                         f"{m['flat_cloudy_neighbor_fraction']:.2%} | {m['range_on_old_plateau']:.4f} |")
    lines += ['', '## Performance', '',
              'Five warmed repetitions, median elapsed time and a separate traced peak',
              'allocation measurement. CPU functions only; no full-collect speed claim.', '',
              '| Operation / mode | Median ms | Peak traced MiB |', '| --- | ---: | ---: |']
    for key, name in [('mapping_benchmark', '1,048,576-pixel optical mapping'),
                      ('generation_benchmark', '1024 × 1024 layer generation'),
                      ('crop_benchmark', '512 × 512 moving crop')]:
        for mode in ('old', 'default', 'bounded'):
            m = results[key][mode]
            lines.append(f"| {name} / {mode} | {1000 * m['median_seconds']:.2f} | {m['peak_traced_mb']:.2f} |")
    lines += ['', 'The retained texture adds 4 bytes per pixel per layer to source and cropped',
              'fields (4 MiB per 1024² source layer). Traced peaks above measure allocations',
              'inside each operation, excluding preexisting input arrays. Bounded mapping',
              'uses float64 intermediates to keep nearly opaque endpoints stable.', '',
              '## Qualification', '',
              f"- Parameter sweep: {len(results['parameter_sweep']['cases'])} cases passed; six types, two seeds, five coverages, two optical modes."]
    for label, counts in results['qualification'].items():
        lines.append(f"- {label}: {counts['passed']} passed, {counts['failures']} failures, {counts['errors']} errors, {counts['skipped']} skipped.")
    lines += ['- The focused cloud suite is a subset of the full suite; counts are not additive.',
              '- Tests cover interior bounds, smooth edges, clear sky, constant inputs,',
              '  equal bounds, nearly opaque endpoints, ambiguous config rejection, moving',
              '  crops, layer stacking, glow, FFT/ePSF target flux, star flux and TIFF truth.',
              '- Float32 bounds allow rounding tolerance. Exact zero per-layer minimum is',
              '  rejected to keep optical depth finite; near-zero positive values are supported.',
              '- Bounds apply to full interiors; edges blend toward clear sky and can exceed max.',
              '  Positive-floor haze is cloudy everywhere and obeys bounds throughout the frame.',
              '  Individual layer minima do not bound the combined stack.',
              '- Different source-field sizes may still change original texture normalization',
              '  and coverage quantiles; only crops of the same generated field share the map.', '',
              '## Reproduce', '', 'From the repository root:', '', '```bash',
              'PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\',
              '  TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \\',
              '  python3 benchmarks/clouds/compare.py', '```', '',
              'Optionally supply `--junit` and `--targeted-junit` with freshly generated',
              'pytest XML files to include test counts and evidence hashes. The baseline',
              'Git history must be available. `results.json` contains exact configs, source',
              'hashes, raw measurements and the parameter sweep. See',
              '[usage](../../docs/usage.md#cloud-transmission-variation-v0270) for configuration.', '']
    (output / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
