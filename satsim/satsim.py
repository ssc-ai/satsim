from __future__ import division, print_function, absolute_import

import os
import logging
import copy
from datetime import datetime
import pickle
import json
import math
from time import sleep
from collections import OrderedDict
import numbers

import pydash
import tensorflow as tf
import numpy as np
from astropy import units as u
from sgp4.api import SatrecArray

from satsim.math import signal_to_noise_ratio, mean_degrees, diff_degrees, interp_degrees
from satsim.geometry.transform import rotate_and_translate, apply_wrap_around
from satsim.geometry.sprite import load_sprite_from_file
from satsim.image.fpa import analog_to_digital, mv_to_pe, pe_to_mv, add_patch, crop
from satsim.image.psf import gen_gaussian, eod_to_sigma, gen_from_poppy_configuration
from satsim.image.noise import add_photon_noise, add_read_noise
from satsim.image.render import render_piecewise, render_full
from satsim.geometry.draw import gen_line, gen_line_from_endpoints, gen_curve_from_points
from satsim.geometry.random import gen_random_points
from satsim.geometry.sstr7 import query_by_los
from satsim.geometry.csvsc import query_by_los as csvsc_query_by_los
from satsim.geometry.sgp4 import (
    create_satrec,
    create_sgp4,
    create_sgp4_from_satrec,
    batch_sgp4_position_gcrs_km,
)
from satsim.geometry.ephemeris import create_ephemeris_object
from satsim.geometry.astrometric import (
    create_topocentric,
    gen_track,
    get_los,
    get_los_azel,
    get_analytical_los,
    load_earth,
    optimized_angle_from_los_cosine,
    radec_to_eci,
)
from satsim.geometry.greatcircle import GreatCircle
from skyfield.toposlib import _ltude
from satsim.geometry.photometric import model_to_mv
from satsim.geometry.shadow import earth_shadow_umbra_mask
from satsim.geometry.twobody import create_twobody
from satsim.geometry.observation import create_observation
from satsim.io.satnet import write_frame, write_annotation, set_frame_annotation, init_annotation
from satsim.io.image import save_apng
from satsim.io.czml import save_czml
from satsim.util import tic, toc, MultithreadedTaskQueue, configure_eager, configure_single_gpu, merge_dicts, Profiler
from satsim.geometry import analytic_obs
from satsim.io import analytical
from satsim.config import transform, save_debug, _transform, save_cache
from satsim.pipeline import _delta_t, _avg_t
from satsim import time

logger = logging.getLogger(__name__)


def gen_multi(ssp, eager=True, output_dir='./', input_dir='./', device=None, memory=None, pid=0, output_debug=False, folder_name=None):
    """Generates multiple sets of images. Number of sets is based on the
    parameters `ssp['sim']['samples']`.

    Examples::

        # load a template json file
        ssp = load_json('input/config.json')

        # edit some parameters
        ssp['sim']['samples'] = 50
        ssp['geometry']['obs']['list']['mv'] = 17.5

        # generate SatNet files to the output directory
        gen_multi(ssp, eager=True, output_dir='output/')

    Args:
        ssp: `dict`, static or dynamic satsim configuration and parameters.
        eager: `boolean`, Has no effect. `True` only.
        output_dir: `str`, output directory to save SatNet files.
        input_dir: `str`, typically the input directory of the configuration file.
        device: `array`, array of GPU device IDs to enable. If `None`, enable all.
        pid: `int`, an ID to associate this instance to.
        output_debug: `boolean`, output intermediate debug files.
        folder_name: `str`, Optional name for folder to save files to.
    """
    if(eager):
        configure_eager()

    if device is not None:
        logger.info('Starting process id {} on GPU {} with {} MB.'.format(pid, device, memory if memory is not None else 'MAX'))
        configure_single_gpu(device, memory)

    queue = MultithreadedTaskQueue()

    n = ssp['sim']['samples'] if 'samples' in ssp['sim'] else 1

    for set_num in range(n):

        tic('gen_set', set_num)

        logger.info('Generating set {} of {} on process {}.'.format(set_num + 1, n, pid))

        # transform the original satsim parameters (eval any random sampling)
        # do a deep copy to preserve the original configuration
        tssp, issp = transform(copy.deepcopy(ssp), input_dir, with_debug=True)

        # run the transformed parameters
        dir_name = gen_images(tssp, eager, output_dir, set_num, queue=queue, output_debug=output_debug, set_name=folder_name)

        # save intermediate transforms
        save_debug(issp, dir_name)

        logger.info('Finished set {} of {} in {} sec on process {}.'.format(set_num + 1, n, toc('gen_set', set_num), pid))

    queue.stop()
    queue.waitUntilEmpty()
    logger.info('SatSim process {} exiting.'.format(pid))


def gen_images(ssp, eager=True, output_dir='./', sample_num=0, output_debug=False, queue=None, set_name=None):
    """Generates a single set of images.

    Examples::

        # load a template json file
        ssp = load_json('input/config.json')
        ssp = transform(copy.deepcopy(ssp), 'input/')

        # generate SatNet files to the output directory
        gen_images(ssp, eager=True, output_dir='output/')

    Args:
        ssp: `dict`, static satsim configuration and parameters. Any dynamic
            parameters should already be transformed.
        eager: `boolean`, Has no effect. `True` only.
        output_dir: `str`, output directory to save SatNet files.
        sample_num: `int`, recorded to annotation files.
        output_debug: `boolean`, output intermediate debug files.
        queue: `MultithreadedTaskQueue`, if not None, files will be written.
        set_name: `str`, sets the directory name to save the images to, if None, is set to current time.

    Returns:
        A `str`, directory to where the output files are saved.
    """
    ssp_orig = copy.deepcopy(ssp)

    (num_frames, t_exposure, h_fpa_os, w_fpa_os, s_osf, y_ifov, x_ifov, a2d_dtype, height, width) = _parse_sensor_params(ssp)

    dt = datetime.now()
    if set_name is None:
        dir_name = os.path.join(output_dir, dt.isoformat().replace(':','-'))
        set_name = 'sat_{:05d}'.format(sample_num)
    else:
        dir_name = os.path.join(output_dir, set_name)

    # make output dirs
    os.makedirs(dir_name, exist_ok=True)
    dir_debug = os.path.join(dir_name,'Debug')
    if output_debug and not os.path.exists(dir_debug):
        os.makedirs(dir_debug, exist_ok=True)

    # init annotations
    meta_data = init_annotation(
        'dir.name',
        ['{}.{:04d}.json'.format(set_name,x) for x in range(num_frames)],
        height,
        width,
        y_ifov,
        x_ifov)

    astrometrics_list = []
    for fpa_digital, frame_num, astrometrics, obs_os_pix, fpa_conv_star, fpa_conv_targ, bg_tf, dc_tf, rn_tf, num_shot_noise_samples, obs_cache, ground_truth, star_os_pix, segmentation in image_generator(ssp, dir_name, output_debug, dir_debug, with_meta=True, num_sets=1):
        astrometrics_list.append(astrometrics)
        if fpa_digital is not None:
            snr = signal_to_noise_ratio(fpa_conv_targ, fpa_conv_star + bg_tf + dc_tf, rn_tf)
            if num_shot_noise_samples is not None:
                snr = snr * np.sqrt(num_shot_noise_samples)
            meta_data = set_frame_annotation(meta_data, frame_num, h_fpa_os, w_fpa_os, obs_os_pix, [20 * s_osf, 20 * s_osf], snr=snr, star_os_pix=star_os_pix, metadata=astrometrics)
            if queue is not None:
                queue.task(write_frame, {
                    'dir_name': dir_name,
                    'sat_name': set_name,
                    'fpa_digital': fpa_digital.numpy(),
                    'meta_data': copy.deepcopy(meta_data),
                    'frame_num': frame_num,
                    'exposure_time': t_exposure,
                    'time_stamp': dt,
                    'ssp': ssp_orig,
                    'show_obs_boxes': ssp['sim']['show_obs_boxes'],
                    'show_star_boxes': ssp['sim']['show_star_boxes'],
                    'astrometrics': astrometrics,
                    'save_pickle': ssp['sim']['save_pickle'],
                    'dtype': a2d_dtype,
                    'save_jpeg': ssp['sim']['save_jpeg'],
                    'ground_truth': ground_truth,
                    'segmentation': segmentation,
                }, tag=dir_name)
            if output_debug:
                with open(os.path.join(dir_debug, 'metadata_{}.json'.format(frame_num)), 'w') as jsonfile:
                    json.dump(meta_data, jsonfile, indent=2, default=str)

            logger.debug('Finished frame {} of {} in {} sec.'.format(frame_num + 1, num_frames, toc('gen_frame', frame_num)))
        else:
            if ssp['sim'].get('analytical_obs', False):
                meta_data = set_frame_annotation(
                    meta_data,
                    frame_num,
                    h_fpa_os,
                    w_fpa_os,
                    obs_os_pix,
                    [20 * s_osf, 20 * s_osf],
                    snr=None,
                    star_os_pix=None,
                    metadata=astrometrics,
                )
                if queue is not None:
                    queue.task(
                        write_annotation,
                        {
                            'dir_name': dir_name,
                            'sat_name': set_name,
                            'meta_data': copy.deepcopy(meta_data),
                            'frame_num': frame_num,
                            'ssp': ssp_orig,
                            'save_pickle': ssp['sim']['save_pickle'],
                        },
                        tag=dir_name,
                    )
            else:
                if queue is not None:
                    def f():
                        return None
                    queue.task(f, {}, tag=dir_name)
                logger.debug('Render mode off. Skipping frame generation.')

    # write movie
    def wait_and_run():
        while True:
            if queue.has_tag(dir_name):
                sleep(0.1)
            else:
                if ssp['sim']['save_movie']:
                    logger.debug('Saving PNG movie.')
                    save_apng(dirname=os.path.join(dir_name,'AnnotatedImages'), filename='movie.png')
                if ssp['sim']['save_czml']:
                    logger.debug('Saving CZML.')
                    save_czml(ssp, obs_cache, astrometrics_list, os.path.join(dir_name,'satsim.czml'))
                return

    if queue is not None:
        queue.task(wait_and_run, {})

    return dir_name


def _parse_sensor_params(ssp):
    """ TODO Temporary.
    """
    s_osf = ssp['sim']['spacial_osf']

    y_ifov = ssp['fpa']['y_fov'] / ssp['fpa']['height']
    x_ifov = ssp['fpa']['x_fov'] / ssp['fpa']['width']

    if 'crop' in ssp['fpa']:
        cp = ssp['fpa']['crop']
        h_fpa_os = cp['height'] * s_osf
        w_fpa_os = cp['width'] * s_osf
        height = cp['height']
        width = cp['width']
    else:
        h_fpa_os = ssp['fpa']['height'] * s_osf
        w_fpa_os = ssp['fpa']['width'] * s_osf
        height = ssp['fpa']['height']
        width = ssp['fpa']['width']

    t_exposure = ssp['fpa']['time']['exposure']

    num_frames = ssp['fpa']['num_frames']

    if 'dtype' in ssp['fpa']['a2d']:
        a2d_dtype = ssp['fpa']['a2d']['dtype']
    else:
        a2d_dtype = 'uint16'

    return num_frames, t_exposure, h_fpa_os, w_fpa_os, s_osf, y_ifov, x_ifov, a2d_dtype, height, width


def image_generator(ssp, output_dir='.', output_debug=False, dir_debug='./Debug', with_meta=False, eager=True, num_sets=0):
    """Generator function for a single set of images.

    Examples::

        # load a template json file
        ssp = load_json('input/config.json')
        ssp = transform(copy.deepcopy(ssp), 'input/')

        # generate SatNet files to the output directory
        for results in image_generator(ssp, eager=True, output_dir='output/'):
            imshow(results.numpy())

    Args:
        ssp: `dict`, static satsim configuration and parameters. Any dynamic
            parameters should already be transformed.
        output_dir: `str`, root directory for SatSim configuration file.
        sample_num: `int`, recorded to annotation files.
        output_debug: `boolean`, output intermediate debug files.
        dir_debug: `str`, directory to output debug files.
        with_meta: `boolean`, add metadata to the return function.
        eager: `boolean`, Has no effect. `True` only.
        num_sets: `int`, number of sets until generator exits. Set to 0 to run forever.

    Returns:
        An `image`, the rendered image. If `with_meta` is set to `True`, additional values are returned:
            fpa_digital: `image`, rendered image.
            frame_num: `int`, frame number.
            astrometrics: `dict`, astrometry meta data.
            obs_os_pix: `dict`, observations meta data.
            fpa_conv_star: `image`, pristine image with stars.
            fpa_conv_targ: `image`, pristine image with targets.
            bg_tf: `image`, background image.
            dc_tf: `image`, dark current image.
            rn_tf: `image`, read noise image.
            num_shot_noise_samples: `int`, number of shot noise samples averaged
    """
    tic('init')
    logger.debug('Initializing variables.')

    # evaluate any python pipelines
    ssp = _transform(ssp, output_dir, False, True)

    astrometrics = {}

    h = ssp['fpa']['height']
    w = ssp['fpa']['width']

    s_osf = ssp['sim']['spacial_osf']
    t_osf = ssp['sim']['temporal_osf']

    h_pad_os = 2 * ssp['sim']['padding'] * s_osf
    w_pad_os = 2 * ssp['sim']['padding'] * s_osf

    h_pad_os_div2 = ssp['sim']['padding'] * s_osf
    w_pad_os_div2 = ssp['sim']['padding'] * s_osf

    y_ifov = ssp['fpa']['y_fov'] / ssp['fpa']['height']
    x_ifov = ssp['fpa']['x_fov'] / ssp['fpa']['width']

    y_ifov_os = ssp['fpa']['y_fov'] / ssp['fpa']['height'] / s_osf
    x_ifov_os = ssp['fpa']['x_fov'] / ssp['fpa']['width'] / s_osf

    h_fpa_os = ssp['fpa']['height'] * s_osf
    w_fpa_os = ssp['fpa']['width'] * s_osf

    h_fpa_pad_os = h_fpa_os + h_pad_os
    w_fpa_pad_os = w_fpa_os + w_pad_os

    y_fov = h_fpa_os * y_ifov_os
    x_fov = w_fpa_os * x_ifov_os

    y_fov_pad = h_fpa_pad_os * y_ifov_os
    x_fov_pad = w_fpa_pad_os * x_ifov_os

    if 'render_size' in ssp['sim']:
        h_sub = ssp['sim']['render_size'][0]
        w_sub = ssp['sim']['render_size'][1]
        render_mode = 'piecewise'
    else:
        h_sub = h
        w_sub = w
        render_mode = 'full'

    h_sub_os = h_sub * s_osf
    w_sub_os = w_sub * s_osf
    h_sub_pad_os = h_sub_os + h_pad_os
    w_sub_pad_os = w_sub_os + w_pad_os

    t_exposure = ssp['fpa']['time']['exposure']
    t_frame = ssp['fpa']['time']['gap'] + t_exposure

    num_frames = ssp['fpa']['num_frames']
    astrometrics['num_frames'] = num_frames

    zeropoint = ssp['fpa']['zeropoint']

    if 'flip_up_down' not in ssp['fpa']:
        ssp['fpa']['flip_up_down'] = False

    if 'flip_left_right' not in ssp['fpa']:
        ssp['fpa']['flip_left_right'] = False

    if 'detection' not in ssp['fpa']:
        ssp['fpa']['detection'] = {
            'snr_threshold': 0.0,
            'pixel_error': 0.0,
            'false_alarm_rate': 0.0,
            'max_false': 10,
        }
    else:
        ssp['fpa']['detection'].setdefault('snr_threshold', 0.0)
        ssp['fpa']['detection'].setdefault('pixel_error', 0.0)
        ssp['fpa']['detection'].setdefault('false_alarm_rate', 0.0)
        ssp['fpa']['detection'].setdefault('max_false', 10)

    star_mode = ssp['geometry']['stars']['mode']

    # TODO move defaults to a different file
    if 'velocity_units' in ssp['sim']:
        if ssp['sim']['velocity_units'] == 'arcsec':
            y_to_pix = y_ifov * 3600
            x_to_pix = x_ifov * 3600
    else:
        y_to_pix = 1.0
        x_to_pix = 1.0

    if 'star_render_mode' not in ssp['sim']:
        ssp['sim']['star_render_mode'] = 'transform'

    if 'show_obs_boxes' not in ssp['sim']:
        ssp['sim']['show_obs_boxes'] = True

    if 'show_star_boxes' not in ssp['sim']:
        ssp['sim']['show_star_boxes'] = False

    if 'enable_shot_noise' not in ssp['sim']:
        ssp['sim']['enable_shot_noise'] = True

    if 'num_shot_noise_samples' not in ssp['sim']:
        ssp['sim']['num_shot_noise_samples'] = None

    if 'star_catalog_query_mode' not in ssp['sim']:
        ssp['sim']['star_catalog_query_mode'] = 'frame'

    if 'apply_star_wrap_around' not in ssp['sim']:
        ssp['sim']['apply_star_wrap_around'] = False

    if 'num_target_samples' not in ssp['sim']:
        ssp['sim']['num_target_samples'] = 3

    if 'calculate_snr' not in ssp['sim']:
        ssp['sim']['calculate_snr'] = True

    if 'save_jpeg' not in ssp['sim']:
        ssp['sim']['save_jpeg'] = True

    if 'save_movie' not in ssp['sim']:
        ssp['sim']['save_movie'] = ssp['sim']['save_jpeg']

    if 'save_czml' not in ssp['sim']:
        ssp['sim']['save_czml'] = True

    if 'save_ground_truth' not in ssp['sim']:
        ssp['sim']['save_ground_truth'] = False

    if 'star_annotation_threshold' not in ssp['sim']:
        ssp['sim']['star_annotation_threshold'] = False

    if 'save_segmentation' not in ssp['sim']:
        ssp['sim']['save_segmentation'] = False

    if 'mode' not in ssp['sim']:
        ssp['sim']['mode'] = 'fftconv2p'

    if 'save_pickle' not in ssp['sim']:
        ssp['sim']['save_pickle'] = False

    if 'analytical_obs' not in ssp['sim']:
        ssp['sim']['analytical_obs'] = False

    if 'analytical_obs_frame' not in ssp['sim']:
        ssp['sim']['analytical_obs_frame'] = 'geocentric'

    if 'psf_sample_frequency' not in ssp['sim']:
        ssp['sim']['psf_sample_frequency'] = 'once'

    if 'enable_deflection' not in ssp['sim']:
        ssp['sim']['enable_deflection'] = False

    if 'enable_light_transit' not in ssp['sim']:
        ssp['sim']['enable_light_transit'] = True

    if 'enable_stellar_aberration' not in ssp['sim']:
        ssp['sim']['enable_stellar_aberration'] = True

    if 'enable_fov_filter' not in ssp['sim']:
        ssp['sim']['enable_fov_filter'] = False

    if 'fov_filter_radius' not in ssp['sim']:
        ssp['sim']['fov_filter_radius'] = None

    if star_mode == 'bins':
        star_dn = ssp['geometry']['stars']['mv']['density']
        star_pe = mv_to_pe(zeropoint, ssp['geometry']['stars']['mv']['bins']) * t_exposure
    elif star_mode == 'sstr7' or star_mode == 'csv':
        star_ra = ssp['geometry']['stars']['ra'] if 'ra' in ssp['geometry']['stars'] else 0
        star_dec = ssp['geometry']['stars']['dec'] if 'dec' in ssp['geometry']['stars'] else 0
        star_rot = ssp['geometry']['stars']['rotation'] if 'rotation' in ssp['geometry']['stars'] else 0
        star_path = ssp['geometry']['stars']['path']

    if ssp['geometry']['stars']['motion']['mode'] == 'affine':
        star_rot_rate = ssp['geometry']['stars']['motion']['rotation']
        star_tran_os = [ssp['geometry']['stars']['motion']['translation'][0],ssp['geometry']['stars']['motion']['translation'][1]]
        star_tran_os[0] = star_tran_os[0] / y_to_pix * s_osf
        star_tran_os[1] = star_tran_os[1] / x_to_pix * s_osf
    elif ssp['geometry']['stars']['motion']['mode'] == 'affine-polar':
        star_rot_rate = ssp['geometry']['stars']['motion']['rotation']
        star_vel_angle = ssp['geometry']['stars']['motion']['translation'][0] * math.pi / 180
        star_vel_mag = ssp['geometry']['stars']['motion']['translation'][1]
        star_tran_os = [star_vel_mag * math.sin(star_vel_angle), star_vel_mag * math.cos(star_vel_angle)]
        star_tran_os[0] = star_tran_os[0] / y_to_pix * s_osf
        star_tran_os[1] = star_tran_os[1] / x_to_pix * s_osf
    elif ssp['geometry']['stars']['motion']['mode'] == 'none':
        star_rot_rate = 0.0
        star_tran_os = [0.0, 0.0]

    # todo make this auto
    if 'pad' in ssp['geometry']['stars']:
        star_pad = ssp['geometry']['stars']['pad']
    else:
        star_pad = 1.2
    star_bounds = [[-star_pad * h_fpa_pad_os, (star_pad + 1) * h_fpa_pad_os], [-star_pad * w_fpa_pad_os, (star_pad + 1) * w_fpa_pad_os], [0, 0]]

    obs = ssp['geometry']['obs']['list']

    bg = mv_to_pe(zeropoint, ssp['background']['galactic']) * (y_ifov * 3600 * x_ifov * 3600) * t_exposure

    if 'stray' in ssp['background']:
        if isinstance(ssp['background']['stray'], dict) and 'mode' in ssp['background']['stray'] and ssp['background']['stray']['mode'] == 'none':
            pass  # for backward compat
        else:
            bg = bg + ssp['background']['stray'] * t_exposure

    dc = ssp['fpa']['dark_current'] * t_exposure
    rn = ssp['fpa']['noise']['read']
    en = ssp['fpa']['noise']['electronic']
    gain = ssp['fpa']['gain']
    bias = ssp['fpa']['bias']
    a2d_fwc = ssp['fpa']['a2d']['fwc']
    a2d_gain = ssp['fpa']['a2d']['gain']
    a2d_bias = ssp['fpa']['a2d']['bias']

    if 'dtype' in ssp['fpa']['a2d']:
        a2d_dtype = ssp['fpa']['a2d']['dtype']
    else:
        a2d_dtype = 'uint16'

    astrometrics['bias'] = a2d_bias

    pipeline_profiler = Profiler.from_sim(ssp['sim'], logger)

    # gen psf
    if ssp['sim']['psf_sample_frequency'] == 'once':
        psf_profiler = pipeline_profiler.child('PSF generation (once)')
        with psf_profiler.time('total'):
            psf_os = _gen_psf(ssp, h_sub_pad_os, w_sub_pad_os, y_ifov, x_ifov, s_osf)
        psf_profiler.log(order_times=['total'])

    set_number = 0
    while num_sets <= 0 or set_number < num_sets:
        set_number = set_number + 1

        # time
        if 'time' in ssp['geometry']:
            tt = ssp['geometry']['time']
        else:
            tt = [2020, 1, 1, 0, 0, 0.0]

        ts_collect_start = time.utc_from_list(tt)
        ts_frame_end = time.utc_from_list(tt, t_exposure)
        ts_collect_end = time.utc_from_list(tt, t_frame * num_frames)

        # site
        site_mode = None
        if 'site' in ssp['geometry']:

            # note: stars will track horizontally where zenith is pointed up. focal plane rotation is simulated with the `rotation` variable
            star_rot = ssp['geometry']['site']['gimbal']['rotation']
            track_mode = ssp['geometry']['site']['track']['mode']
            astrometrics['track_mode'] = _parse_track_mode(track_mode, 0, num_frames)

            if 'tle' in ssp['geometry']['site']:
                observer = create_sgp4(ssp['geometry']['site']['tle'][0], ssp['geometry']['site']['tle'][1])
                site_mode = 'space'
            elif 'tle1' in ssp['geometry']['site']:
                observer = create_sgp4(ssp['geometry']['site']['tle1'], ssp['geometry']['site']['tle2'])
                site_mode = 'space'
            else:
                lat = ssp['geometry']['site']['lat']
                lon = ssp['geometry']['site']['lon']
                astrometrics['lat'] = float(_ltude(lat, 'latitude', 'N', 'S')) if isinstance(lat, str) else float(lat)
                astrometrics['lon'] = float(_ltude(lon, 'longitude', 'E', 'W')) if isinstance(lon, str) else float(lon)
                astrometrics['alt'] = float(ssp['geometry']['site'].get('alt', 0))
                observer = create_topocentric(astrometrics['lat'], astrometrics['lon'], astrometrics['alt'])
                site_mode = 'ground'

            if 'tle' in ssp['geometry']['site']['track']:
                track = create_sgp4(ssp['geometry']['site']['track']['tle'][0], ssp['geometry']['site']['track']['tle'][1])
            elif 'tle1' in ssp['geometry']['site']['track']:
                track = create_sgp4(ssp['geometry']['site']['track']['tle1'], ssp['geometry']['site']['track']['tle2'])
            elif 'epoch' in ssp['geometry']['site']['track']:
                track_epoch = time.utc_from_list_or_scalar(ssp['geometry']['site']['track']['epoch'])
                track = create_twobody(np.array(ssp['geometry']['site']['track']['position']) * u.km, np.array(ssp['geometry']['site']['track']['velocity']) * u.km / u.s, track_epoch)
            else:
                track = None

            astrometrics['object'] = pydash.objects.get(ssp, 'geometry.site.track.name', '')
            track_az = pydash.objects.get(ssp, 'geometry.site.track.az', 0)
            track_el = pydash.objects.get(ssp, 'geometry.site.track.el', 0)

            if type(track_az) is not list:
                track_az = [track_az, track_az]

            if type(track_el) is not list:
                track_el = [track_el, track_el]

            # deprecated
            if 'time' in ssp['geometry']['site']:
                logger.warning('geometry.site.time is deprecated. Use geometry.time instead.')
                tt = ssp['geometry']['site']['time']
                ts_collect_start = time.utc_from_list(tt)
                ts_frame_end = time.utc_from_list(tt, t_exposure)
                ts_collect_end = time.utc_from_list(tt, t_frame * num_frames)

            star_ra, star_dec, star_tran_os, star_rot_rate = _calculate_star_position_and_motion(ssp, astrometrics,
                                                                                                 ts_collect_start, ts_collect_end, ts_collect_start, ts_frame_end, t_exposure,
                                                                                                 h_fpa_pad_os, w_fpa_pad_os,
                                                                                                 y_fov_pad, x_fov_pad,
                                                                                                 y_fov, x_fov,
                                                                                                 y_ifov, x_ifov,
                                                                                                 observer, track, star_rot, astrometrics['track_mode'], track_az, track_el)
        else:
            observer = None
            track = None
            star_rot = None
            track_mode = None
            track_az = None
            track_el = None
            ssp['sim']['star_catalog_query_mode'] = 'at_start'

        if ssp['sim']['temporal_osf'] == 'auto':
            if star_mode != 'none':
                rrr, ccc = rotate_and_translate(h_fpa_pad_os, w_fpa_pad_os, [0,h_fpa_pad_os], [0,w_fpa_pad_os], t_exposure, star_rot_rate, star_tran_os)
                rrr -= [0, h_fpa_pad_os]
                ccc -= [0, w_fpa_pad_os]
                t_osf = tf.cast(max([tf.sqrt(rrr[0] * rrr[0] + ccc[0] * ccc[0]), tf.sqrt(rrr[1] * rrr[1] + ccc[1] * ccc[1])]) + 1, tf.int32)
                logger.debug('Auto temporal oversample factor set to {}.'.format(t_osf.numpy()))
            else:
                t_osf = 1

        # gen stars
        if star_mode == 'bins':
            r_stars_os, c_stars_os, pe_stars_os = gen_random_points(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, star_pe, star_dn, pad_mult=star_pad)
            ra_stars = np.array(range(len(r_stars_os)))
            dec_stars = np.zeros(len(r_stars_os))
            m_stars_os = pe_to_mv(zeropoint, pe_stars_os / t_exposure)
        else:
            r_stars_os, c_stars_os, m_stars_os = [], [], []
            pe_stars_os = []

        # gen psf
        if ssp['sim']['psf_sample_frequency'] == 'collect':
            psf_profiler = pipeline_profiler.child('PSF generation (collect)')
            with psf_profiler.time('total'):
                psf_os = _gen_psf(ssp, h_sub_pad_os, w_sub_pad_os, y_ifov, x_ifov, s_osf)
            psf_profiler.log(order_times=['total'])

        if pydash.objects.has(ssp, 'augment.background.stray'):
            bg = ssp['augment']['background']['stray'](bg)
            bg = tf.cast(bg, tf.float32)
            bg = tf.where(tf.math.is_nan(bg), tf.zeros_like(bg), bg)

        logger.debug('Exposure time {}.'.format(t_exposure))
        logger.debug('Background pe/pix {}.'.format(np.mean(bg)))
        logger.debug('Finished initializing variables in {} sec.'.format(toc('init')))

        star_uid_map = {}
        obs_cache = [None] * len(obs)
        batch_cache = {}
        gain_tf = tf.cast(gain, tf.float32)
        bg_tf = tf.cast(bg, tf.float32)
        dc_tf = tf.cast(dc, tf.float32)
        bias_tf = tf.cast(bias, tf.float32)
        _rn = tf.cast(rn, tf.float32)
        _en = tf.cast(en, tf.float32)
        rn_tf = tf.math.sqrt(_rn * _rn + _en * _en)

        # gen frame and yield
        for frame_num in range(num_frames):
            frame_profiler = pipeline_profiler.child('Frame {} profile'.format(frame_num + 1))
            frame_total = frame_profiler.start('total')
            frame_time_order = [
                'psf',
                'objects',
                'track',
                'star_query',
                'star_wrap',
                'render',
                'noise',
                'a2d',
                'segmentation',
                'crop',
                'analytical',
                'total',
            ]
            tic('gen_frame', frame_num)
            logger.debug('Generating frame {} of {}.'.format(frame_num + 1, num_frames))
            astrometrics['frame_num'] = frame_num + 1
            astrometrics['track_mode'] = _parse_track_mode(track_mode, frame_num, num_frames)

            if ssp['sim']['psf_sample_frequency'] == 'frame':
                with frame_profiler.time('psf'):
                    psf_os = _gen_psf(ssp, h_sub_pad_os, w_sub_pad_os, y_ifov, x_ifov, s_osf)

            t_start = frame_num * t_frame
            t_end = t_start + t_exposure
            ts_start = time.utc_from_list(tt, t_start)
            ts_end = time.utc_from_list(tt, t_end)
            t_start_star = t_start
            t_end_star = t_end
            t_frame_track_start = _parse_start_track_time(track_mode, frame_num, num_frames, ts_collect_start, ts_start)

            # sensor position and velocity
            if site_mode == 'space':
                ts_mid = time.mid(ts_start, ts_end)
                eci_sv = (observer - load_earth()).at(ts_mid)
                pos = eci_sv.position.km
                vel = eci_sv.velocity.km_per_s
                astrometrics['x'] = float(pos[0])
                astrometrics['y'] = float(pos[1])
                astrometrics['z'] = float(pos[2])
                astrometrics['vx'] = float(vel[0])
                astrometrics['vy'] = float(vel[1])
                astrometrics['vz'] = float(vel[2])

            # calculate object pixels
            with frame_profiler.time('objects'):
                r_obs_os, c_obs_os, pe_obs_os, obs_os_pix, obs_model = _gen_objects(ssp, render_mode,
                                                                                    obs, obs_cache, batch_cache,
                                                                                    t_frame_track_start, ts_collect_end, t_start, t_end, tt,
                                                                                    observer, track, track_az, track_el,
                                                                                    zeropoint, s_osf,
                                                                                    h_fpa_os, w_fpa_os,
                                                                                    h_pad_os, w_pad_os,
                                                                                    h_pad_os_div2, w_pad_os_div2,
                                                                                    h_fpa_pad_os, w_fpa_pad_os,
                                                                                    y_fov, x_fov,
                                                                                    y_to_pix, x_to_pix,
                                                                                    star_rot, astrometrics['track_mode'])
            logger.debug('Number of objects {}.'.format(len(obs_os_pix)))

            if track_mode is not None:
                with frame_profiler.time('track'):
                    star_ra, star_dec, star_tran_os, star_rot_rate = _calculate_star_position_and_motion(ssp, astrometrics,
                                                                                                         t_frame_track_start, ts_collect_end, ts_start, ts_end, t_exposure,
                                                                                                         h_fpa_pad_os, w_fpa_pad_os,
                                                                                                         y_fov_pad, x_fov_pad,
                                                                                                         y_fov, x_fov,
                                                                                                         y_ifov, x_ifov,
                                                                                                         observer, track, star_rot, astrometrics['track_mode'], track_az, track_el)

            # if image rendering is disabled, optionally generate analytical observations and return
            if ssp['sim']['mode'] == 'none':
                if ssp['sim'].get('analytical_obs', False):
                    if 'crop' in ssp['fpa']:
                        cp = ssp['fpa']['crop']
                        for opp in obs_os_pix:
                            opp['rr'] = opp['rr'] - cp['height_offset'] * s_osf
                            opp['cc'] = opp['cc'] - cp['width_offset'] * s_osf
                            opp['rrr'] = opp['rrr'] - cp['height_offset']
                            opp['rcc'] = opp['rcc'] - cp['width_offset']

                    bg_val = float(tf.reduce_mean(bg_tf).numpy()) if hasattr(bg_tf, 'numpy') else float(np.mean(bg_tf))
                    dc_val = float(tf.reduce_mean(dc_tf).numpy()) if hasattr(dc_tf, 'numpy') else float(np.mean(dc_tf))
                    rn_val = float(rn)
                    en_val = float(en)
                    obs_list = analytic_obs.generate(
                        ssp,
                        obs_os_pix,
                        astrometrics,
                        bg_val,
                        dc_val,
                        rn_val,
                        en_val,
                    )
                    analytical.save(output_dir, frame_num, obs_list)
                frame_profiler.stop('total', frame_total)
                frame_profiler.log(order_times=frame_time_order)
                if with_meta:
                    yield None, frame_num, astrometrics.copy(), obs_os_pix, None, None, None, None, None, None, obs_cache, None, None, None
                else:
                    yield None
                continue

            # refresh catalog stars
            if (star_mode == 'sstr7' or star_mode == 'csv') and (ssp['sim']['star_catalog_query_mode'] == 'frame' or frame_num == 0):
                with frame_profiler.time('star_query'):
                    if star_mode == 'sstr7':
                        # note star_ra and star_dec are apparent positions which includes stellar aberration
                        r_stars_os, c_stars_os, m_stars_os, ra_stars, dec_stars = query_by_los(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, star_ra, star_dec, rot=star_rot, rootPath=star_path, pad_mult=star_pad, flipud=ssp['fpa']['flip_up_down'], fliplr=ssp['fpa']['flip_left_right'])
                    elif star_mode == 'csv':
                        r_stars_os, c_stars_os, m_stars_os, ra_stars, dec_stars = csvsc_query_by_los(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, star_ra, star_dec, rot=star_rot, rootPath=star_path, flipud=ssp['fpa']['flip_up_down'], fliplr=ssp['fpa']['flip_left_right'])

                    pe_stars_os = mv_to_pe(zeropoint, m_stars_os) * t_exposure
                    t_start_star = 0.0
                    t_end_star = t_exposure
                    if ssp['sim']['star_catalog_query_mode'] == 'frame':
                        ssp['sim']['apply_star_wrap_around'] = False

            # wrap stars around
            if ssp['sim']['apply_star_wrap_around']:
                with frame_profiler.time('star_wrap'):
                    r_stars_os, c_stars_os, star_bounds = apply_wrap_around(h_fpa_pad_os, w_fpa_pad_os, r_stars_os, c_stars_os, t_start, t_end, star_rot_rate, star_tran_os, star_bounds)

            logger.debug('Number of stars {}.'.format(len(r_stars_os)))

            t_start_star = tf.cast(t_start_star, tf.float32)
            t_end_star = tf.cast(t_end_star, tf.float32)
            r_stars_os = tf.cast(r_stars_os, tf.float32)
            c_stars_os = tf.cast(c_stars_os, tf.float32)
            pe_stars_os = tf.cast(pe_stars_os, tf.float32)

            r_obs_os = tf.cast(r_obs_os, tf.float32)
            c_obs_os = tf.cast(c_obs_os, tf.float32)
            pe_obs_os = tf.cast(pe_obs_os, tf.float32)

            # augment TODO abstract this
            if pydash.objects.has(ssp, 'augment.fpa.psf'):
                psf_os_curr = ssp['augment']['fpa']['psf'](psf_os)
                psf_os_curr = tf.cast(psf_os_curr, tf.float32)
            else:
                psf_os_curr = psf_os

            # helper function for image rendering and segmentation rendering
            def _render(r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, render_separate):
                if render_mode == 'piecewise':
                    return render_piecewise(h, w, h_sub, w_sub, h_pad_os, w_pad_os, s_osf, psf_os_curr, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, star_render_mode=ssp['sim']['star_render_mode'])
                else:
                    return render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os_curr, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=ssp['sim']['star_render_mode'])

            # render
            with frame_profiler.time('render'):
                fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, fpa_conv_os, fpa_conv_crop = _render(r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, ssp['sim']['calculate_snr'])

            # add noise
            with frame_profiler.time('noise'):
                fpa_conv = (fpa_conv_star + fpa_conv_targ + bg_tf) * gain_tf + dc_tf
                if ssp['sim']['enable_shot_noise'] is True:
                    fpa_conv_noise = add_photon_noise(fpa_conv, ssp['sim']['num_shot_noise_samples'])
                else:
                    fpa_conv_noise = fpa_conv
                fpa, rn_gt = add_read_noise(fpa_conv_noise, rn, en)

            # analog to digital
            with frame_profiler.time('a2d'):
                fpa_digital = analog_to_digital(fpa + bias_tf, a2d_gain, a2d_fwc, a2d_bias, dtype=a2d_dtype)

            # augment TODO abstract this
            if pydash.objects.has(ssp, 'augment.image'):
                if ssp['augment']['image']['post'] is None:
                    pass
                elif callable(ssp['augment']['image']['post']):
                    fpa_digital = ssp['augment']['image']['post'](fpa_digital)
                else:
                    fpa_digital = fpa_digital + ssp['augment']['image']['post']

            # save segmentation matadata
            segmentation = None
            seg_id_stars = None
            if ssp['sim']['save_segmentation']:
                with frame_profiler.time('segmentation'):
                    star_threshold = 50.0
                    if ssp['sim']['star_annotation_threshold'] is not False:
                        star_threshold = ssp['sim']['star_annotation_threshold']

                    if 'crop' in ssp['fpa']:
                        cp = ssp['fpa']['crop']
                        segmentation = {
                            'star_segmentation': np.zeros((cp['height'], cp['width']), dtype=np.uint16),
                            'object_segmentation': np.zeros((cp['height'], cp['width']), dtype=np.uint16)
                        }
                    else:
                        segmentation = {
                            'star_segmentation': np.zeros((h, w), dtype=np.uint16),
                            'object_segmentation': np.zeros((h, w), dtype=np.uint16)
                        }
                    seg_id_stars = np.zeros(len(r_stars_os))
                    min_dn = 1

                    # sort stars by brightness to ensure bright stars are rendered last
                    star_order = tf.argsort(pe_stars_os, direction='ASCENDING')
                    r_stars_os_sort = tf.gather(r_stars_os, star_order)
                    c_stars_os_sort = tf.gather(c_stars_os, star_order)
                    pe_stars_os_sort = tf.gather(pe_stars_os, star_order)
                    m_stars_os_sort = tf.gather(m_stars_os, star_order)
                    ra_stars_sort = tf.gather(ra_stars, star_order)
                    dec_stars_sort = tf.gather(dec_stars, star_order)

                    # for each bright star, create a segmentation mask
                    for i in range(len(ra_stars_sort)):
                        if m_stars_os_sort[i] < star_threshold:
                            fpa_segmentation, _, _, _, _ = _render([], [], [], [r_stars_os_sort[i]], [c_stars_os_sort[i]], [pe_stars_os_sort[i]], False)

                            if 'crop' in ssp['fpa']:
                                cp = ssp['fpa']['crop']
                                fpa_segmentation = crop(fpa_segmentation, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])

                            fpa_segmentation = analog_to_digital(fpa_segmentation, a2d_gain, a2d_fwc, 0, dtype=a2d_dtype)
                            fpa_segmentation = fpa_segmentation.numpy()
                            if np.max(fpa_segmentation.flatten()) > min_dn:
                                # need to assign a unique id to each star and keep track of it as they could pop in and out of the frame
                                star_uid = '{}, {}, {}'.format(ra_stars_sort[i], dec_stars_sort[i], m_stars_os_sort[i])
                                if star_uid in star_uid_map:
                                    star_id = star_uid_map[star_uid]
                                else:
                                    star_id = len(star_uid_map) + 1
                                    star_uid_map[star_uid] = star_id

                                segmentation['star_segmentation'][fpa_segmentation > min_dn] = star_id
                                seg_id_stars[star_order[i]] = star_id
                                logger.debug('Generated star segmentation {}, {} mv.'.format(star_id, m_stars_os_sort[i]))

                    # for each observation, create a segmentation mask
                    obs_os_pix_sorted = sorted(obs_os_pix, key=lambda k: k['pe'], reverse=False)
                    for ob in obs_os_pix_sorted:
                        fpa_segmentation, _, _, _, _ = _render(ob['rr'], ob['cc'], ob['pp'], [], [], [], False)
                        if 'crop' in ssp['fpa']:
                            cp = ssp['fpa']['crop']
                            fpa_segmentation = crop(fpa_segmentation, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])

                        fpa_segmentation = analog_to_digital(fpa_segmentation, a2d_gain, a2d_fwc, 0, dtype=a2d_dtype)
                        fpa_segmentation = fpa_segmentation.numpy()
                        if np.max(fpa_segmentation.flatten()) > min_dn:
                            segmentation['object_segmentation'][fpa_segmentation > min_dn] = ob['id']
                            logger.debug('Generated object segmentation {}, {} mv.'.format(ob['id'], ob['mv']))

            # cropped sensor
            crop_bg_tf = bg_tf
            crop_dc_tf = dc_tf
            crop_rn_tf = rn_tf
            crop_h_fpa_os = h_fpa_os
            crop_w_fpa_os = w_fpa_os
            crop_gain_tf = gain_tf
            crop_bias_tf = bias_tf
            if 'crop' in ssp['fpa']:
                cp = ssp['fpa']['crop']

                # crop images
                with frame_profiler.time('crop'):
                    fpa_digital = crop(fpa_digital, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    fpa_conv_targ = crop(fpa_conv_targ, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    fpa_conv_star = crop(fpa_conv_star, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    if len(bg_tf.shape) == 2:
                        crop_bg_tf = crop(bg_tf, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    if len(dc_tf.shape) == 2:
                        crop_dc_tf = crop(dc_tf, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    if len(rn_tf.shape) == 2:
                        crop_rn_tf = crop(rn_tf, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    fpa_conv_noise = crop(fpa_conv_noise, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    fpa_conv = crop(fpa_conv, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    rn_gt = crop(rn_gt, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    if len(gain_tf.shape) == 2:
                        crop_gain_tf = crop(gain_tf, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])
                    if len(bias_tf.shape) == 2:
                        crop_bias_tf = crop(bias_tf, cp['height_offset'], cp['width_offset'], cp['height'], cp['width'])

                # update star positions
                r_stars_os = r_stars_os - cp['height_offset'] * s_osf
                c_stars_os = c_stars_os - cp['width_offset'] * s_osf
                crop_h_fpa_os = cp['height'] * s_osf
                crop_w_fpa_os = cp['width'] * s_osf

                # update target positions
                for opp in obs_os_pix:
                    opp['rr'] = opp['rr'] - cp['height_offset'] * s_osf
                    opp['cc'] = opp['cc'] - cp['width_offset'] * s_osf
                    opp['rrr'] = opp['rrr'] - cp['height_offset']
                    opp['rcc'] = opp['rcc'] - cp['width_offset']

            if ssp['sim']['star_annotation_threshold'] is not False:
                pes_stars_os = pe_stars_os / t_exposure
                star_os_pix = {
                    'h': crop_h_fpa_os,
                    'w': crop_w_fpa_os,
                    'h_pad': h_pad_os_div2,
                    'w_pad': w_pad_os_div2,
                    'rr': r_stars_os,
                    'cc': c_stars_os,
                    'pe': pes_stars_os,
                    'mv': m_stars_os if m_stars_os is not None else pe_to_mv(zeropoint, pes_stars_os),
                    'ra': ra_stars,
                    'dec': dec_stars,
                    'seg_id': seg_id_stars,
                    't_start': t_start_star,
                    't_end': t_end_star,
                    'rot': star_rot_rate,
                    'tran': star_tran_os,
                    'min_mv': ssp['sim']['star_annotation_threshold'] if isinstance(ssp['sim']['star_annotation_threshold'], numbers.Number) else 15
                }
            else:
                star_os_pix = None

            if ssp['sim']['save_ground_truth']:
                ground_truth = OrderedDict()
                ground_truth['target_pe'] = fpa_conv_targ.numpy()
                ground_truth['star_pe'] = fpa_conv_star.numpy()
                ground_truth['background_pe'] = crop_bg_tf.numpy()
                ground_truth['dark_current_pe'] = crop_dc_tf.numpy()
                ground_truth['photon_noise_pe'] = (fpa_conv_noise - fpa_conv).numpy()
                ground_truth['read_noise_pe'] = rn_gt.numpy()
                ground_truth['gain'] = crop_gain_tf.numpy()
                ground_truth['bias_pe'] = crop_bias_tf.numpy()
            else:
                ground_truth = None

            if ssp['sim'].get('analytical_obs', False):
                with frame_profiler.time('analytical'):
                    bg_val = float(tf.reduce_mean(crop_bg_tf).numpy()) if hasattr(crop_bg_tf, 'numpy') else float(np.mean(crop_bg_tf))
                    dc_val = float(tf.reduce_mean(crop_dc_tf).numpy()) if hasattr(crop_dc_tf, 'numpy') else float(np.mean(crop_dc_tf))
                    rn_val = float(rn)
                    en_val = float(en)

                    obs_list = analytic_obs.generate(
                        ssp,
                        obs_os_pix,
                        astrometrics,
                        bg_val,
                        dc_val,
                        rn_val,
                        en_val,
                    )
                    analytical.save(output_dir, frame_num, obs_list)

            if output_debug:
                if fpa_os_w_targets is not None:
                    with open(os.path.join(dir_debug, 'fpa_os_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                        pickle.dump(fpa_os_w_targets.numpy(), picklefile)
                if fpa_conv_os is not None:
                    with open(os.path.join(dir_debug, 'fpa_conv_os_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                        pickle.dump(fpa_conv_os.numpy(), picklefile)
                if fpa_conv_crop is not None:
                    with open(os.path.join(dir_debug, 'fpa_conv_crop_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                        pickle.dump(fpa_conv_crop.numpy(), picklefile)
                with open(os.path.join(dir_debug, 'fpa_conv_star_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                    pickle.dump(fpa_conv_star.numpy(), picklefile)
                with open(os.path.join(dir_debug, 'fpa_conv_targ_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                    pickle.dump(fpa_conv_targ.numpy(), picklefile)
                with open(os.path.join(dir_debug, 'fpa_conv_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                    pickle.dump(fpa_conv.numpy(), picklefile)
                with open(os.path.join(dir_debug, 'fpa_conv_noise_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                    pickle.dump(fpa_conv_noise.numpy(), picklefile)
                with open(os.path.join(dir_debug, 'fpa_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                    pickle.dump(fpa.numpy(), picklefile)
                with open(os.path.join(dir_debug, 'fpa_digital_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                    pickle.dump(fpa_digital.numpy(), picklefile)
                with open(os.path.join(dir_debug, 'stars_os_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                    pickle.dump([r_stars_os.numpy(), c_stars_os.numpy(), pe_stars_os.numpy(), t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os], picklefile)

            frame_profiler.stop('total', frame_total)
            frame_profiler.log(order_times=frame_time_order)

            if with_meta:
                yield fpa_digital, frame_num, astrometrics.copy(), obs_os_pix, fpa_conv_star, fpa_conv_targ, crop_bg_tf, crop_dc_tf, crop_rn_tf, ssp['sim']['num_shot_noise_samples'], obs_cache, ground_truth, star_os_pix, segmentation
            else:
                yield fpa_digital


def _gen_psf(ssp, height, width, y_ifov, x_ifov, s_osf):
    """Generate the point spread function (PSF) for the focal plane array (FPA). """
    psf_os = None
    if ssp['sim']['mode'] != 'none':
        if not isinstance(ssp['fpa']['psf'], dict):  # loaded from config
            psf_os = ssp['fpa']['psf']
            psf_os = tf.cast(psf_os, tf.float32)
        elif ssp['fpa']['psf']['mode'] == 'gaussian':
            logger.debug('Generating {} PSF.'.format(ssp['fpa']['psf']['mode']))
            eod = ssp['fpa']['psf']['eod']
            sigma = eod_to_sigma(eod, s_osf)
            psf_os = gen_gaussian(height, width, sigma)
            save_cache(ssp['fpa']['psf'], psf_os)
        elif ssp['fpa']['psf']['mode'] == 'poppy':
            logger.debug('Generating {} PSF.'.format(ssp['fpa']['psf']['mode']))
            psf_os = gen_from_poppy_configuration(height / s_osf, width / s_osf, y_ifov, x_ifov, s_osf, ssp['fpa']['psf'])
            save_cache(ssp['fpa']['psf'], psf_os)
            psf_os = tf.cast(psf_os, tf.float32)

    return psf_os


def _extract_tle_lines(obs_entry):
    if 'tle' in obs_entry:
        return obs_entry['tle'][0], obs_entry['tle'][1]
    return obs_entry['tle1'], obs_entry['tle2']


def _refresh_tle_cache(obs_entry, obs_cache, index):
    tle1, tle2 = _extract_tle_lines(obs_entry)
    satrec = create_satrec(tle1, tle2)
    target = create_sgp4_from_satrec(satrec, obs_entry.get('name'))
    obs_cache[index] = [target, satrec]
    if 'id' not in obs_entry:
        obs_entry['id'] = tle1[2:7]
    return satrec


def _get_or_create_tle_satrec(obs_entry, obs_cache, updated_flags, index):
    if obs_cache[index] is None or updated_flags[index]:
        updated_flags[index] = False
        return _refresh_tle_cache(obs_entry, obs_cache, index)

    if len(obs_cache[index]) > 1 and obs_cache[index][1] is not None:
        return obs_cache[index][1]

    return _refresh_tle_cache(obs_entry, obs_cache, index)


def _load_sgp4_batch_cache(batch_cache, obs_len):
    if not batch_cache:
        return None
    if batch_cache.get('obs_len') != obs_len:
        return None

    tle_indices = batch_cache.get('tle_indices', [])
    sat_array = batch_cache.get('sat_array')
    if not tle_indices or sat_array is None:
        return None
    try:
        if len(tle_indices) != len(sat_array):
            return None
    except TypeError:
        return None

    return tle_indices, sat_array


def _build_sgp4_batch(obs, obs_cache, active_flags, updated_flags):
    tle_indices = []
    tle_satrecs = []
    for i, o in enumerate(obs):
        if not active_flags[i] or o['mode'] != 'tle':
            continue
        satrec = _get_or_create_tle_satrec(o, obs_cache, updated_flags, i)
        if satrec is not None:
            tle_indices.append(i)
            tle_satrecs.append(satrec)

    sat_array = SatrecArray(tle_satrecs) if tle_satrecs else None
    return tle_indices, sat_array


def _batch_sgp4_fov_filter(obs, obs_cache, active_flags, updated_flags, has_events, batch_cache,
                           ts_mid, observer, ra_c, dec_c, fov_half_diag_pad_cos):
    cached = None
    if batch_cache is not None and not has_events:
        cached = _load_sgp4_batch_cache(batch_cache, len(obs))

    if cached is not None:
        tle_indices, sat_array = cached
    else:
        tle_indices, sat_array = _build_sgp4_batch(obs, obs_cache, active_flags, updated_flags)
        if batch_cache is not None and not has_events and sat_array is not None:
            batch_cache.clear()
            batch_cache.update({
                'obs_len': len(obs),
                'tle_indices': tle_indices,
                'sat_array': sat_array,
            })

    if not tle_indices or sat_array is None:
        return None, 0

    try:
        sat_positions, sat_errors = batch_sgp4_position_gcrs_km(sat_array, ts_mid)
        observer_pos = (observer - load_earth()).at(ts_mid).position.km
        los_to_targets = sat_positions - observer_pos
        norms = np.linalg.norm(los_to_targets, axis=1)
        los_unit = np.zeros_like(los_to_targets)
        valid = norms > 0
        los_unit[valid] = (los_to_targets[valid].T / norms[valid]).T

        los_center = np.array(radec_to_eci(ra_c, dec_c, 1.0))
        cos_angles = np.einsum('ij,j->i', los_unit, los_center)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        if sat_errors is not None:
            cos_angles = np.where(sat_errors != 0, -1.0, cos_angles)

        visible_tle_set = {
            idx for idx, cos in zip(tle_indices, cos_angles)
            if cos >= fov_half_diag_pad_cos
        }
        return visible_tle_set, len(visible_tle_set)
    except Exception:
        logger.exception('Batch SGP4 FOV filter failed; falling back to per-target propagation.')
        return None, 0


def _gen_objects(ssp, render_mode,
                 obs, obs_cache, batch_cache,
                 tc_start, tc_end, t_start, t_end, tt,
                 observer, track, track_az, track_el,
                 zeropoint, s_osf,
                 h_fpa_os, w_fpa_os,
                 h_pad_os, w_pad_os,
                 h_pad_os_div2, w_pad_os_div2,
                 h_fpa_pad_os, w_fpa_pad_os,
                 y_fov, x_fov,
                 y_to_pix, x_to_pix,
                 star_rot, track_mode):
    """Generate object pixels. TODO move to submodule
    """
    obs_model = []
    obs_os_pix = []
    orrr = np.array([], dtype=np.int32)
    occc = np.array([], dtype=np.int32)
    oppp = np.array([], dtype=np.float32)
    ts_start = time.utc_from_list(tt, t_start)
    ts_mid = time.utc_from_list(tt, t_start + 0.5 * (t_end - t_start))
    ts_end = time.utc_from_list(tt, t_end)

    enable_deflection = ssp['sim']['enable_deflection']
    enable_light_transit = ssp['sim']['enable_light_transit']
    enable_stellar_aberration = ssp['sim']['enable_stellar_aberration']
    enable_fov_filter = ssp['sim'].get('enable_fov_filter', False)
    fov_filter_radius = ssp['sim'].get('fov_filter_radius')

    az_arr = None
    el_arr = None
    ra_c = None
    dec_c = None
    fov_half_diag_pad_cos = None

    if enable_fov_filter:
        if track_mode == 'fixed':
            az_arr, el_arr = _calculate_az_el(tc_start, tc_end, [ts_start, ts_mid, ts_end], track_az, track_el)
            ra_c, dec_c, _, _, _, _ = get_los_azel(
                observer,
                az_arr[1],
                el_arr[1],
                ts_mid,
                deflection=enable_deflection,
                aberration=enable_light_transit,
                stellar_aberration=enable_stellar_aberration,
            )
        else:
            ra_c, dec_c, _, _, _, _ = get_los(
                observer,
                track,
                ts_mid,
                deflection=enable_deflection,
                aberration=enable_light_transit,
                stellar_aberration=enable_stellar_aberration,
            )

        if fov_filter_radius is not None:
            fov_half_diag_pad = fov_filter_radius
        else:
            y_ifov_os = y_fov / h_fpa_os
            x_ifov_os = x_fov / w_fpa_os
            y_fov_pad = y_fov + h_pad_os * y_ifov_os
            x_fov_pad = x_fov + w_pad_os * x_ifov_os
            fov_half_diag_pad = 0.5 * math.sqrt(y_fov_pad ** 2 + x_fov_pad ** 2)

        # Convert angle threshold to cosine for efficient comparison
        fov_half_diag_pad_cos = math.cos(math.radians(fov_half_diag_pad))
    else:
        fov_half_diag_pad = None
        fov_half_diag_pad_cos = None
        ra_c = dec_c = None

    obj_profiler = Profiler.from_sim(ssp['sim'], logger, prefix='Object profile')
    total_start = obj_profiler.start('total')
    num_active = num_visible = num_tracks = 0

    has_events = False
    active_flags = [True] * len(obs)
    updated_flags = [False] * len(obs)

    with obj_profiler.time('events'):
        for i, o in enumerate(obs):
            updated = False
            active = True
            if 'events' in o:
                has_events = True
                if 'create' in o['events']:
                    ts_start_ob = time.utc_from_list_or_scalar(o['events']['create'], default_t=tt)
                    if ts_end.tt <= ts_start_ob.tt:
                        active = False
                if active and 'delete' in o['events']:
                    ts_end_ob = time.utc_from_list_or_scalar(o['events']['delete'], default_t=tt)
                    if ts_end.tt >= ts_end_ob.tt:
                        active = False
                if active and 'update' in o['events']:
                    for eu in o['events']['update']:
                        ts_start_ob = time.utc_from_list_or_scalar(eu['time'], default_t=tt)
                        if ts_end.tt >= ts_start_ob.tt:
                            merge_dicts(o, eu['values'])
                            updated = True

            active_flags[i] = active
            updated_flags[i] = updated
            if active:
                num_active += 1

    visible_tle_set = None
    if enable_fov_filter and ra_c is not None and dec_c is not None and observer is not None:
        with obj_profiler.time('batch'):
            visible_tle_set, batch_visible = _batch_sgp4_fov_filter(
                obs,
                obs_cache,
                active_flags,
                updated_flags,
                has_events,
                batch_cache,
                ts_mid,
                observer,
                ra_c,
                dec_c,
                fov_half_diag_pad_cos,
            )
            num_visible += batch_visible

    if visible_tle_set is not None:
        process_indices = [
            i for i, active in enumerate(active_flags)
            if active and (obs[i]['mode'] != 'tle' or i in visible_tle_set)
        ]
    else:
        process_indices = [i for i, active in enumerate(active_flags) if active]

    num_processed = len(process_indices)
    loop_start = obj_profiler.start('loop')
    for i in process_indices:
        o = obs[i]
        updated = updated_flags[i]
        target = None

        if 'mv' in o:
            ope = mv_to_pe(zeropoint, o['mv']) if not callable(o['mv']) else 0.0
            pe_func = (lambda x, t: x) if not callable(o['mv']) else (lambda x, t: mv_to_pe(zeropoint, o['mv'](x, t)) * _delta_t(t))
        elif 'pe' in o:
            ope = o['pe'] if not callable(o['pe']) else 0.0
            pe_func = (lambda x, t: x) if not callable(o['pe']) else (lambda x, t: o['pe'](x, t) * _delta_t(t))
        else:
            ope = 0.0
            pe_func = (lambda x, t: x)

        if o['mode'] == 'line':
            ovrc = [o['velocity'][0] / y_to_pix * s_osf, o['velocity'][1] / x_to_pix * s_osf]
            epoch = o['epoch'] if 'epoch' in o else 0
            (orr, occ, opp, ott) = gen_line(h_fpa_os, w_fpa_os, o['origin'], ovrc, ope, t_start + epoch, t_end + epoch)
        elif o['mode'] == 'line-polar':
            o_vel_angle = o['velocity'][0] * math.pi / 180
            ovrc = [o['velocity'][1] * math.sin(o_vel_angle), o['velocity'][1] * math.cos(o_vel_angle)]
            ovrc = [ovrc[0] / y_to_pix * s_osf, ovrc[1] / x_to_pix * s_osf]
            epoch = o['epoch'] if 'epoch' in o else 0
            (orr, occ, opp, ott) = gen_line(h_fpa_os, w_fpa_os, o['origin'], ovrc, ope, t_start + epoch, t_end + epoch)
        elif o['mode'] == 'tle' or o['mode'] == 'twobody' or o['mode'] == 'gc' or o['mode'] == 'ephemeris' or o['mode'] == 'observation':
            if obs_cache[i] is None or updated:
                if o['mode'] == 'tle':
                    if 'tle' in o:
                        tle1, tle2 = o['tle'][0], o['tle'][1]
                    else:
                        tle1, tle2 = o['tle1'], o['tle2']
                    satrec = create_satrec(tle1, tle2)
                    obs_cache[i] = [create_sgp4_from_satrec(satrec, o.get('name')), satrec]
                    if 'id' not in o:
                        o['id'] = tle1[2:7]

                elif o['mode'] == 'gc':
                    ts_epoch = time.utc_from_list_or_scalar(o['epoch'], default_t=tt)
                    if 'az' in o:
                        obs_cache[i] = [GreatCircle(o['az'], o['el'], o['heading'], o['velocity'], ts_epoch, observer)]
                    else:
                        obs_cache[i] = [GreatCircle(o['ra'], o['dec'], o['heading'], o['velocity'], ts_epoch, None)]
                elif o['mode'] == 'ephemeris':
                    ts_epoch = time.utc_from_list(o['epoch'])
                    obs_cache[i] = [create_ephemeris_object(o['positions'], o['velocities'], o['seconds_from_epoch'], ts_epoch)]
                elif o['mode'] == 'observation':
                    obs_cache[i] = [create_observation(o['ra'], o['dec'], time.utc_from_list(o['time']), observer, track, o.get('range', None))]
                else:
                    ts_epoch = time.utc_from_list_or_scalar(o['epoch'], default_t=tt)
                    obs_cache[i] = [create_twobody(np.array(o['position']) * u.km, np.array(o['velocity']) * u.km / u.s, ts_epoch)]

            if az_arr is None:
                az_arr, el_arr = _calculate_az_el(tc_start, tc_end, [ts_start, ts_mid, ts_end], track_az, track_el)
                if enable_fov_filter and ra_c is None:
                    if track_mode == 'fixed':
                        ra_c, dec_c, _, _, _, _ = get_los_azel(
                            observer,
                            az_arr[1],
                            el_arr[1],
                            ts_mid,
                            deflection=enable_deflection,
                            aberration=enable_light_transit,
                            stellar_aberration=enable_stellar_aberration,
                        )
                    else:
                        ra_c, dec_c, _, _, _, _ = get_los(
                            observer,
                            track,
                            ts_mid,
                            deflection=enable_deflection,
                            aberration=enable_light_transit,
                            stellar_aberration=enable_stellar_aberration,
                        )

            o_offset = [0.0, 0.0]
            if 'offset' in o:
                o_offset = [o['offset'][0] * h_fpa_os, o['offset'][1] * w_fpa_os]

            target = obs_cache[i][0]
            if enable_fov_filter and not (visible_tle_set is not None and o['mode'] == 'tle'):
                # Determine the center line-of-sight of the sensor at mid-exposure
                # Skip propagation if target is outside the padded field of view
                # Use cosine comparison for efficiency (cos decreases as angle increases)
                with obj_profiler.time('fov', accumulate=True):
                    cos_ang = optimized_angle_from_los_cosine(observer, target, ra_c, dec_c, ts_mid)
                if cos_ang < fov_half_diag_pad_cos:
                    continue

            try:
                with obj_profiler.time('gen_track', accumulate=True):
                    [rr0, rr1, rr2], [cc0, cc1, cc2], _, _, _ = gen_track(
                        h_fpa_os,
                        w_fpa_os,
                        y_fov,
                        x_fov,
                        observer,
                        track,
                        [target],
                        [ope],
                        tc_start,
                        [ts_start, ts_mid, ts_end],
                        star_rot,
                        1,
                        track_mode,
                        offset=o_offset,
                        flipud=ssp['fpa']['flip_up_down'],
                        fliplr=ssp['fpa']['flip_left_right'],
                        az=az_arr,
                        el=el_arr,
                        deflection=enable_deflection,
                        aberration=enable_light_transit,
                        stellar_aberration=False,  # disable stellar aberration for target
                    )
                num_tracks += 1
            except Exception:
                logger.exception("Error propagating target. {}".format(o))
                continue

            if len(rr0) > 0 and not math.isnan(rr0[0]) and not math.isnan(cc0[0]) and not math.isnan(rr2[0]) and not math.isnan(cc2[0]):

                if rr0[0] < -h_pad_os and rr2[0] < -h_pad_os:
                    continue
                elif rr0[0] > h_fpa_pad_os and rr2[0] > h_fpa_pad_os:
                    continue
                elif cc0[0] < -w_pad_os and cc2[0] < -w_pad_os:
                    continue
                elif cc0[0] > w_fpa_pad_os and cc2[0] > w_fpa_pad_os:
                    continue

                if ssp['sim']['num_target_samples'] == 2:
                    (orr, occ, opp, ott) = gen_line_from_endpoints(rr0[0], cc0[0], rr2[0], cc2[0], ope, t_start, t_end)
                else:
                    (orr, occ, opp, ott) = gen_curve_from_points(rr0[0], cc0[0], rr1[0], cc1[0], rr2[0], cc2[0], ope, t_start, t_end)
            else:
                continue
        elif o['mode'] == 'none':
            continue

        object_name = o.get('name', '')
        object_id = o.get('id', '')

        # non-sprite based brightness models
        has_brightness_model = 'model' in o and not callable(o['model']) and 'mode' in o['model'] and o['model']['mode'] != 'sprite'
        if has_brightness_model:
            pe_func = (lambda x, t: mv_to_pe(zeropoint, model_to_mv(observer, target, o['model'], time.utc_from_list(tt, _avg_t(t)))) * _delta_t(t))

        opp = pe_func(opp, ott)

        # Apply Earth umbra shadow mask if enabled for the simulation
        if target is not None and ssp.get('sim', {}).get('enable_earth_shadow', False):
            t_mid = time.utc_from_list(tt, _avg_t(ott))
            mask = earth_shadow_umbra_mask(target, t_mid)
            # Ensure mask aligns with opp samples
            if mask.shape[0] == opp.shape[0]:
                opp = opp * mask

        avg_pe = np.sum(opp) / (t_end - t_start)
        avg_mv = pe_to_mv(zeropoint, avg_pe)

        logger.debug('Average brightness for target: {:.2f} mv, {:.2f} pix.'.format(avg_mv, len(opp) / s_osf))

        # TODO generalize `model`
        # sprint based brightness models
        if 'model' in o and not has_brightness_model:
            if render_mode != 'piecewise':
                if callable(o['model']):
                    fpa_os_w_targets = tf.zeros([h_fpa_pad_os, w_fpa_pad_os], tf.float32)
                    patch = o['model'](x=None, t=ott[0])
                    fpa_os_w_targets = add_patch(fpa_os_w_targets, orr, occ, opp, patch, h_pad_os_div2, w_pad_os_div2)
                    obs_model.append(fpa_os_w_targets)
                    # TODO support sub-sample patching
                elif o['model']['mode'] == 'sprite':
                    patch = load_sprite_from_file(filename=o['model']['filename']) if 'mode' in o['model'] else o['model']
                    fpa_os_clear = tf.zeros([h_fpa_pad_os, w_fpa_pad_os], tf.float32)
                    fpa_os_w_targets = add_patch(fpa_os_clear, orr, occ, opp, patch, h_pad_os_div2, w_pad_os_div2)
                    obs_model.append(fpa_os_w_targets)
            else:
                logger.warning('Sprite models not supported for piecewise rendering.')
        else:
            orrr = np.concatenate((orrr, orr))
            occc = np.concatenate((occc, occ))
            oppp = np.concatenate((oppp, opp))

        if obs_cache[i] is not None:
            ra_mid, dec_mid, _ = get_analytical_los(
                observer,
                target,
                ts_mid,
                frame=ssp['sim']['analytical_obs_frame']
            )
            ra_true, dec_true, _, _, _, _ = get_los(
                observer,
                target,
                ts_mid,
                deflection=False,
                aberration=False,
                stellar_aberration=False,
            )
            ra_mid = float(ra_mid)
            dec_mid = float(dec_mid)
            ra_true = float(ra_true)
            dec_true = float(dec_true)
        else:
            ra_mid = None
            dec_mid = None
            ra_true = None
            dec_true = None

        entry = {
            'rr': orr,
            'cc': occ,
            'pp': opp,
            'rrr': orr / s_osf,
            'rcc': occ / s_osf,
            'mv': avg_mv,
            'pe': avg_pe,
            'id': i + 1,  # 0 is reserved for the background for segmentation
            'object_name': object_name,
            'object_id': object_id,
        }
        if ra_mid is not None and dec_mid is not None:
            entry['ra_obs'] = ra_mid
            entry['dec_obs'] = dec_mid
            entry['obs_frame'] = ssp['sim']['analytical_obs_frame']
            entry['ra'] = ra_true
            entry['dec'] = dec_true

        obs_os_pix.append(entry)

    obj_profiler.stop('loop', loop_start)
    obj_profiler.stop('total', total_start)
    obj_profiler.set_metric('active', num_active)
    obj_profiler.set_metric('processed', num_processed)
    obj_profiler.set_metric('visible', num_visible)
    obj_profiler.set_metric('tracks', num_tracks)
    obj_profiler.log(
        order_times=['total', 'events', 'batch', 'loop', 'fov', 'gen_track'],
        order_metrics=['active', 'processed', 'visible', 'tracks'],
    )

    return orrr, occc, oppp, obs_os_pix, obs_model


def _calculate_az_el(tc_start, tc_end, t_curr, track_az, track_el):
    tt = [t.tt for t in t_curr]
    az = interp_degrees(tt, tc_start.tt, tc_end.tt, track_az[0], track_az[1])
    el = interp_degrees(tt, tc_start.tt, tc_end.tt, track_el[0], track_el[1], normalize_360=False)

    return az, el


def _calculate_star_position_and_motion(ssp, astrometrics,
                                        ts_collect_start, ts_collect_end, t_start, t_end, t_exposure,
                                        h_fpa_pad_os, w_fpa_pad_os,
                                        y_fov_pad, x_fov_pad,
                                        y_fov, x_fov,
                                        y_ifov, x_ifov,
                                        observer, track, star_rot, track_mode, track_az, track_el):
    az, el = _calculate_az_el(ts_collect_start, ts_collect_end, [t_start, t_end], track_az, track_el)
    enable_deflection = ssp['sim']['enable_deflection']
    enable_light_transit = ssp['sim']['enable_light_transit']
    enable_stellar_aberration = ssp['sim']['enable_stellar_aberration']
    if track_mode == 'rate':
        track_target = [track]
        star_ra0, star_dec0, _, _, _, _ = get_los(observer, track, t_start, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=enable_stellar_aberration)
        star_ra1, star_dec1, _, _, _, _ = get_los(observer, track, t_end, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=enable_stellar_aberration)
        ra0, dec0, dis0, az0, el0, los0 = get_los(observer, track, t_start, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=False)
        ra1, dec1, dis1, az1, el1, los1 = get_los(observer, track, t_end, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=False)
    elif track_mode == 'fixed':
        track_target = []
        star_ra0, star_dec0, _, _, _, _ = get_los_azel(observer, az[0], el[0], t_start, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=enable_stellar_aberration)
        star_ra1, star_dec1, _, _, _, _ = get_los_azel(observer, az[1], el[1], t_end, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=enable_stellar_aberration)
        ra0, dec0, dis0, az0, el0, los0 = get_los_azel(observer, az[0], el[0], t_start, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=False)
        ra1, dec1, dis1, az1, el1, los1 = get_los_azel(observer, az[1], el[1], t_end, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=False)
    elif track_mode == 'sidereal':
        track_target = [track]
        star_ra0, star_dec0, _, _, _, _ = get_los(observer, track, ts_collect_start, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=enable_stellar_aberration)
        star_ra1, star_dec1, _, _, _, _ = get_los(observer, track, ts_collect_start, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=enable_stellar_aberration)
        ra0, dec0, dis0, az0, el0, los0 = get_los(observer, track, ts_collect_start, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=False)
        ra1, dec1, dis1, az1, el1, los1 = get_los(observer, track, ts_collect_start, deflection=enable_deflection, aberration=enable_light_transit, stellar_aberration=False)
    else:
        logger.error('Unknown track mode: {}.'.format(track_mode))

    [rr0, rr1], [cc0, cc1], drr, dcc, _ = gen_track(
        h_fpa_pad_os,
        w_fpa_pad_os,
        y_fov_pad,
        x_fov_pad,
        observer,
        track,
        track_target,
        [0],
        ts_collect_start,
        [t_start, t_end],
        star_rot,
        1,
        track_mode,
        flipud=ssp['fpa']['flip_up_down'],
        fliplr=ssp['fpa']['flip_left_right'],
        az=az,
        el=el,
        deflection=enable_deflection,
        aberration=enable_light_transit,
        stellar_aberration=False,
    )
    star_tran_os = [-drr, -dcc]  # stars move in the opposite direction of target
    star_rot_rate = 0  # TODO

    astrometrics['ra'] = mean_degrees(ra0, ra1)
    astrometrics['dec'] = (dec0 + dec1) / 2
    astrometrics['ra_apparent'] = mean_degrees(star_ra0, star_ra1)
    astrometrics['dec_apparent'] = (star_dec0 + star_dec1) / 2
    astrometrics['range'] = (dis0 + dis1) / 2
    astrometrics['roll'] = star_rot
    astrometrics['ra_rate'] = diff_degrees(star_ra1, star_ra0) / t_exposure * 3600
    astrometrics['dec_rate'] = (star_dec1 - star_dec0) / t_exposure * 3600
    astrometrics['az'] = mean_degrees(az0, az1)
    astrometrics['el'] = (el0 + el1) / 2
    astrometrics['time'] = time.mid(t_start, t_end).utc_datetime()
    astrometrics['x_ifov'] = x_ifov
    astrometrics['y_ifov'] = y_ifov
    astrometrics['x_fov'] = x_fov
    astrometrics['y_fov'] = y_fov

    logger.debug('Boresight RA, Dec, Roll, Az, El: {}, {}, {}, {}, {}.'.format(astrometrics['ra'], astrometrics['dec'], astrometrics['roll'], astrometrics['az'], astrometrics['el']))

    return star_ra0, star_dec0, star_tran_os, star_rot_rate


def _parse_track_mode(track_mode, frame_num, total_frames):
    if track_mode is not None and track_mode == 'rate-sidereal':
        if frame_num < total_frames - 1:  # TODO make this more general, only supports last frame
            return 'rate'
        else:
            return 'sidereal'

    return track_mode


def _parse_start_track_time(track_mode, frame_num, total_frames, ts_collect_start, ts_start):
    if track_mode is not None and track_mode == 'rate-sidereal':
        if frame_num < total_frames - 1:  # TODO make this more general, only supports last frame
            return ts_collect_start
        else:
            return ts_start
    return ts_collect_start
