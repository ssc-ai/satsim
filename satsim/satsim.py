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

import pydash
import tensorflow as tf
import numpy as np
from astropy import units as u

from satsim.math import signal_to_noise_ratio, mean_degrees, diff_degrees, interp_degrees
from satsim.geometry.transform import rotate_and_translate, apply_wrap_around
from satsim.geometry.sprite import load_sprite_from_file
from satsim.image.fpa import analog_to_digital, mv_to_pe, pe_to_mv, add_patch
from satsim.image.psf import gen_gaussian, eod_to_sigma, gen_from_poppy_configuration
from satsim.image.noise import add_photon_noise, add_read_noise
from satsim.image.render import render_piecewise, render_full
from satsim.geometry.draw import gen_line, gen_line_from_endpoints, gen_curve_from_points
from satsim.geometry.random import gen_random_points
from satsim.geometry.sstr7 import query_by_los
from satsim.geometry.csvsc import query_by_los as csvsc_query_by_los
from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.ephemeris import create_ephemeris_object
from satsim.geometry.astrometric import create_topocentric, gen_track, get_los, get_los_azel, GreatCircle
from satsim.geometry.twobody import create_twobody
from satsim.io.satnet import write_frame, set_frame_annotation, init_annotation
from satsim.io.image import save_apng
from satsim.io.czml import save_czml
from satsim.util import tic, toc, MultithreadedTaskQueue, configure_eager, configure_single_gpu
from satsim.config import transform, save_debug, _transform, save_cache
from satsim.pipeline import _delta_t
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

    (num_frames, exposure_time, h_fpa_os, w_fpa_os, s_osf, y_ifov, x_ifov, a2d_dtype) = _parse_sensor_params(ssp)

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
        ssp['fpa']['height'],
        ssp['fpa']['width'],
        y_ifov,
        x_ifov)

    astrometrics_list = []
    for fpa_digital, frame_num, astrometrics, obs_os_pix, fpa_conv_star, fpa_conv_targ, bg_tf, dc_tf, rn_tf, num_shot_noise_samples, obs_cache, ground_truth in image_generator(ssp, output_dir, output_debug, dir_debug, with_meta=True, num_sets=1):
        astrometrics_list.append(astrometrics)
        if fpa_digital is not None:
            snr = signal_to_noise_ratio(fpa_conv_targ, fpa_conv_star + bg_tf + dc_tf, rn_tf)
            if num_shot_noise_samples is not None:
                snr = snr * np.sqrt(num_shot_noise_samples)
            meta_data = set_frame_annotation(meta_data, frame_num, h_fpa_os, w_fpa_os, obs_os_pix, [20 * s_osf, 20 * s_osf], snr=snr)
            if queue is not None:
                queue.task(write_frame, {
                    'dir_name': dir_name,
                    'sat_name': set_name,
                    'fpa_digital': fpa_digital.numpy(),
                    'meta_data': copy.deepcopy(meta_data),
                    'frame_num': frame_num,
                    'exposure_time': exposure_time,
                    'time_stamp': dt,
                    'ssp': ssp_orig,
                    'show_obs_boxes': ssp['sim']['show_obs_boxes'],
                    'astrometrics': astrometrics,
                    'save_pickle': ssp['sim']['save_pickle'],
                    'dtype': a2d_dtype,
                    'save_jpeg': ssp['sim']['save_jpeg'],
                    'ground_truth': ground_truth,
                }, tag=dir_name)
            if output_debug:
                with open(os.path.join(dir_debug, 'metadata_{}.json'.format(frame_num)), 'w') as jsonfile:
                    json.dump(meta_data, jsonfile, indent=2)

            logger.debug('Finished frame {} of {} in {} sec.'.format(frame_num + 1, num_frames, toc('gen_frame', frame_num)))
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

    h_fpa_os = ssp['fpa']['height'] * s_osf
    w_fpa_os = ssp['fpa']['width'] * s_osf

    exposure_time = ssp['fpa']['time']['exposure']

    num_frames = ssp['fpa']['num_frames']

    if 'dtype' in ssp['fpa']['a2d']:
        a2d_dtype = ssp['fpa']['a2d']['dtype']
    else:
        a2d_dtype = 'uint16'

    return num_frames, exposure_time, h_fpa_os, w_fpa_os, s_osf, y_ifov, x_ifov, a2d_dtype


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

    exposure_time = ssp['fpa']['time']['exposure']
    frame_time = ssp['fpa']['time']['gap'] + exposure_time

    num_frames = ssp['fpa']['num_frames']
    astrometrics['num_frames'] = num_frames

    zeropoint = ssp['fpa']['zeropoint']

    if 'flip_up_down' not in ssp['fpa']:
        ssp['fpa']['flip_up_down'] = False

    if 'flip_left_right' not in ssp['fpa']:
        ssp['fpa']['flip_left_right'] = False

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

    if 'mode' not in ssp['sim']:
        ssp['sim']['mode'] = 'fftconv2p'

    if 'save_pickle' not in ssp['sim']:
        ssp['sim']['save_pickle'] = False

    if star_mode == 'bins':
        star_dn = ssp['geometry']['stars']['mv']['density']
        star_pe = mv_to_pe(zeropoint, ssp['geometry']['stars']['mv']['bins']) * exposure_time
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

    bg = mv_to_pe(zeropoint, ssp['background']['galactic']) * (y_ifov * 3600 * x_ifov * 3600) * exposure_time

    if 'stray' in ssp['background']:
        if isinstance(ssp['background']['stray'], dict) and 'mode' in ssp['background']['stray'] and ssp['background']['stray']['mode'] == 'none':
            pass  # for backward compat
        else:
            bg = bg + ssp['background']['stray'] * exposure_time

    dc = ssp['fpa']['dark_current'] * exposure_time
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

    set_number = 0
    while num_sets <= 0 or set_number < num_sets:
        if num_sets == 0:
            tic('init')

        set_number = set_number + 1

        # time
        if 'time' in ssp['geometry']:
            tt = ssp['geometry']['time']
        else:
            tt = [2020, 1, 1, 0, 0, 0.0]

        t0 = time.utc_from_list(tt)
        t1 = time.utc_from_list(tt, exposure_time)
        t2 = time.utc_from_list(tt, frame_time * num_frames)

        def calculate_az_el(tt):
            ttt = [t.tt for t in tt]
            az = interp_degrees(ttt, t0.tt, t2.tt, track_az[0], track_az[1])
            el = interp_degrees(ttt, t0.tt, t2.tt, track_el[0], track_el[1], normalize_360=False)
            # az = np.interp(ttt, [t0.tt, t2.tt], track_az)
            # el = np.interp(ttt, [t0.tt, t2.tt], track_el)

            return az, el

        def calculate_star_position_and_motion(t_start, t_end, star_rot, track_mode):
            az, el = calculate_az_el([t_start, t_end])
            if track_mode == 'rate':
                track_target = [track]
                star_ra0, star_dec0, dis0, az0, el0, los0 = get_los(observer, track, t_start)
                star_ra1, star_dec1, dis1, az1, el1, los1 = get_los(observer, track, t_end)
            elif track_mode == 'fixed':
                track_target = []
                star_ra0, star_dec0, dis0, az0, el0, los0 = get_los_azel(observer, az[0], el[0], t_start)
                star_ra1, star_dec1, dis1, az1, el1, los1 = get_los_azel(observer, az[1], el[1], t_end)
            else:
                track_target = [track]
                star_ra0, star_dec0, dis0, az0, el0, los0 = get_los(observer, track, t0)
                star_ra1, star_dec1, dis1, az1, el1, los1 = get_los(observer, track, t0)

            [rr0, rr1], [cc0, cc1], drr, dcc, _ = gen_track(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, observer, track, track_target, [0], t0, [t_start, t_end], star_rot, 1, track_mode, flipud=ssp['fpa']['flip_up_down'], fliplr=ssp['fpa']['flip_left_right'], az=az, el=el)
            star_tran_os = [-drr, -dcc]  # stars move in the opposite direction of target
            star_rot_rate = 0  # TODO

            astrometrics['ra'] = mean_degrees(star_ra0, star_ra1)
            astrometrics['dec'] = (star_dec0 + star_dec1) / 2
            astrometrics['range'] = (dis0 + dis1) / 2
            astrometrics['roll'] = star_rot
            astrometrics['ra_rate'] = diff_degrees(star_ra1, star_ra0) / exposure_time * 3600
            astrometrics['dec_rate'] = (star_dec1 - star_dec0) / exposure_time * 3600
            astrometrics['az'] = mean_degrees(az0, az1)
            astrometrics['el'] = (el0 + el1) / 2
            astrometrics['time'] = time.mid(t_start, t_end).utc_datetime()
            astrometrics['x_ifov'] = x_ifov
            astrometrics['y_ifov'] = y_ifov
            astrometrics['x_fov'] = x_fov
            astrometrics['y_fov'] = y_fov

            logger.debug('Boresight RA, Dec, Roll, Az, El: {}, {}, {}, {}, {}.'.format(astrometrics['ra'], astrometrics['dec'], astrometrics['roll'], astrometrics['az'], astrometrics['el']))

            return star_ra0, star_dec0, star_tran_os, star_rot_rate

        # site
        if 'site' in ssp['geometry']:

            # note: stars will track horizontally where zenith is pointed up. focal plane rotation is simulated with the `rotation` variable
            star_rot = ssp['geometry']['site']['gimbal']['rotation']
            track_mode = ssp['geometry']['site']['track']['mode']
            observer = create_topocentric(ssp['geometry']['site']['lat'], ssp['geometry']['site']['lon'])
            astrometrics['lat'] = ssp['geometry']['site']['lat']
            astrometrics['lon'] = ssp['geometry']['site']['lon']
            astrometrics['track_mode'] = track_mode

            if 'tle' in ssp['geometry']['site']['track']:
                track = create_sgp4(ssp['geometry']['site']['track']['tle'][0], ssp['geometry']['site']['track']['tle'][1])
            elif 'tle1' in ssp['geometry']['site']['track']:
                track = create_sgp4(ssp['geometry']['site']['track']['tle1'], ssp['geometry']['site']['track']['tle2'])
            elif 'epoch' in ssp['geometry']['site']['track']:
                track_epoch = time.utc_from_list_or_scalar(ssp['geometry']['site']['track']['epoch'])
                track = create_twobody(np.array(ssp['geometry']['site']['track']['position']) * u.km, np.array(ssp['geometry']['site']['track']['velocity']) * u.km / u.s, track_epoch)
            else:
                track = None

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
                t0 = time.utc_from_list(tt)
                t1 = time.utc_from_list(tt, exposure_time)
                t2 = time.utc_from_list(tt, frame_time * num_frames)

            star_ra, star_dec, star_tran_os, star_rot_rate = calculate_star_position_and_motion(t0, t1, star_rot, track_mode)
        else:
            track_mode = None
            ssp['sim']['star_catalog_query_mode'] = 'at_start'

        if ssp['sim']['temporal_osf'] == 'auto':
            if star_mode != 'none':
                rrr, ccc = rotate_and_translate(h_fpa_pad_os, w_fpa_pad_os, [0,h_fpa_pad_os], [0,w_fpa_pad_os], exposure_time, star_rot_rate, star_tran_os)
                rrr -= [0, h_fpa_pad_os]
                ccc -= [0, w_fpa_pad_os]
                t_osf = tf.cast(max([tf.sqrt(rrr[0] * rrr[0] + ccc[0] * ccc[0]), tf.sqrt(rrr[1] * rrr[1] + ccc[1] * ccc[1])]) + 1, tf.int32)
                logger.debug('Auto temporal oversample factor set to {}.'.format(t_osf.numpy()))
            else:
                t_osf = 1

        # gen stars
        if star_mode == 'bins':
            r_stars_os, c_stars_os, pe_stars_os = gen_random_points(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, star_pe, star_dn, pad_mult=star_pad)
        else:
            r_stars_os, c_stars_os, m_stars_os = [], [], []
            pe_stars_os = []

        # gen psf
        if ssp['sim']['mode'] != 'none':
            if not isinstance(ssp['fpa']['psf'], dict):  # loaded from config
                psf_os = ssp['fpa']['psf']
                psf_os = tf.cast(psf_os, tf.float32)
            elif ssp['fpa']['psf']['mode'] == 'gaussian':
                eod = ssp['fpa']['psf']['eod']
                sigma = eod_to_sigma(eod, s_osf)
                psf_os = gen_gaussian(h_sub_pad_os, w_sub_pad_os, sigma)
                save_cache(ssp['fpa']['psf'], psf_os)
            elif ssp['fpa']['psf']['mode'] == 'poppy':
                psf_os = gen_from_poppy_configuration(h_sub_pad_os / s_osf, w_sub_pad_os / s_osf, y_ifov, x_ifov, s_osf, ssp['fpa']['psf'])
                save_cache(ssp['fpa']['psf'], psf_os)
                psf_os = tf.cast(psf_os, tf.float32)

        if pydash.objects.has(ssp, 'augment.background.stray'):
            bg = ssp['augment']['background']['stray'](bg)
            bg = tf.cast(bg, tf.float32)
            bg = tf.where(tf.math.is_nan(bg), tf.zeros_like(bg), bg)

        logger.debug('Exposure time {}.'.format(exposure_time))
        logger.debug('Background pe/pix {}.'.format(np.mean(bg)))
        logger.debug('Finished initializing variables in {} sec.'.format(toc('init')))

        obs_cache = [None] * len(obs)

        def gen_objects(obs, t_start, t_end):
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

            for i, o in enumerate(obs):

                # TODO support in frame events
                updated = False
                if 'events' in o:
                    if 'create' in o['events']:
                        ts_start_ob = time.utc_from_list_or_scalar(o['events']['create'], default_t=tt)
                        if ts_end.tt <= ts_start_ob.tt:
                            continue
                    if 'delete' in o['events']:
                        ts_end_ob = time.utc_from_list_or_scalar(o['events']['delete'], default_t=tt)
                        if ts_end.tt >= ts_end_ob.tt:
                            continue
                    if 'update' in o['events']:
                        for eu in o['events']['update']:
                            ts_start_ob = time.utc_from_list_or_scalar(eu['time'], default_t=tt)
                            if ts_end.tt >= ts_start_ob.tt:
                                o.update(eu['values'])
                                updated = True

                if 'mv' in o:
                    ope = mv_to_pe(zeropoint, o['mv']) if not callable(o['mv']) else 0.0
                    pe_func = (lambda x, t: x) if not callable(o['mv']) else (lambda x, t: mv_to_pe(zeropoint, o['mv'](x, t)) * _delta_t(t))
                elif 'pe' in o:
                    ope = o['pe'] if not callable(o['pe']) else 0.0
                    pe_func = (lambda x, t: x) if not callable(o['pe']) else (lambda x, t: o['pe'](x, t) * _delta_t(t))

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
                elif o['mode'] == 'tle' or o['mode'] == 'twobody' or o['mode'] == 'gc' or o['mode'] == 'ephemeris':
                    if obs_cache[i] is None or updated:
                        if o['mode'] == 'tle':
                            if 'tle' in o:
                                obs_cache[i] = [create_sgp4(o['tle'][0], o['tle'][1])]
                            else:
                                obs_cache[i] = [create_sgp4(o['tle1'], o['tle2'])]
                        elif o['mode'] == 'gc':
                            ts_epoch = time.utc_from_list_or_scalar(o['epoch'], default_t=tt)
                            if 'az' in o:
                                obs_cache[i] = [GreatCircle(o['az'], o['el'], o['heading'], o['velocity'], ts_epoch, observer)]
                            else:
                                obs_cache[i] = [GreatCircle(o['ra'], o['dec'], o['heading'], o['velocity'], ts_epoch, None)]
                        elif o['mode'] == 'ephemeris':
                            ts_epoch = time.utc_from_list(o['epoch'])
                            obs_cache[i] = [create_ephemeris_object(o['positions'], o['velocities'], o['seconds_from_epoch'], ts_epoch)]
                        else:
                            ts_epoch = time.utc_from_list_or_scalar(o['epoch'], default_t=tt)
                            obs_cache[i] = [create_twobody(np.array(o['position']) * u.km, np.array(o['velocity']) * u.km / u.s, ts_epoch)]

                    # skip rest if not rendering images
                    if ssp['sim']['mode'] == 'none':
                        continue

                    o_offset = [0.0, 0.0]
                    if 'offset' in o:
                        o_offset = [o['offset'][0] * h_fpa_os, o['offset'][1] * w_fpa_os]

                    sat = obs_cache[i]
                    az, el = calculate_az_el([ts_start, ts_mid, ts_end])

                    try:
                        [rr0, rr1, rr2], [cc0, cc1, cc2], _, _, _ = gen_track(h_fpa_os, w_fpa_os, y_fov, x_fov, observer, track, sat, [ope], t0, [ts_start, ts_mid, ts_end], star_rot, 1, track_mode, offset=o_offset, flipud=ssp['fpa']['flip_up_down'], fliplr=ssp['fpa']['flip_left_right'], az=az, el=el)
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

                opp = pe_func(opp, ott)
                avg_pe = np.sum(opp) / (t_end - t_start)
                avg_mv = pe_to_mv(zeropoint, avg_pe)

                logger.debug('Average brightness for target: {:.2f} mv, {:.2f} pix.'.format(avg_mv, len(opp) / s_osf))

                if 'model' in o:
                    if render_mode != 'piecewise':
                        if not callable(o['model']):
                            patch = load_sprite_from_file(filename=o['model']['filename']) if 'mode' in o['model'] else o['model']
                            fpa_os_clear = tf.zeros([h_fpa_pad_os, w_fpa_pad_os], tf.float32)
                            fpa_os_w_targets = add_patch(fpa_os_clear, orr, occ, opp, patch, h_pad_os_div2, w_pad_os_div2)
                            obs_model.append(fpa_os_w_targets)
                        else:
                            fpa_os_w_targets = tf.zeros([h_fpa_pad_os, w_fpa_pad_os], tf.float32)
                            patch = o['model'](x=None, t=ott[0])
                            fpa_os_w_targets = add_patch(fpa_os_w_targets, orr, occ, opp, patch, h_pad_os_div2, w_pad_os_div2)
                            obs_model.append(fpa_os_w_targets)
                            # TODO support sub-sample patching
                    else:
                        logger.warning('Sprite models not supported for piecewise rendering.')
                else:
                    orrr = np.concatenate((orrr, orr))
                    occc = np.concatenate((occc, occ))
                    oppp = np.concatenate((oppp, opp))

                obs_os_pix.append({
                    'rr': orr,
                    'cc': occ,
                    'rrr': orr / s_osf,
                    'rcc': occ / s_osf,
                    'mv': avg_mv,
                    'pe': avg_pe,
                })

            return orrr, occc, oppp, obs_os_pix, obs_model

        if True:  # TODO remove eager

            gain_tf = tf.cast(gain, tf.float32)
            bg_tf = tf.cast(bg, tf.float32)
            dc_tf = tf.cast(dc, tf.float32)
            bias_tf = tf.cast(bias, tf.float32)
            rn_tf = tf.cast(math.sqrt(rn * rn + en * en), tf.float32)

            for frame_num in range(num_frames):
                tic('gen_frame', frame_num)
                logger.debug('Generating frame {} of {}.'.format(frame_num + 1, num_frames))
                astrometrics['frame_num'] = frame_num + 1

                t_start = frame_num * frame_time
                t_end = t_start + exposure_time
                ts_start = time.utc_from_list(tt, t_start)
                ts_end = time.utc_from_list(tt, t_end)
                t_start_star = t_start
                t_end_star = t_end

                # if image rendering is disabled, then propagate objects and return
                if ssp['sim']['mode'] == 'none':
                    r_obs_os, c_obs_os, pe_obs_os, obs_os_pix, obs_model = gen_objects(obs, t_start, t_end)
                    if track_mode is not None:
                        star_ra, star_dec, star_tran_os, star_rot_rate = calculate_star_position_and_motion(ts_start, ts_end, star_rot, track_mode)
                    if with_meta:
                        yield None, frame_num, astrometrics.copy(), None, None, None, None, None, None, None, obs_cache, None
                    else:
                        yield None
                    continue

                # refresh catalog stars
                # TODO should save stars and transform to FPA again on every frame
                if (star_mode == 'sstr7' or star_mode == 'csv') and (ssp['sim']['star_catalog_query_mode'] == 'frame' or frame_num == 0):
                    if track_mode is not None:
                        star_ra, star_dec, star_tran_os, star_rot_rate = calculate_star_position_and_motion(ts_start, ts_end, star_rot, track_mode)
                    if star_mode == 'sstr7':
                        r_stars_os, c_stars_os, m_stars_os = query_by_los(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, star_ra, star_dec, rot=star_rot, rootPath=star_path, pad_mult=star_pad, flipud=ssp['fpa']['flip_up_down'], fliplr=ssp['fpa']['flip_left_right'])
                    elif star_mode == 'csv':
                        r_stars_os, c_stars_os, m_stars_os = csvsc_query_by_los(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, star_ra, star_dec, rot=star_rot, rootPath=star_path, flipud=ssp['fpa']['flip_up_down'], fliplr=ssp['fpa']['flip_left_right'])

                    pe_stars_os = mv_to_pe(zeropoint, m_stars_os) * exposure_time
                    t_start_star = 0.0
                    t_end_star = exposure_time
                    if ssp['sim']['star_catalog_query_mode'] == 'frame':
                        ssp['sim']['apply_star_wrap_around'] = False

                # wrap stars around
                if ssp['sim']['apply_star_wrap_around']:
                    r_stars_os, c_stars_os, star_bounds = apply_wrap_around(h_fpa_pad_os, w_fpa_pad_os, r_stars_os, c_stars_os, t_start, t_end, star_rot_rate, star_tran_os, star_bounds)

                logger.debug('Number of stars {}.'.format(len(r_stars_os)))
                t_start_star = tf.cast(t_start_star, tf.float32)
                t_end_star = tf.cast(t_end_star, tf.float32)
                r_stars_os = tf.cast(r_stars_os, tf.float32)
                c_stars_os = tf.cast(c_stars_os, tf.float32)
                pe_stars_os = tf.cast(pe_stars_os, tf.float32)

                # calculate object pixels
                r_obs_os, c_obs_os, pe_obs_os, obs_os_pix, obs_model = gen_objects(obs, t_start, t_end)
                logger.debug('Number of objects {}.'.format(len(obs_os_pix)))
                r_obs_os = tf.cast(r_obs_os, tf.float32)
                c_obs_os = tf.cast(c_obs_os, tf.float32)
                pe_obs_os = tf.cast(pe_obs_os, tf.float32)

                # augment TODO abstract this
                if pydash.objects.has(ssp, 'augment.fpa.psf'):
                    psf_os_curr = ssp['augment']['fpa']['psf'](psf_os)
                    psf_os_curr = tf.cast(psf_os_curr, tf.float32)
                else:
                    psf_os_curr = psf_os

                # render
                if render_mode == 'piecewise':
                    fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, fpa_conv_os, fpa_conv_crop = render_piecewise(h, w, h_sub, w_sub, h_pad_os, w_pad_os, s_osf, psf_os_curr, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=ssp['sim']['calculate_snr'], star_render_mode=ssp['sim']['star_render_mode'])
                else:
                    fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, fpa_conv_os, fpa_conv_crop = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os_curr, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=ssp['sim']['calculate_snr'], obs_model=obs_model, star_render_mode=ssp['sim']['star_render_mode'])

                # add noise
                fpa_conv = (fpa_conv_star + fpa_conv_targ + bg_tf) * gain_tf + dc_tf
                if ssp['sim']['enable_shot_noise'] is True:
                    fpa_conv_noise = add_photon_noise(fpa_conv, ssp['sim']['num_shot_noise_samples'])
                else:
                    fpa_conv_noise = fpa_conv
                fpa, rn_gt = add_read_noise(fpa_conv_noise, rn, en)

                # analog to digital
                fpa_digital = analog_to_digital(fpa + bias_tf, a2d_gain, a2d_fwc, a2d_bias, dtype=a2d_dtype)

                # augment TODO abstract this
                if pydash.objects.has(ssp, 'augment.image'):
                    if ssp['augment']['image']['post'] is None:
                        pass
                    elif callable(ssp['augment']['image']['post']):
                        fpa_digital = ssp['augment']['image']['post'](fpa_digital)
                    else:
                        fpa_digital = fpa_digital + ssp['augment']['image']['post']

                if ssp['sim']['save_ground_truth']:
                    ground_truth = OrderedDict()
                    ground_truth['target_pe'] = fpa_conv_targ.numpy()
                    ground_truth['star_pe'] = fpa_conv_star.numpy()
                    ground_truth['background_pe'] = bg_tf.numpy()
                    ground_truth['dark_current_pe'] = dc_tf.numpy()
                    ground_truth['photon_noise_pe'] = (fpa_conv_noise - fpa_conv).numpy()
                    ground_truth['read_noise_pe'] = rn_gt.numpy()
                    ground_truth['gain'] = gain_tf.numpy()
                    ground_truth['bias_pe'] = bias_tf.numpy()
                else:
                    ground_truth = None

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

                if with_meta:
                    yield fpa_digital, frame_num, astrometrics.copy(), obs_os_pix, fpa_conv_star, fpa_conv_targ, bg_tf, dc_tf, rn_tf, ssp['sim']['num_shot_noise_samples'], obs_cache, ground_truth
                else:
                    yield fpa_digital
