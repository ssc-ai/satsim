from __future__ import division, print_function, absolute_import

import os
import logging
import copy
from datetime import datetime
import pickle
import math
import itertools
import numpy as np
import threading
import multiprocessing
import time

import tensorflow as tf

from satsim.math import fftconv2p, signal_to_noise_ratio
from satsim.image.fpa import downsample, crop, analog_to_digital, mv_to_pe, add_counts, transform_and_add_counts
from satsim.image.psf import gen_gaussian, eod_to_sigma
from satsim.image.noise import add_photon_noise, add_read_noise
from satsim.geometry.draw import gen_line, gen_line_from_endpoints
from satsim.geometry.random import gen_random_points
from satsim.geometry.sstr7 import query_by_los
from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.astrometric import create_topocentric, gen_track, get_los
from satsim.io.satnet import write_frame, set_frame_annotation, init_annotation
from satsim.io.image import save_apng
from satsim.util import tic, toc, configure_eager, configure_multi_gpu
from satsim.config import transform
from satsim import time

logger = logging.getLogger(__name__)


def gen_multi(ssp, eager=True, output_dir='./', input_dir='./', device=None, memory=None):
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
        eager: `boolean`, `True` to run TensorFlow in eager mode, `False` to run
            in graph mode.
        output_dir: `str`, output directory to save SatNet files
        input_dir: `str`, typically the input directory of the configuration
            file
    """
    if(eager):
        configure_eager()
        # tf.config.experimental.set_synchronous_execution()
        # tf.debugging.set_log_device_placement(True)

    devices = []
    if device is not None:
        devices = configure_multi_gpu(device, memory)
        devices = [d.name for d in devices]

    n = ssp['sim']['samples'] if 'samples' in ssp['sim'] else 1
    n = math.ceil(n//len(devices))
    batch_size = len(devices)

    logger.error('Devices {}'.format(batch_size))
    logger.error('Batch size = {}'.format(batch_size))

    dataset0 = tf.data.Dataset.from_generator(lambda: gen_sample(ssp, n=n),
        (tf.int32, tf.int32, tf.int32,
        tf.float32, tf.float32, tf.int32,
        tf.float32, tf.float32,
        tf.int32, tf.int32,
        tf.int32, tf.int32,
        tf.int32, tf.int32,
        tf.int32, tf.int32, tf.int32,
        tf.float32,
        tf.int32, tf.int32, tf.float32, tf.float32, #tf.string,
        tf.float32, tf.float32, tf.float32,tf.float32, tf.float32, tf.float32,tf.float32, tf.float32, tf.float32),
        (tf.TensorShape((None,)),tf.TensorShape((None,)),tf.TensorShape((None,)),
        tf.TensorShape(()),tf.TensorShape(()),tf.TensorShape(()),
        tf.TensorShape(()),tf.TensorShape((2,)),
        tf.TensorShape(()),tf.TensorShape(()),
        tf.TensorShape(()),tf.TensorShape(()),
        tf.TensorShape(()),tf.TensorShape(()),
        tf.TensorShape(()),tf.TensorShape(()),tf.TensorShape(()),
        tf.TensorShape(()),
        tf.TensorShape((None,)),tf.TensorShape((None,)),tf.TensorShape((None,)),tf.TensorShape((None,)), #tf.TensorShape(()),
        tf.TensorShape(()),tf.TensorShape((None,None)),tf.TensorShape(()),tf.TensorShape((None,None)),tf.TensorShape(()),tf.TensorShape(()),tf.TensorShape(()),tf.TensorShape(()),tf.TensorShape(()),
        )
    )
    dataset = dataset0

    if devices:

        # batch_size = 2 #len(devices)
        dataset = tf.data.Dataset.range(batch_size).interleave(lambda x: dataset, num_parallel_calls=batch_size)
        strategy = tf.distribute.MirroredStrategy(devices=devices)
        # strategy = tf.distribute.OneDeviceStrategy(device=devices[0])
        dist_dataset = strategy.experimental_distribute_dataset(dataset.batch(1).prefetch(1))
        i = 0
        for d in dist_dataset:
            tic('gen_frame', 0)
            logger.error('len(d)={}'.format(len(d)))
            strategy.experimental_run_v2(gen_frame, args=(d,))
            i = i + batch_size
            logger.error('Finished sample {} of {} in {} sec.'.format(i, n*6*batch_size, toc('gen_frame', 0)))

    else:
        i = 0
        for d in dataset.prefetch(1):
            tic('gen_frame', 0)
            gen_frame(d)
            i = i + 1
            logger.error('Finished sample {} of {} in {} sec.'.format(i, n*6, toc('gen_frame', 0)))


def gen_sample_demo(init_ssp, eager=True, output_dir='./', input_dir='./', output_debug=False, n=1):

    for sample_num in itertools.count(1):

        if sample_num > n:
            return

        print('gen', sample_num)

        yield (sample_num, tf.random.uniform((2,2)))


def gen_frame_demo(d):

    print('samplenum=',d[0])
    
#    return tf.reduce_sum(d[1]) * tf.cast(d[0], tf.float32)
    return d[0]


def gen_sample(init_ssp, eager=True, output_dir='./', input_dir='./', output_debug=False, n=1, db={}):

    for sample_num in itertools.count(1):

        if sample_num > n:
            return

        ssp = transform(copy.deepcopy(init_ssp), input_dir)

        s_osf = ssp['sim']['spacial_osf']
        t_osf = ssp['sim']['temporal_osf']

        h_pad_os = 2 * ssp['sim']['padding'] * s_osf
        w_pad_os = 2 * ssp['sim']['padding'] * s_osf

        y_ifov = ssp['fpa']['y_fov'] / ssp['fpa']['height']
        x_ifov = ssp['fpa']['x_fov'] / ssp['fpa']['width']

        y_ifov_os = ssp['fpa']['y_fov'] / ssp['fpa']['height'] / s_osf
        x_ifov_os = ssp['fpa']['x_fov'] / ssp['fpa']['width'] / s_osf

        h_fpa_os = ssp['fpa']['height'] * s_osf
        w_fpa_os = ssp['fpa']['width'] * s_osf
        h_psf_os = h_fpa_os
        w_psf_os = w_fpa_os

        h_fpa_pad_os = h_fpa_os + h_pad_os
        w_fpa_pad_os = w_fpa_os + w_pad_os
        h_psf_pad_os = h_psf_os + h_pad_os
        w_psf_pad_os = w_psf_os + w_pad_os

        y_fov = h_fpa_os * y_ifov_os
        x_fov = w_fpa_os * x_ifov_os

        y_fov_pad = h_fpa_pad_os * y_ifov_os
        x_fov_pad = w_fpa_pad_os * x_ifov_os

        exposure_time = ssp['fpa']['time']['exposure']
        frame_time = ssp['fpa']['time']['gap'] + exposure_time

        num_frames = ssp['fpa']['num_frames']

        zeropoint = ssp['fpa']['zeropoint']

        star_mode = ssp['geometry']['stars']['mode']

        if star_mode == 'bins':
            star_dn = ssp['geometry']['stars']['mv']['density']
            star_pe = mv_to_pe(zeropoint, ssp['geometry']['stars']['mv']['bins']) * exposure_time
        elif star_mode == 'sstr7':
            star_ra = ssp['geometry']['stars']['ra'] if 'ra' in ssp['geometry']['stars'] else 0
            star_dec = ssp['geometry']['stars']['dec'] if 'dec' in ssp['geometry']['stars'] else 0
            star_rot = ssp['geometry']['stars']['rotation'] if 'rotation' in ssp['geometry']['stars'] else 0
            star_path = ssp['geometry']['stars']['path']

        if ssp['geometry']['stars']['motion']['mode'] == 'affine':
            star_rot_rate = ssp['geometry']['stars']['motion']['rotation']
            star_tran_os = [ssp['geometry']['stars']['motion']['translation'][0],ssp['geometry']['stars']['motion']['translation'][1]]
            star_tran_os[0] = star_tran_os[0] * s_osf
            star_tran_os[1] = star_tran_os[1] * s_osf

        obs = ssp['geometry']['obs']['list']

        bg = mv_to_pe(zeropoint, ssp['background']['galactic']) * (y_ifov * 3600 * x_ifov * 3600) * exposure_time
        dc = ssp['fpa']['dark_current'] * exposure_time
        rn = ssp['fpa']['noise']['read']
        en = ssp['fpa']['noise']['electronic']
        gain = ssp['fpa']['gain']
        bias = ssp['fpa']['bias']
        a2d_fwc = ssp['fpa']['a2d']['fwc']
        a2d_gain = ssp['fpa']['a2d']['gain']
        a2d_bias = ssp['fpa']['a2d']['bias']
        eod = ssp['fpa']['psf']['eod']

        # site
        if 'site' in ssp['geometry']:
            from skyfield.api import load

            # note: stars will track horizontally where zenith is pointed up. focal plane rotation is simulated with the `rotation` variable
            star_rot = ssp['geometry']['site']['gimbal']['rotation']
            track_mode = ssp['geometry']['site']['track']['mode']
            observer = create_topocentric(ssp['geometry']['site']['lat'], ssp['geometry']['site']['lon'])
            track = create_sgp4(ssp['geometry']['site']['track']['tle1'], ssp['geometry']['site']['track']['tle2'])

            tt = ssp['geometry']['site']['time']
            t0 = time.utc(tt[0],tt[1],tt[2],tt[3],tt[4],tt[5])
            t1 = time.utc(tt[0],tt[1],tt[2],tt[3],tt[4],tt[5] + exposure_time)

            star_ra, star_dec, dis, az, el, los = get_los(observer, track, t0)
            rr0, cc0, rr1, cc1, drr, dcc, _ = gen_track(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, observer, track, [track], [0], t0, t0, t1, star_rot, 1, track_mode)
            star_tran_os = [-drr, -dcc]  # stars move in the opposite direction of target
            star_rot_rate = 0  # TODO

        # gen stars
        if star_mode == 'bins':
            r_stars_os, c_stars_os, p_stars_os = gen_random_points(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, star_pe, star_dn)
        elif star_mode == 'sstr7':
            r_stars_os, c_stars_os, m_stars_os = query_by_los(h_fpa_pad_os, w_fpa_pad_os, y_fov_pad, x_fov_pad, star_ra, star_dec, rot=star_rot, rootPath=star_path, pad_mult=1)
            p_stars_os = mv_to_pe(zeropoint, m_stars_os) * exposure_time

        # gen psf
        sigma = eod_to_sigma(eod, s_osf)
        # psf_os = gen_gaussian(h_psf_pad_os, w_psf_pad_os, sigma)

        dt = datetime.now()
        dir_name = os.path.join(output_dir, dt.isoformat().replace(':','-'))
        set_name = 'sat_{:05d}'.format(sample_num)
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

        for frame_num in range(num_frames):
            t_start = tf.cast(frame_num * frame_time, tf.float32)
            t_end = tf.cast(t_start + exposure_time, tf.float32)

            obs_os_pix = []
            orr = np.array([], dtype=int)
            occ = np.array([], dtype=int)
            opp = np.array([], dtype=float)
            ott = np.array([], dtype=float)
            for o in obs:
                if o['mode'] == 'line':
                    ovrc = [o['velocity'][0] * s_osf, o['velocity'][1] * s_osf]
                    ope = mv_to_pe(zeropoint, o['mv'])
                    (torr, tocc, topp, tott) = gen_line(h_fpa_os, w_fpa_os, o['origin'], ovrc, ope, t_start, t_end)
                elif o['mode'] == 'tle':
                    ts_start = time.utc(tt[0], tt[1], tt[2], tt[3], tt[4], tt[5] + t_start)
                    ts_end = time.utc(tt[0], tt[1], tt[2], tt[3], tt[4], tt[5] + t_end)
                    sat = [create_sgp4(o['tle1'], o['tle2'])]
                    ope = [mv_to_pe(zeropoint, o['mv'])]
                    rr0, cc0, rr1, cc1, _, _, _ = gen_track(h_fpa_os, w_fpa_os, y_fov, x_fov, observer, track, sat, ope, t0, ts_start, ts_end, star_rot, 1, track_mode)
                    (torr, tocc, topp, tott) = gen_line_from_endpoints(rr0[0], cc0[0], rr1[0], cc1[0], ope[0], t_start, t_end)
                
                orr = np.append(orr, torr)
                occ = np.append(occ, tocc)
                opp = np.append(opp, topp)
                ott = np.append(ott, tott)

            yield (r_stars_os, c_stars_os, p_stars_os, 
                   t_start, t_end, t_osf,
                   star_rot_rate, star_tran_os, 
                   h_fpa_pad_os, w_fpa_pad_os,
                   h_psf_pad_os, w_psf_pad_os,
                   h_fpa_os, w_fpa_os,
                   h_pad_os, w_pad_os, s_osf,
                   sigma,
                   orr, occ, opp, ott, #dt.isoformat(),
                   bg, gain, bias, dc, rn, en, a2d_gain, a2d_fwc, a2d_bias)


def gen_images():
    pass


@tf.function(experimental_relax_shapes=True)
def gen_frame(d, osf=15):

    # osf = 15

    # r_stars_os, c_stars_os, p_stars_os, t_start, t_end, t_osf, star_rot_rate, star_tran_os, h_fpa_pad_os, w_fpa_pad_os, h_psf_pad_os, w_psf_pad_os,h_fpa_os, w_fpa_os, h_pad_os, w_pad_os, s_osf, sigma, orr, occ, opp, ott, dt, bg, gain, bias, dc, rn, en, a2d_gain, a2d_fwc, a2d_bias = d
    r_stars_os, c_stars_os, p_stars_os, t_start, t_end, t_osf, star_rot_rate, star_tran_os, h_fpa_pad_os, w_fpa_pad_os, h_psf_pad_os, w_psf_pad_os,h_fpa_os, w_fpa_os, h_pad_os, w_pad_os, s_osf, sigma, orr, occ, opp, ott, bg, gain, bias, dc, rn, en, a2d_gain, a2d_fwc, a2d_bias = d

    # print('r_stars_os', r_stars_os[0])
    # print('h_psf_pad_os', h_psf_pad_os[0])
    # print('w_psf_pad_os', w_psf_pad_os[0])

    psf_os = gen_gaussian(h_psf_pad_os[0], w_psf_pad_os[0], sigma[0])

    h_pad_os_div2 = tf.cast(h_pad_os[0] / 2, tf.int32)
    w_pad_os_div2 = tf.cast(w_pad_os[0] / 2, tf.int32)

    # build star graph
    # fpa_os_clear = tf.compat.v1.assign(fpa_os, tf.zeros_like(fpa_os))
    fpa_os_clear = tf.zeros([h_fpa_pad_os[0], w_fpa_pad_os[0]], tf.float32)
    fpa_os_w_stars = transform_and_add_counts(fpa_os_clear, r_stars_os[0], c_stars_os[0], p_stars_os[0], t_start[0], t_end[0], t_osf[0], star_rot_rate[0], star_tran_os[0])
    fpa_conv_os = fftconv2p(fpa_os_w_stars, psf_os, pad=1)
    fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os[0], w_fpa_os[0])
    fpa_conv_star = downsample(fpa_conv_crop, osf)

    # build target graph
    # fpa_os_clear = tf.compat.v1.assign(fpa_os, tf.zeros_like(fpa_os))
    fpa_os_clear = tf.zeros([h_fpa_pad_os[0], w_fpa_pad_os[0]], tf.float32)
    fpa_os_w_targets = add_counts(fpa_os_clear, orr[0], occ[0], opp[0], h_pad_os_div2, w_pad_os_div2)
    fpa_conv_os = fftconv2p(fpa_os_w_targets, psf_os, pad=1)
    fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os[0], w_fpa_os[0])
    fpa_conv_targ = downsample(fpa_conv_crop, osf)

    # add noise
    fpa_conv = (fpa_conv_targ + fpa_conv_star + bg[0]) * gain[0] + dc[0]
    fpa_conv_noise = add_photon_noise(fpa_conv)
    fpa = add_read_noise(fpa_conv_noise, rn[0], en[0])
    fpa_digital = analog_to_digital(fpa + bias[0], a2d_gain[0], a2d_fwc[0], a2d_bias[0])

    # meta data
    snr = signal_to_noise_ratio(fpa_conv_targ, fpa_conv_star + bg[0] + dc[0], tf.math.sqrt(rn[0] * rn[0] + en[0] * en[0]))

#    print('snr:', tf.shape(snr))

    return fpa_digital, snr


def write_frame():

    meta_data = set_frame_annotation(meta_data, frame_num, h_fpa_os, w_fpa_os, obs_os_pix, [20 * s_osf, 20 * s_osf], snr=snr)

    if eager:
        write_frame(dir_name, set_name, fpa_digital.numpy(), meta_data, frame_num, exposure_time, dt, ssp_orig)
        if output_debug:
            with open(os.path.join(dir_debug, 'fpa_os_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                pickle.dump(fpa_os_w_targets.numpy(), picklefile)
            with open(os.path.join(dir_debug, 'fpa_conv_os_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                pickle.dump(fpa_conv_os.numpy(), picklefile)
            with open(os.path.join(dir_debug, 'fpa_conv_crop_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                pickle.dump(fpa_conv_crop.numpy(), picklefile)
            with open(os.path.join(dir_debug, 'fpa_conv_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                pickle.dump(fpa_conv.numpy(), picklefile)
            with open(os.path.join(dir_debug, 'fpa_conv_noise_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                pickle.dump(fpa_conv_noise.numpy(), picklefile)
            with open(os.path.join(dir_debug, 'fpa_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                pickle.dump(fpa.numpy(), picklefile)
            with open(os.path.join(dir_debug, 'fpa_digital_{}.pickle'.format(frame_num)), 'wb') as picklefile:
                pickle.dump(fpa_digital.numpy(), picklefile)


def write_movie():

    save_apng(os.path.join(dir_name,'AnnotatedImages'), 'movie.png')

