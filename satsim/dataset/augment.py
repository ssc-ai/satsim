import copy

import numpy as np
import tensorflow as tf

from satsim import image_generator
from satsim.config import transform
from satsim.io.satnet import set_frame_annotation, init_annotation
from satsim.math import signal_to_noise_ratio


def augment_satnet_with_satsim(dataset, augment_satsim_params, prob=0.5, rn=0, min_snr=2.0, box_pad=10):
    """Augments a SatNet Dataset with SatSim images. The SatSim image is added onto the
    example image. New synthetic targets are appended to the bounding box annotation list.

    Args:
        dataset: `tf.Dataset`, Dataset object that returns SatNet types:
            image: tf.float32,
            bounding box annotations: tf.float32,
            filename: tf.string,
            annotation filename : tf.string
        augment_satsim_params: `dict`, SatSim simulation parameters
        prob: `float`, probability of augmenting the example image. default=0.5
        rn: `float`, estimated read noise of sensor in photoelectrons used to estimate SNR. default=0
        min_snr: `float`, minimum SNR to include synthetic targets in annotation list. default=2.0
        box_pad: `int`, number of pixels to pad synthetic targets on each side. default=10

    Returns:
        A `tf.Dataset`, the mapped Dataset with SatSim augmentation
    """

    def _augment_satnet_with_satsim(image, bboxs, filename=None, annotational_filename=None, prob=prob, rn=rn, min_snr=min_snr, box_pad=box_pad, ssp=augment_satsim_params):

        if np.random.uniform() > prob:
            return image, bboxs, filename, annotational_filename

        s_osf = ssp['sim']['spacial_osf']
        a2d_gain = ssp['fpa']['a2d']['gain']
        h = ssp['fpa']['height']
        w = ssp['fpa']['width']
        y_fov = ssp['fpa']['y_fov']
        x_fov = ssp['fpa']['x_fov']
        box_pad = box_pad / w

        sspc = transform(copy.deepcopy(ssp), '.')
        sspc['augment']['image']['post'] = tf.squeeze(image)

        ig = image_generator(sspc, with_meta=True)
        fpa_digital, frame_num, astrometrics, obs_os_pix, fpa_conv_star, fpa_conv_targ, bg_tf, dc_tf, rn_tf, num_shot_noise_samples, obs_cache, ground_truth, star_os_pix, segmentation = ig.__next__()

        anno = init_annotation('.', 0, h, w, y_fov, x_fov)
        snr = signal_to_noise_ratio(fpa_conv_targ, tf.squeeze(image) * a2d_gain, rn)
        set_frame_annotation(anno, ['null'], h * s_osf, w * s_osf, obs_os_pix, snr=snr, star_os_pix=star_os_pix)

        unbboxs = tf.unstack(bboxs)
        for ii in range(len(unbboxs)):
            if unbboxs[ii][4] == 0:
                break

        for ob in anno['data']['objects']:
            if(max(ob['snr']) > min_snr):
                unbboxs[ii] = [ob['y_start'] - box_pad, ob['x_start'] - box_pad, ob['y_end'] + box_pad, ob['x_end'] + box_pad, 1]
                ii += 1

        return tf.expand_dims(fpa_digital, -1), tf.stack(unbboxs), filename, annotational_filename

    def _wrapper_pyfunc(image, bboxs, filename, annotational_filename):

        a, b, c, d = tf.py_function(func=_augment_satnet_with_satsim, inp=[image, bboxs, filename, annotational_filename], Tout=[tf.float32, tf.float32, tf.string, tf.string])

        # important to set shape to know output dimensions for tensorflow dataset API
        a.set_shape(image.shape)
        b.set_shape(bboxs.shape)
        c.set_shape(filename.shape)
        d.set_shape(annotational_filename.shape)

        return a, b, c, d

    return dataset.map(_wrapper_pyfunc, num_parallel_calls=1)
