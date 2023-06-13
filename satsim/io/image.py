from __future__ import division, print_function, absolute_import

from os import listdir
from os.path import isfile, join, splitext, exists

import numpy as np
import matplotlib.image as mpimg

from skimage.draw import rectangle_perimeter


def save(filename, fpa, vauto=False, vmin=None, vmax=None, cmap='gray', annotation=None, pad=5, show_obs_boxes=True, show_star_boxes=False):
    """Save an array as an image file.

    Args:
        filename: `string`, the image filename.
        fpa: `np.array`, input image as a 2D numpy array.
        vauto: vmin is set to min value, and vmax is set to 2*median value
        vmin, vmax: `int`, vmin and vmax set the color scaling for the image by
            fixing the values that map to the colormap color limits. If either
            vmin or vmax is None, that limit is determined from the arr min/max
            value.
        cmap: `str`, A Colormap instance or registered colormap name.
        annotation: `dict`, annotation object created from
            `satsim.io.satnet.set_frame_annotation` used to place a box around
            objects in the image
        pad: `int`, pad length in pixels to add to each side of the annotation
            box
    """

    fpa_flat = fpa.flatten()
    min_val = vmin or np.min(fpa_flat)
    max_val = vmax or np.max(fpa_flat)

    if vauto:
        max_val = (np.median(fpa_flat) - min_val) * 4 + min_val

    fpa_np = fpa

    if annotation is not None and (show_obs_boxes is True or show_star_boxes is True):
        (h, w) = fpa_np.shape
        for a in annotation:
            if (show_obs_boxes and a['class_name'] == 'Satellite') or (show_star_boxes and a['class_name'] == 'Star'):
                start = (a['y_min'] * h - pad, a['x_min'] * w - pad)
                end = (a['y_max'] * h + pad, a['x_max'] * w + pad)
                rr, cc = rectangle_perimeter(start, end=end, shape=fpa_np.shape)
                fpa_np[rr,cc] = 100000000

    mpimg.imsave(filename, fpa_np, vmin=min_val, vmax=max_val, cmap=cmap)


def save_apng(dirname, filename):
    """Combine all jpg and png image files in the specified directory into an
    animated PNG file. Useful to view images in a web browser.

    Args:
        dirname: `string`, directory containing image files to combine.
        filename: `string`, file name of the animated PNG.
    """
    if exists(dirname):

        from apng import APNG

        files = [join(dirname, f) for f in sorted(listdir(dirname)) if isfile(join(dirname, f)) and (splitext(f)[1] == '.png' or splitext(f)[1] == '.jpg')]

        APNG.from_files(files, delay=100).save(join(dirname, filename))
