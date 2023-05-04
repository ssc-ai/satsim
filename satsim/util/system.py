from __future__ import division, print_function, absolute_import

import re
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)
IS_CPU = len(tf.config.list_physical_devices('GPU')) == 0


def get_semantic_version(module):
    """Gets the sematic version from a module as an array of integers. Non digit characters
    will be removed.

    Example::

        import tensorflow as tf

        ver = get_semantic_version(tf)

        # example ver == [2,0,0]

    Args:
        module: `object`, a module which has __version__ as a string in semantic format
    """
    return [int(re.sub(r'\D.*$','',v)) for v in module.__version__.split('.')]


def configure_eager(allow_growth=True):
    """Configure TensorFlow in eager mode.

    Args:
        allow_growth: `bool`, enables memory growth
    """
    tf.compat.v1.enable_eager_execution()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, allow_growth)


def configure_single_gpu(gpu_id, memory=None):
    """Configure a single GPU to be visible and limit its maximum memory allocation.

    Args:
        gpu_id: `int`, system GPU id. On single GPU systems, this should be set to 0.
        memory: `int`, maximum memory allocation in megabytes. Default is `None` for no limit.

    Returns:
        gpus: `string`, a list of size 1 with the visible gpu name
    """
    return configure_multi_gpu([gpu_id], memory)


def configure_multi_gpu(gpu_ids, memory=None):
    """Configure multiple GPUs to be visible and limit their maximum memory allocation.

    Args:
        gpu_ids: `array`, int array of system GPU ids. On single GPU systems, this should be set to [0].
        memory: `int`, maximum memory allocation in megabytes. Default is `None` for no limit.

    Returns:
        gpus: `string`, a list of logical gpu names
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    logger.debug('Visible gpus: {}'.format(gpus))

    if len(gpus) == 0:
        return gpus

    tf.config.experimental.set_visible_devices([gpus[i] for i in gpu_ids], 'GPU')

    logger.debug('Configuring GPU device {} with {} MB'.format(gpu_ids, memory))

    if memory is not None:
        for gpu_id in gpu_ids:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_id],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)]
            )

    gpus = tf.config.experimental.list_logical_devices('GPU')

    logger.debug('Logical gpus: {}'.format(gpus))

    return gpus


def is_tensorflow_running_on_cpu():
    """Returns True if TensorFlow is running on the CPU. """
    return IS_CPU
