from __future__ import division, print_function, absolute_import

import sys
import os
import logging
import multiprocessing
import math
import numpy as np
from satsim import __version__

import click

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name='SatSim')
@click.option('-d', '--debug', default='WARNING', show_default=True, help='Set the logging level. [DEBUG,INFO,WARNING,ERROR,OFF]')
@click.pass_context
def main(ctx, debug):
    """ Command line interface (CLI) for SatSim.
    """

    logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
    if debug == 'OFF':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger().propagate = False
    elif debug == 'DEBUG':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.DEBUG)
    elif debug == 'INFO':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        logging.getLogger().setLevel(logging.INFO)
    elif debug == 'ERROR':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    return 0


@main.command()
@click.pass_context
def version(ctx):
    from satsim import __version__
    print(__version__)
    return 0


@main.command(help='Run simulation from configuration file (auto-detects radar vs EO).')
@click.option('-d', '--device', default='0', type=str, help='GPU device ids to utilize. example: 0,1,3,4')
@click.option('-r', '--memory', default=None, type=int, help='GPU maximum memory limit in megabytes per process.')
@click.option('-j', '--jobs', default=1, type=int, help='Allow N jobs at once per GPU device.')
@click.option('-m', '--mode', default='eager', help='Mode to run Tensorflow backend. eager or dataset. (only eager is supported)')
@click.option('-o', '--output_dir', default='./', help='Output directory to save files.')
@click.option('-i', '--output_intermediate', is_flag=True, help='Output intermediate images as pickle files.')
@click.argument('config_file', required=True)
@click.pass_context
def run(ctx, device, memory, jobs, mode, output_dir, config_file, output_intermediate):

    # multiprocessing.set_start_method('spawn')

    if device is not None:
        device = [int(d) for d in device.split(',')]

    total_jobs = jobs * len(device)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # if device is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    from satsim import load_yaml, load_json, __version__

    logger.info('SatSim version {}.'.format(__version__))

    ssp = {}
    if config_file.endswith('.yml'):
        logger.info('Loading yaml file: {}.'.format(config_file))
        ssp = load_yaml(config_file)

    elif config_file.endswith('.json'):
        logger.info('Loading json file: {}.'.format(config_file))
        ssp = load_json(config_file)

    else:
        logger.error('File type unknown. Config file must be .json or .yml type.')
        sys.exit(1)

    # If config contains a radar block, dispatch to radar simulator
    if isinstance(ssp, dict) and 'radar' in ssp:
        from satsim.radar import simulate_from_file
        logger.info('Detected radar configuration. Dispatching to radar simulator.')
        out_dir = simulate_from_file(config_file, output_dir)
        logger.info('Saved radar observations to: {}'.format(out_dir))
        return 0

    # Otherwise, run EO/optical pipeline as before
    logger.info('Running Tensorflow backend in {} mode.'.format(mode))
    ssp['sim']['samples'] = math.ceil(ssp['sim']['samples'] / total_jobs)

    if mode == 'eager':
        logger.info('Running {} parallel jobs on GPUs {}, each with {} samples.'.format(total_jobs, device, ssp['sim']['samples']))
        from satsim import gen_multi
        if total_jobs > 1:
            mp_dev = np.repeat(device, jobs)
            mp_mem = np.repeat(memory, total_jobs)
            args = [(ssp, True, output_dir, os.path.dirname(config_file), d, m, i, output_intermediate) for d, m, i in zip(mp_dev, mp_mem, range(total_jobs))]
            pool = multiprocessing.Pool(processes=total_jobs)
            pool.starmap(gen_multi, args)
        else:
            gen_multi(ssp, True, output_dir, os.path.dirname(config_file), device[0], memory, 0, output_intermediate)
    else:
        logger.error('Error: Unrecognized mode.')


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
