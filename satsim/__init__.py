# -*- coding: utf-8 -*-

"""Top-level package for SatSim."""

__author__ = """Alex Cabello"""
__email__ = 'alexander.cabello@algoritics.com'
__version__ = '0.22.0'

from .satsim import gen_images, gen_multi, image_generator
from .config import load_json, load_yaml