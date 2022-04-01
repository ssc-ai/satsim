#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'Click~=7.0',
    'Cython~=0.29.0',
    'scikit-image~=0.17.2',
    'numpy<1.22,>=1.16.0',
    'pandas~=0.24',
    'PyYAML~=5.1',
    'matplotlib~=3.3.0',
    'astropy==3.2.3',
    'astroquery==0.4.1',
    'apng~=0.3.4',
    'scipy~=1.5.4',
    'tensorflow~=2.2',
    'tensorflow-addons~=0.11',
    'skyfield==1.26',
    'sgp4==2.17',
    'poliastro==0.14.0',
    'poppy==0.9.1',
    'diskcache==5.0.3',
    'pydash~=4.9.0',
    'asdf==2.7.1',
    'pygc==1.1.0',
    'numba~=0.53.0',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Alex Cabello",
    author_email='alexander.cabello@algoritics.com',
    classifiers=[
        'Development Status :: 4 - Beta ',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Satellite observation and scene simulator.",
    entry_points={
        'console_scripts': [
            'satsim=satsim.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='satsim',
    name='satsim',
    packages=find_packages(include=['satsim', 'satsim.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.pacificds.com/machine-learning/satsim',
    version='0.12.0',
    zip_safe=False,
)
