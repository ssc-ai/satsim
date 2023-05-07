#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'Click~=8.1',
    'Cython~=0.29.0',
    'scikit-image~=0.19.3',
    'numpy>=1.21.0',
    'PyYAML~=6.0',
    'matplotlib~=3.7.0',
    'astropy==5.2.1',
    'apng~=0.3.4',
    'scipy~=1.10',
    'tensorflow<=2.11',
    'tensorflow-addons<=0.19.0',
    'skyfield==1.45',
    'sgp4==2.21',
    'poliastro==0.17.0',
    'poppy==1.0.3',
    'diskcache==5.0.3',
    'pydash~=4.9.0',
    'asdf==2.14.3',
    'pygc==1.3.0',
    'numba~=0.56.0',
    'pooch~=1.6.0',
    'czmlpy~=0.9.0',
    'tifffile==2023.4.12',
    'imagecodecs==2023.3.16',
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
    url='https://github.com/ssc-ai/satsim',
    version='0.15.1',
    zip_safe=False,
)
