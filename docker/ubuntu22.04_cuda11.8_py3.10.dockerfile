#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda@sha256:bd746eb3b9953805ebe644847a227e218b5da775f47007c69930569a75c9ad7d
LABEL maintainer="Alexander Cabello <alexander.cabello@algoritics.com>"

# install prereqs
RUN set -ex; \
  apt-get update; \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    python3 \
    python3-dev \
    python3-pip \
    locales \
    zip \
    unzip \
    vim \
    libnvinfer8=8.6.1.6-1+cuda11.8 \
    libnvinfer-plugin8=8.6.1.6-1+cuda11.8

# configure locale
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# upgrade pip
RUN pip3 --no-cache-dir install --upgrade pip setuptools

# install tensorflow that works with installed cuda version
RUN pip3 --no-cache-dir install tensorflow~=2.14.0

# copy wheel file
ENV SATSIM_VERSION='0.17.1'
COPY dist/satsim-0.17.1-py2.py3-none-any.whl /tmp

# install satsim wheel file together with jupyterlab so dependency compatibility are resolved
RUN pip3 --no-cache-dir install \
        tmp/satsim-0.17.1-py2.py3-none-any.whl \
        jupyterlab \
        scikit-learn \
        virtualenv

# cache astrometry files
RUN echo "from astropy.coordinates import SkyCoord, ICRS, ITRS\nfrom astropy.time import Time\nsc = SkyCoord(ra=0, dec=0, unit='deg', frame=ICRS)\nsc2 = sc.transform_to(ITRS(obstime=Time.now()))" | python3
RUN mkdir /root/.skyfield
RUN cd /root/.skyfield && echo "from skyfield.api import load\nload('de421.bsp')" | python3
ENV SATSIM_SKYFIELD_LOAD_DIR='/root/.skyfield'

# setup local path
RUN mkdir -p /root/.local
RUN echo "PATH=$PATH:/root/.local/bin" >> /root/.profile

# setup workspace
RUN mkdir /workspace
WORKDIR /workspace
COPY examples/ /workspace/examples/
COPY docs/_build/html/ /workspace/docs/

# expose jupyter lab
EXPOSE 8888

ENV SHELL=/bin/bash

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root