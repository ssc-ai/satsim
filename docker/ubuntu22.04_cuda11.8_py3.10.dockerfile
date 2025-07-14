#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda@sha256:f6913f3c02f297877f6859d12ff330043c0be668fdad86868c29a239a5a82151
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
    wget \
    libnvinfer8=8.6.1.6-1+cuda11.8 \
    libnvinfer-plugin8=8.6.1.6-1+cuda11.8

# configure locale
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# upgrade pip
RUN pip3 --no-cache-dir install --upgrade pip setuptools

# install tensorflow that works with installed cuda version
RUN pip3 --no-cache-dir install tensorflow~=2.13.0

# copy wheel file
ENV SATSIM_VERSION='0.20.4'
COPY dist/satsim-0.20.4-py2.py3-none-any.whl /tmp

# install satsim wheel file together with jupyterlab so dependency compatibility are resolved
RUN pip3 --no-cache-dir install \
        tmp/satsim-0.20.4-py2.py3-none-any.whl \
        jupyterlab \
        scikit-learn \
        virtualenv

# cache astrometry files
RUN echo "from astropy.coordinates import SkyCoord, ICRS, ITRS\nfrom astropy.time import Time\nsc = SkyCoord(ra=0, dec=0, unit='deg', frame=ICRS)\nsc2 = sc.transform_to(ITRS(obstime=Time.now()))" | python3
RUN mkdir /root/.skyfield
RUN cd /root/.skyfield && wget --no-check-certificate https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp
ENV SATSIM_SKYFIELD_LOAD_DIR='/root/.skyfield'

# setup local path
RUN mkdir -p /root/.local
RUN echo "PATH=$PATH:/root/.local/bin" >> /root/.profile

# setup workspace
RUN mkdir /workspace
WORKDIR /workspace
COPY examples/ /workspace/examples/
COPY docs/_build/html/ /workspace/docs/

# other env
ENV SHELL=/bin/bash
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

# expose jupyter lab
EXPOSE 8888

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root