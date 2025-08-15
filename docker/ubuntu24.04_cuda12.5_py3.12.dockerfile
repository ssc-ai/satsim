FROM ubuntu:24.04
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
    wget

# configure locale
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# upgrade pip
RUN python3 -m pip config set global.break-system-packages true

# install tensorflow that works with installed cuda version
RUN pip3 --no-cache-dir install tensorflow[and-cuda]~=2.19.0

# copy wheel file
ENV SATSIM_VERSION='0.21.2'
COPY dist/satsim-0.21.2-py2.py3-none-any.whl /tmp

# install satsim wheel file together with jupyterlab so dependency compatibility are resolved
RUN pip3 --no-cache-dir install \
        tmp/satsim-0.21.2-py2.py3-none-any.whl \
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