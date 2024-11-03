#FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda@sha256:081a31f56decc6c460df1808ed1c4867fb30b0fbfa8929258b10e3e5d6dc1a2e
LABEL maintainer="Alexander Cabello <alexander.cabello@algoritics.com>"

# install prereqs
RUN set -ex; \
  apt-get update; \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    python3 \
    python3-pip \
    locales \
    zip \
    unzip \
    vim \
    libnvinfer8=8.0.0-1+cuda11.0 \
    libnvinfer-plugin8=8.0.0-1+cuda11.0

# configure locale
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# upgrade pip
RUN pip3 --no-cache-dir install --upgrade pip setuptools

# install tensorflow that works with installed cuda version
RUN pip3 --no-cache-dir install tensorflow~=2.11.0

# copy wheel file
ENV SATSIM_VERSION='0.19.3'
COPY dist/satsim-0.19.3-py2.py3-none-any.whl /tmp

# install satsim wheel file together with jupyterlab so dependency compatibility are resolved
RUN pip3 --no-cache-dir install \
        tmp/satsim-0.19.3-py2.py3-none-any.whl \
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

# fix linking issues
RUN ln -s /usr/local/cuda/lib64/libnvrtc.so.11.2 /usr/local/cuda/lib64/libnvrtc.so.11.0
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer.so.7
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

# other env
ENV SHELL=/bin/bash
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

# expose jupyter lab
EXPOSE 8888

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root