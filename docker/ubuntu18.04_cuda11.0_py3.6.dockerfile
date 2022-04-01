#FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda@sha256:f6cb1146ee72111eab6f025b5996bea4a44a6e1319b82e036e876fbe1ca43cd9
LABEL maintainer="Alexander Cabello <alexander.cabello@algoritics.com>"

# install prereqs
RUN set -ex; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    software-properties-common \
    python3 \
    python3-pip \
    locales \
    zip \
    unzip \
    vim \
    libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0

# configure locale
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools

# copy and install satsim wheel file
ENV SATSIM_VERSION='0.12.0'
COPY dist/satsim-0.12.0-py2.py3-none-any.whl /tmp

# install python prereqs and satsim
RUN pip3 --no-cache-dir install \
        imageio==2.15.0 \
        tensorflow~=2.4.4 \
        tensorflow-addons~=0.14.0 \
        tmp/satsim-0.12.0-py2.py3-none-any.whl

RUN pip3 --no-cache-dir install \
        jupyterlab \
        scikit-learn \
        virtualenv

# copy and install satsim wheel file
ENV SATSIM_VERSION='0.12.0'
COPY dist/satsim-0.12.0-py2.py3-none-any.whl /tmp
RUN pip3 install tmp/satsim-0.12.0-py2.py3-none-any.whl

RUN mkdir /workspace
WORKDIR /workspace
COPY examples/ /workspace/examples/
COPY docs/_build/html/ /workspace/docs/

# expose jupyter lab
EXPOSE 8888

ENV SHELL=/bin/bash

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root