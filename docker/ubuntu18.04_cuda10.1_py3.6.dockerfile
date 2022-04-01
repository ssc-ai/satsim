#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda@sha256:557de4ba2cb674029ffb602bed8f748d44d59bb7db9daa746ea72a102406d3ec
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
    libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1

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
        tensorflow~=2.2.3 \
        tensorflow-addons~=0.11.2 \
        tmp/satsim-0.12.0-py2.py3-none-any.whl

RUN pip3 --no-cache-dir install \
        jupyterlab \
        scikit-learn \
        virtualenv

RUN mkdir /workspace
WORKDIR /workspace
COPY examples/ /workspace/examples/
COPY docs/_build/html/ /workspace/docs/

# expose jupyter lab
EXPOSE 8888

ENV SHELL=/bin/bash

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
