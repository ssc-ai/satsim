#FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda@sha256:bdfbf44c08855938e4396baac3cc6e28546aa16b45a70bac4fb20621d9867cf1
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
    libnvinfer-dev=8.0.0-1+cuda11.0 \
    libnvinfer-plugin8=8.0.0-1+cuda11.0

# configure locale
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools

# copy and install satsim wheel file
ENV SATSIM_VERSION='0.13.0'
COPY dist/satsim-0.13.0-py2.py3-none-any.whl /tmp

# install python prereqs and satsim
RUN pip3 --no-cache-dir install \
        imageio==2.15.0 \
        tensorflow~=2.8.0 \
        tensorflow-addons~=0.16.1 \
        tmp/satsim-0.13.0-py2.py3-none-any.whl

RUN pip3 --no-cache-dir install \
        jupyterlab \
        scikit-learn \
        virtualenv

# copy and install satsim wheel file
ENV SATSIM_VERSION='0.13.0'
COPY dist/satsim-0.13.0-py2.py3-none-any.whl /tmp
RUN pip3 install tmp/satsim-0.13.0-py2.py3-none-any.whl

RUN mkdir /workspace
WORKDIR /workspace
COPY examples/ /workspace/examples/
COPY docs/_build/html/ /workspace/docs/

# expose jupyter lab
EXPOSE 8888

ENV SHELL=/bin/bash

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root