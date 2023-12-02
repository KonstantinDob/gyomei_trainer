# Dockerfile based on: https://github.com/osai-ai/dokai
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV PYTHONPATH $PYTHONPATH:/workdir
WORKDIR /workdir

# Install python and apt-get packages
RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm \
    cmake unzip git wget tmux nano curl \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev &&\
    ln -sf /usr/bin/python3 /usr/bin/python &&\
    ln -sf /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# Install pip and setuptools
RUN pip3 install --upgrade --no-cache-dir \
    pip==21.3.1 \
    setuptools==59.5.0 \
    packaging==21.2 \
    protobuf==3.20.*

# Install python packages
COPY . /workdir
RUN pip3 install -r ./requirements.txt
RUN pip3 install -r ./tests/requirements.txt
