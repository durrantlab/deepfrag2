FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04

RUN apt update && apt install -y wget unzip curl bzip2 git gcc
RUN rm -f /miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-py38_4.10.3-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -y pip
RUN python3 -m pip install --upgrade pip

RUN apt-get update && \
    apt-get -y install sudo
RUN apt-get -y install vim
RUN apt-get -y install g++

RUN python --version

COPY requirements.txt /tmp/requirements.txt
RUN cd /tmp/; /miniconda/bin/python3 -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN conda install -y -c conda-forge rdkit

WORKDIR /mnt
