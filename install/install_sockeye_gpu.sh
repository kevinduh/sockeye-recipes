#!/bin/bash
set -e

# 1. setup python virtual environment 
venv=sockeye_gpu_dev # set your virtual enviroment name
conda create -y -n $venv python=3
source activate $venv

# 2. clone sockeye NMT as submodule and install
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule init
git submodule update sockeye
cd sockeye
git checkout 762ce78e4e49b9ba5d14eb0a48d97f19c8807707 # version 1.16.2 
pip install mxnet-cu80#==0.10.0
pip install pyaml numpy matplotlib tensorboard
python setup.py install

