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
git checkout 17b4d3229067414bec527b5c7b6c70fd43b32cf6
pip install mxnet-cu80==0.10.0
pip install pyaml numpy matplotlib tensorboard
python setup.py install

