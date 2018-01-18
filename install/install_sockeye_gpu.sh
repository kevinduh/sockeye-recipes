#!/bin/bash
set -e

# 1. setup python virtual environment 
venv=sockeye_gpu # set your virtual enviroment name
conda create -y -n $venv python=3
source activate $venv

# 2. clone sockeye NMT as submodule and install
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule init
git submodule update sockeye
cd sockeye
git checkout 6c205805eca4d3e74b5a158c22509ba9ec847957
pip install mxnet-cu80==0.10.0
pip install pyaml numpy matplotlib tensorboard
python setup.py install

