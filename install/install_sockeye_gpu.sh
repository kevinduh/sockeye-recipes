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
git submodule update --recursive --remote sockeye
cd sockeye
git checkout cce1acc825f5dfbcd5330756d6abe738b973b3f8 # version 1.18.1
pip install -r requirements.gpu-cu80.txt
pip install . --no-deps

# 3. install optional dependencies
pip install mxboard
pip install tensorboard tensorflow
pip install matplotlib
