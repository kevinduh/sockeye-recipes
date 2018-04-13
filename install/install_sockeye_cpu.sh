#!/bin/bash
set -e

# 1. setup python virtual environment 
venv=sockeye_cpu # set your virtual enviroment name
conda create -y -n $venv python=3
source activate $venv

# 2. clone sockeye NMT as submodule and install
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule init
git submodule update --recursive --remote sockeye
cd sockeye
git checkout 762ce78e4e49b9ba5d14eb0a48d97f19c8807707 # version 1.16.2
pip install -r requirements.txt
python setup.py install

# 3. install optional dependencies
pip install tensorboard==1.0.0a6
pip install matplotlib


