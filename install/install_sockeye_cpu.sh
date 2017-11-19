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
git submodule update sockeye
cd sockeye
pip install -e '.[optional]'

