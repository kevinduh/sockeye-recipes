#!/bin/bash
# Install a custom version of sockeye in a conda environment
# CUDA 10.0

set -e

CONDA_HOME=$HOME/.conda/envs

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Install environment for custom sockeye"
  errcho "usage: install_sockeye_custom.sh [-h] -s SOCKEYE_LOCATION -e ENV_NAME [-f] [-d DEVICE_NAME]"
  errcho ""
}

function check_dir_exists() {
  if [ ! -d $1 ]; then
    errcho "FATAL: Could not find directory $1"
    exit 1
  fi
}

DEVICE=gpu
FORCE_NEW_ENV=false

while getopts ":h?s:e:fd:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    s) SOCKEYE=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
    f) FORCE_NEW_ENV=true
      ;;
    d) DEVICE=$OPTARG
      ;;
  esac
done

if [[ -z $SOCKEYE || -z $ENV_NAME ]]; then
  errcho "Missing arguments"
  show_help
  exit 1
fi

if [[ "$FORCE_NEW_ENV" == true ]]; then
  BASE_ENV_NAME=$ENV_NAME
  suffix=0
  while [ -d $CONDA_HOME/$ENV_NAME ]; do
    ENV_NAME=${BASE_ENV_NAME}_${suffix}
    suffix=$((suffix+1))
  done

  if [ $ENV_NAME != $BASE_ENV_NAME ]; then
    errcho "$BASE_ENV_NAME was already taken; we will use $ENV_NAME"
  fi
fi

check_dir_exists $SOCKEYE

# 1. setup python virtual environment 
venv=$ENV_NAME # set your virtual enviroment name
if [[ "$FORCE_NEW_ENV" == true || ! -d $CONDA_HOME/$ENV_NAME ]]; then
  errcho "Creating new Conda env : $ENV_NAME"
  conda create -y -n $venv python=3.6
fi

source activate $venv
export PYTHONNOUSERSITE=1

# 2. clone sockeye NMT as submodule and install
cd $SOCKEYE
if [[ "$DEVICE" == "gpu" ]]; then
  pip install -r requirements/requirements.gpu-cu100.txt
elif [[ "$DEVICE" == "cpu" ]]; then
  pip install -r requirements/requirements.txt
else
  errcho "Invalid device name; must be one of cpu or gpu"
  exit 1
fi
pip install . --no-deps

# 3. install optional dependencies
pip install mxboard
pip install tensorboard tensorflow
pip install matplotlib


