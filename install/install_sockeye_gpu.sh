#!/bin/bash
set -e

echo "Hi. This is an install script from a non-master branch of sockeye-recipes."
echo "To prevent clobbering the existing sockeye_cpu/sockeye_gpu conda environment,"
echo "please provide a unique environment name (e.g. sockeye_gpu_dev) for this install:"
read envname
echo "OK. Installing to: " $envname

SOCKEYE_COMMIT=06394b66bace582f7728ada99d14aea93639707b # 1.18.84 (sockeye:master)

# Get this version of sockeye
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule init
git submodule update --recursive --remote sockeye
cd sockeye
git checkout $SOCKEYE_COMMIT

$rootdir/install/install_sockeye_custom.sh -s $rootdir/sockeye -e $envname
