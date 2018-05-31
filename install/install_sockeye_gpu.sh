#!/bin/bash
set -e

SOCKEYE_COMMIT=5873da5bd640f862646ae59cf408ddcee1449777 # 1.18.15 (sockeye:master)

# Get this version of sockeye
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule init
git submodule update --recursive --remote sockeye
cd sockeye
git checkout $SOCKEYE_COMMIT

$rootdir/install/install_sockeye_custom.sh -s $rootdir/sockeye -e sockeye_gpu
