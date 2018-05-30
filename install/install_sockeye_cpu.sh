#!/bin/bash
set -e

SOCKEYE_COMMIT=cce1acc825f5dfbcd5330756d6abe738b973b3f8 # 1.18.1 (sockeye:master)

# Get this version of sockeye
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule init
git submodule update --recursive --remote sockeye
cd sockeye
git checkout $SOCKEYE_COMMIT # version 1.18.1

$rootdir/install/install_sockeye_custom.sh -s $rootdir/sockeye -e sockeye_cpu -d cpu
