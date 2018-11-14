#!/bin/bash
set -e

SOCKEYE_COMMIT=59180f3f6c52aa3169e5cd04c03d4e42e7f9c76d # 1.18.57 (sockeye:master)

# Get this version of sockeye
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule init
git submodule update --recursive --remote sockeye
cd sockeye
git checkout $SOCKEYE_COMMIT

$rootdir/install/install_sockeye_custom.sh -s $rootdir/sockeye -e sockeye_cpu -d cpu

