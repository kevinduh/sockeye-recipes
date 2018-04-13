#!/bin/bash
set -e

# install subword nmt
git submodule init
git submodule update --recursive --remote tools/subword-nmt

# install moses scripts
rootdir="$(readlink -f "$(dirname "$0")/../")"
mkdir -p $rootdir/tools/
for i in multi-bleu.perl ; do
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/$i -O $rootdir/tools/$i
    chmod u+x $rootdir/tools/$i
done

