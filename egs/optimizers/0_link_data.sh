#!/bin/bash

# This example builds on the egs/ted recipe
# We use the de-en bitext. If this has not been prepared, 
# follow the instructions in egs/ted/ 

rootdir="$(readlink -f "$(dirname "$0")/../../")"
teddir=$rootdir/egs/ted/de-en/data-bpe
outdir=./

# 1. Link with downloaded and preprocessed data
if [ -e "$teddir/train.bpe-30000.en" ] ; then
    mkdir -p $outdir
    ln -s $teddir $outdir/data-bpe
else
    echo "Ted Talks de-en (MTTT) has not been prepared. Exiting.." 
    echo "Please generated bpe'd data in $teddir "
    exit 1
fi


