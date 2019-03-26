#!/bin/bash

# This example builds on the egs/ted recipe
# We use the de-en bitext. If this has not been prepared, 
# follow the instructions in egs/ted/ 

rootdir="$(readlink -f "$(dirname "$0")/../../")"
teddir=$rootdir/egs/ted/de-en/data-bpe
outdir=de-en

# 1. Link with downloaded and preprocessed data
if [ -e "$teddir/train.bpe-30000.en" ] ; then
    mkdir -p $outdir
    ln -s $teddir $outdir/data-bpe
else
    echo "Ted Talks de-en (MTTT) has not been prepared. Exiting.." 
    echo "Please generated bpe'd data in $teddir "
    exit 1
fi

# 2. Download example curriculum score file for the above bitext
scorefile=https://raw.githubusercontent.com/Este1le/curriculum_learning_scores/master/scores/data_scores_ted/ted.sentence_average_rank.de.bk
wget $scorefile -O $outdir/curriculum_sent.scores

echo "Make sure the #lines in scores and bitext match up:"
wc -l $outdir/curriculum_sent.scores
wc -l $outdir/data-bpe/train.bpe-30000.en


