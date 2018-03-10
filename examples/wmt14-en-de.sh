#!/bin/sh 

echo "0. Downloading data"
for lang in en de ; do
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.$lang
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.$lang
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.$lang
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.$lang
done

hyperparam_file=hyperparams.wmt14-en-de.txt
source $hyperparam_file
echo "1. Get rootdir from hyperparam file: $rootdir"

echo "2. Running preprocess-BPE.sh"
sh $rootdir/scripts/preprocess-bpe.sh $hyperparam_file

echo "3. Starting training via qsub"
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=240:00:00 -j y -o train1.log $rootdir/scripts/train.sh $hyperparam_file gpu

echo "4. After training, run the following to decode"
testset=newstest2014
echo "qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=2:00:00 -j y -o translate1.log $rootdir/scripts/translate.sh $hyperparam_file $testset.en $testset.de.out gpu"

echo "And to compute BLEU:"
echo "$rootdir/tools/multi-bleu.perl $testset.de < $testset.de.out"
