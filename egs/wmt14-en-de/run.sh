#!/bin/sh 

echo "0. Downloading data"
for lang in en de ; do
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.$lang
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.$lang
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.$lang
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.$lang
done

hyperparam_file=model1.hpm
source $hyperparam_file
echo "1. Get rootdir from hyperparam file: $rootdir"

echo "2. Running preprocess-BPE.sh"
bash $rootdir/scripts/preprocess-bpe.sh $hyperparam_file

echo "3. Starting training"
bash $rootdir/scripts/train.sh -p $hyperparam_file -e sockeye_gpu
#qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=240:00:00,num_proc=2 -j y $rootdir/scripts/train.sh -p $hyperparam_file -e sockeye_gpu

testset=newstest2014
echo "4. Decode $testset"
bash $rootdir/scripts/translate.sh -p $hyperparam_file -i $testset.en -o $testset.de.1best -e sockeye_gpu
#qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=2:00:00 -j y $rootdir/scripts/translate.sh -p $hyperparam_file -i $testset.en -o $testset.de.1best -e sockeye_gpu

echo "5. Compute BLEU:"
echo "$rootdir/tools/multi-bleu.perl $testset.de < $testset.de.1best 2> /dev/null"
