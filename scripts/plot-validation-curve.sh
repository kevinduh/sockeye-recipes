#!/bin/bash


if [ $# -ne 2 ]; then
    echo "Usage: translate.sh hyperparams.txt device(cpu/gpu)"
    exit
fi


source $1

if [ $2 == "cpu" ]; then
    source activate sockeye_cpu
    device="--use-cpu"
else
    source activate sockeye_gpu
    module load cuda80/toolkit
    #device="--device-id 1"
    gpu_id=`/home/hltcoe/kduh/src/mt/sockeye-recipes/scripts/get-gpu.sh`
    device="--device-id $gpu_id"
fi


# basic settings
#rootdir="$(readlink -f "$(dirname "$0")/../")"
rootdir=/home/hltcoe/kduh/src/mt/sockeye-recipes
multibleu=$rootdir/tools/multi-bleu.perl
step=1
resultlog=$modeldir/multibleu.valid_bpe.result
rm $resultlog

# loop over each checkpoint
for c in `seq 1 $step 9999`; do
    checkpoint=`printf %04d $c`

    # quit if checkpoint does not exist
    if [ ! -f $modeldir/params.$checkpoint ]; then
	break
    fi

    # translate validation set using checkpoint model
    output=$modeldir/out.valid_bpe.$checkpoint
    if [ ! -f $output ]; then
	echo `date` "Translating with model $checkpoint"
	python -m sockeye.translate --models $modeldir --checkpoints $checkpoint $device < $valid_bpe.$src > $output
    fi
    
    # compute bleu (note this is multi-bleu on bpe so may not be comparable in all tokenization settings; this is mainly for evaluating the learning curve)
    echo -n "$checkpoint " >> $resultlog
    $multibleu $valid_bpe.$trg < $output >> $resultlog
done
