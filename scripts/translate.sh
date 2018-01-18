#!/bin/bash


if [ $# -ne 5 ]; then
    echo "Usage: translate.sh modeldir bpe_vocab_src input output device(cpu/gpu)"
    exit
fi


modeldir=$1
bpe_vocab_src=$2
input=$3
output=$4

if [ $5 == "cpu" ]; then
    source activate sockeye_cpu_dev
    device="--use-cpu"
else
    source activate sockeye_gpu_dev
    module load cuda80/toolkit
    gpu_id=`$rootdir/scripts/get-gpu.sh`
    device="--device-id $gpu_id"
fi

subword=$rootdir/tools/subword-nmt/

### Apply BPE to input, run Sockeye.translate, then de-BPE ###
python $subword/apply_bpe.py --input $input --codes $bpe_vocab_src | \
    python -m sockeye.translate --models $modeldir $device --max-input-len 100  | \
    sed -r 's/@@( |$)//g' > $output
