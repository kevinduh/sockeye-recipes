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
    source activate sockeye_cpu
    device="--use-cpu"
else
    source activate sockeye_gpu
    module load cuda80/toolkit
    device="--device-id 1"
fi

rootdir="$(readlink -f "$(dirname "$0")/../")"
subword=$rootdir/tools/subword-nmt/

### Apply BPE to input, run Sockeye.translate, then de-BPE ###
python $subword/apply_bpe.py --input $input --codes $bpe_vocab_src | \
    python -m sockeye.translate --models $modeldir $device --max-input-len 100  | \
    sed -r 's/@@( |$)//g' > $output
