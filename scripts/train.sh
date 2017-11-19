#!/bin/bash
#
# Train a Neural Machine Translation model using Sockeye

if [ $# -ne 2 ]; then
    echo "Usage: train.sh hyperparams.txt device(gpu/cpu)"
    exit
fi

###########################################
# (0) Hyperparameter settings
# source hyperparams.txt to get text files and all training hyperparameters
source $1

# options for cpu vs gpu training (may need to modify for different grids)
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

###########################################
# (1) Book-keeping
mkdir -p $modeldir
#datenow=`date '+%Y-%m-%d %H:%M:%S'`
cp $1 $modeldir/hyperparams.txt

###########################################
# (2) train the model (this may take a while) 
python -m sockeye.train -s ${train_bpe}.$src \
                        -t ${train_bpe}.$trg \
                        -vs ${valid_bpe}.$src \
                        -vt ${valid_bpe}.$trg \
                        --num-embed $num_embed \
                        --rnn-num-hidden $rnn_num_hidden \
                        --attention-type $attention_type \
                        --max-seq-len $max_seq_len \
                        --checkpoint-frequency $checkpoint_frequency \
                        --num-words $num_words \
                        --word-min-count $word_min_count \
                        --use-tensorboard \
                        $device \
                        -o $modeldir
