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
    source activate sockeye_cpu_dev
    device="--use-cpu"
else
    source activate sockeye_gpu_dev
    module load cuda80/toolkit
    gpu_id=`$rootdir/scripts/get-gpu.sh`
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
                        --rnn-attention-type $attention_type \ # it was --attention-type in version 1.7.1 (old version)
                        --max-seq-len $max_seq_len \
                        --checkpoint-frequency $checkpoint_frequency \
                        --num-words $num_words \
                        --word-min-count $word_min_count \
                        --max-updates $max_updates \
                        --num-layers $num_layers \
                        --rnn-cell-type $rnn_cell_type \
                        --batch-size $batch_size \
                        --min-num-epochs $min_num_epochs \
                        --embed-dropout $embed_dropout \
                        --keep-last-params $keep_last_params \
                        --use-tensorboard \
                        $device \
                        -o $modeldir
