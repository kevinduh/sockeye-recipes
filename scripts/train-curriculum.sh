#!/bin/bash
#
# Train a Neural Machine Translation model using Sockeye

if [ $# -ne 2 ]; then
    echo "Usage: train_curr.sh hyperparams.txt env"
    exit
fi

###########################################
# (0) Hyperparameter settings
# source hyperparams.txt to get text files and all training hyperparameters
source $1
venv=$2

source activate $venv
module load cuda80/toolkit
gpu_id=`$rootdir/scripts/get-gpu.sh`
device="--device-id $gpu_id"

###########################################
# (1) Book-keeping
mkdir -p $modeldir
cp $1 $modeldir/hyperparams.txt
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "Start training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "$0 $@" >> $modeldir/cmdline.log

###########################################
# (2.1) prepare the data
python -m sockeye.prepare_data -s $train_bpe_src \
                               -t $train_bpe_trg \
                               --sentence-score-file $score_file \
                               --o $modeldir/prepared_data

# (2.2) train the model (this may take a while) 
#python -m sockeye.train -s $train_bpe_src \
                        #-t $train_bpe_trg \
python -m sockeye.train -d $modeldir/prepared_data \
                        -vs $valid_bpe_src \
                        -vt $valid_bpe_trg \
                        --num-embed $num_embed \
                        --rnn-num-hidden $rnn_num_hidden \
                        --rnn-attention-type $rnn_attention_type \
                        --max-seq-len $max_seq_len \
                        --checkpoint-frequency $checkpoint_frequency \
                        --num-words $num_words \
                        --word-min-count $word_min_count \
                        --num-layers $num_layers \
                        --rnn-cell-type $rnn_cell_type \
                        --batch-size $batch_size \
                        --min-num-epochs $min_num_epochs \
                        --max-num-epochs $max_num_epochs \
                        --max-updates $max_updates \
                        --embed-dropout $embed_dropout \
                        --rnn-dropout-inputs $rnn_dropout_inputs \
                        --rnn-dropout-states $rnn_dropout_states \
                        --keep-last-params $keep_last_params \
                        --disable-device-locking \
                        --decode-and-evaluate $decode_and_evaluate \
                        --decode-and-evaluate-use-cpu \
                        --encoder rnn \
                        --decoder rnn \
                        --batch-type sentence \
                        --curriculum-training \
                        --curriculum-update-freq $curriculum_update_freq \
                        $device \
                        -o $modeldir




##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "===========================================" >> $modeldir/cmdline.log
