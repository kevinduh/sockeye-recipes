#!/bin/bash
#
# Train a Neural Machine Translation model using Sockeye

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: train.sh -p hyperparams.txt -e ENV_NAME [-d DEVICE]"
  errcho "Device is optional and inferred from env"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}

while getopts ":h?p:e:d:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    p) HYP_FILE=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
    d) DEVICE=$OPTARG
      ;;
  esac
done

if [[ -z $HYP_FILE || -z $ENV_NAME ]]; then
  errcho "Missing arguments"
  show_help
  exit 1
fi

###########################################
# (0) Setup
# source hyperparams.txt to get text files and all training hyperparameters
check_file_exists $HYP_FILE
source $HYP_FILE
source activate $ENV_NAME

# options for cpu vs gpu training (may need to modify for different grids)
source $rootdir/scripts/get-device.sh $DEVICE ""

###########################################
# (1) Book-keeping
mkdir -p $modeldir
cp $HYP_FILE $modeldir/hyperparams.txt
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "Start training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "$0 $@" >> $modeldir/cmdline.log
echo "$devicelog" >> $modeldir/cmdline.log

###########################################
# (2) Train the model (this may take a while) 
python -m sockeye.train -s $train_bpe_src \
                        -t $train_bpe_trg \
                        -vs $valid_bpe_src \
                        -vt $valid_bpe_trg \
                        --encoder rnn \
                        --decoder rnn \
                        --num-embed $num_embed \
                        --num-layers $num_layers \
                        --rnn-num-hidden $rnn_num_hidden \
                        --rnn-attention-type $rnn_attention_type \
                        --rnn-cell-type $rnn_cell_type \
                        --rnn-dropout-inputs $rnn_dropout_inputs \
                        --rnn-dropout-states $rnn_dropout_states \
                        --embed-dropout $embed_dropout \
                        --max-seq-len $max_seq_len \
                        --num-words $num_words \
                        --word-min-count $word_min_count \
                        --checkpoint-frequency $checkpoint_frequency \
                        --batch-size $batch_size \
                        --min-num-epochs $min_num_epochs \
                        --max-num-epochs $max_num_epochs \
                        --max-updates $max_updates \
                        --keep-last-params $keep_last_params \
                        --disable-device-locking \
                        --decode-and-evaluate $decode_and_evaluate \
                        --decode-and-evaluate-use-cpu \
                        --initial-learning-rate 0.0002 \
                        --label-smoothing 0.1 \
                        --batch-type sentence \
                        --optimizer adam \
                        --gradient-clipping-threshold 1.0 \
                        --gradient-clipping-type abs \
                        --learning-rate-reduce-factor 0.7 \
                        --learning-rate-reduce-num-not-improved 8 \
                        --learning-rate-scheduler-type plateau-reduce \
                        --learning-rate-decay-optimizer-states-reset best \
                        --learning-rate-decay-param-reset \
                        --loss cross-entropy \
                        --seed 13 \
                        $device \
                        -o $modeldir




##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "===========================================" >> $modeldir/cmdline.log
