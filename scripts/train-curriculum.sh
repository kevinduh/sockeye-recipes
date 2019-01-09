#!/bin/bash
#
# Train a Neural Machine Translation model using Sockeye, with Curriculum Learning

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

# get model specific arguments 
source $rootdir/scripts/model-specific-args.sh $HYP_FILE

###########################################
# (2.1) Prepare the data into different shards by curriculum score file
python -m sockeye.prepare_data -s $train_bpe_src \
                               -t $train_bpe_trg \
                               --curriculum-score-file $score_file \
                               --o $modeldir/prepared_data

# (2.2) Train the model (this may take a while) 
python -m sockeye.train -d $modeldir/prepared_data \
                        --curriculum-training \
                        --curriculum-update-freq $curriculum_update_freq \
                        -vs $valid_bpe_src \
                        -vt $valid_bpe_trg \
                        --encoder $encoder \
                        --decoder $decoder \
                        $modelargs \
                        --num-embed $num_embed \
                        --num-layers $num_layers \
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
                        --initial-learning-rate $initial_learning_rate  \
                        --label-smoothing $label_smoothing \
                        --batch-type word \
                        --optimizer $optimizer \
                        --gradient-clipping-threshold 1.0 \
                        --gradient-clipping-type abs \
                        --learning-rate-reduce-factor $learning_rate_reduce_factor \
                        --learning-rate-reduce-num-not-improved  $max_num_checkpoint_not_improved \
                        --learning-rate-scheduler-type plateau-reduce \
                        --learning-rate-decay-optimizer-states-reset best \
                        --learning-rate-decay-param-reset \
                        --loss $loss \
                        --seed $seed \
                        $device \
                        -o $modeldir




##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "===========================================" >> $modeldir/cmdline.log
