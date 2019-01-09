#!/bin/bash
#
# Train a Neural Machine Translation model using Sockeye
# Uses pre-trained embeddings

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: train-embeddings.sh -p hyperparams.txt -e ENV_NAME [-d DEVICE]"
  errcho "Device is optional and inferred from env"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}

function check_var_set_and_file_exists() {
  if [ -z ${1+x} ]; then
    return false
  fi
  if [ ! -f $1 ]; then
    return false
  fi
  return true
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
mkdir -p $modeldir

###########################################
# (1) Get bitext vocab
# This will be used to filter the pre-trained embeddings
echo "################ Creating bitext vocab #####################"
python3 -m sockeye.vocab \
          -i $train_bpe_src \
          -o $modeldir/vocab.src.0.json

python3 -m sockeye.vocab \
          -i $train_bpe_trg \
          -o $modeldir/vocab.tgt.0.json

###########################################
# (2) Convert the pre-trained embeddings into a sockeye-compatible
# format and initialize the word embedding matrices with these embeddings
# If the embedding files are not available a dummy vec file will be created
# This leads to randomly initialized embeddings for that language
# This has the pleasant side effect of backing off to the default
# sockeye behavior of randomly initializing both source and target
# when no embeddings are provided
echo "################ Processing pre-trained embeddings #####################"

if [[ -z ${embeddings_src+x} ]]; then
  # Create dummy file
  embeddings_src=$modeldir/dummy.${src}.vec
  touch $embeddings_src
fi
if [[ -z ${embeddings_trg+x} ]]; then
  # Create dummy file
  embeddings_trg=$modeldir/dummy.${trg}.vec
  touch $embeddings_trg
fi

check_file_exists $embeddings_src
check_file_exists $embeddings_trg

# Convert fasttest format files to npy
python $rootdir/scripts/util/vec2npy.py \
  $embeddings_src $embeddings_src
python $rootdir/scripts/util/vec2npy.py \
  $embeddings_trg $embeddings_trg

# Sanity check
check_file_exists ${embeddings_src}.npy
check_file_exists ${embeddings_src}.vocab
check_file_exists ${embeddings_trg}.npy
check_file_exists ${embeddings_trg}.vocab

echo "################ Initializing Sockeye params #####################"
python3 -m sockeye.init_embedding        \
          -w ${embeddings_src}.npy ${embeddings_trg}.npy    \
          -i ${embeddings_src}.vocab ${embeddings_trg}.vocab   \
          -o $modeldir/vocab.src.0.json $modeldir/vocab.tgt.0.json  \
          -n source_embed_weight target_embed_weight \
          -f $modeldir/params.init

###########################################
# (3) Book-keeping
# options for cpu vs gpu training (may need to modify for different grids)
source $rootdir/scripts/get-device.sh $DEVICE ""
cp $HYP_FILE $modeldir/hyperparams.txt

datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "Start training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "$0 $@" >> $modeldir/cmdline.log
echo "$devicelog" >> $modeldir/cmdline.log

# get model specific arguments 
source $rootdir/scripts/model-specific-args.sh $HYP_FILE

###########################################
# (4) Train the model (this may take a while) 
python -m sockeye.train -s $train_bpe_src \
                        -t $train_bpe_trg \
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
                        --learning-rate-reduce-num-not-improved $max_num_checkpoint_not_improved \
                        --learning-rate-scheduler-type plateau-reduce \
                        --learning-rate-decay-optimizer-states-reset best \
                        --learning-rate-decay-param-reset \
                        --loss $loss \
                        --seed $seed \
                        --params $modeldir/params.init \
                        --allow-missing-params \
                        --source-vocab $modeldir/vocab.src.0.json \
                        --target-vocab $modeldir/vocab.tgt.0.json \
                        $device \
                        -o $modeldir


##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "===========================================" >> $modeldir/cmdline.log
