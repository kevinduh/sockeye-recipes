#!/bin/bash
#
# Specify model specific arguments for Neural Machine Translation models trained using Sockeye

function errcho() {
    >&2 echo $1
}

function show_help() {
    errcho "Usage: model-specific-args.sh hyperparameter.txt"
    errcho "Use with train scripts."
    errcho ""
}

function check_file_exists() {
    if [ ! -f $1 ]; then
        errcho "FATAL: Could not find file $1"
	exit 1
    fi
}

if [[ -z $1 ]]; then
  errcho "Missing arguments"
  show_help
  exit 1
fi

HYP_FILE=$1

check_file_exists $HYP_FILE
source $HYP_FILE

modelargs=""

if [ "$encoder" -eq "rnn" ] || [ "$decoder" -eq "rnn" ] ; then
	modelargs="--rnn-num-hidden $rnn_num_hidden \
           	   --rnn-attention-type $rnn_attention_type \
               --rnn-cell-type $rnn_cell_type \
               --rnn-dropout-inputs $rnn_dropout_inputs \
               --rnn-dropout-states $rnn_dropout_states"
fi

if [ "$encoder" -eq "cnn" ] || [ "$decoder" -eq "cnn" ] ; then
	modelargs="$modelargs \
			   --cnn-kernel-width $cnn_kernel_width \
	           --cnn-num-hidden $cnn_num_hidden \
	           --cnn-activation-type $cnn_activation_type \
	           --cnn-positional-embedding-type learned \
	           --cnn-project-qkv \
	           --cnn-hidden-dropout 0.2"
fi

if [ "$encoder" -eq "transformer" ] || [ "$decoder" -eq "transformer" ] ; then
	modelargs="$modelargs \
			   --transformer-model-size $transformer_model_size \
	           --transformer-attention-heads $transformer_attention_heads \
	           --transformer-feed-forward-num-hidden $transformer_feed_forward_num_hidden \
	           --transformer-positional-embedding-type fixed \
	           --transformer-preprocess n \
	           --transformer-postprocess dr \
	           --transformer-dropout-attention 0.1 \
	           --transformer-dropout-act 0.1 \
	           --transformer-dropout-prepost 0.1"
fi

