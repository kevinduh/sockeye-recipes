#!/bin/bash
#
# Auto-tune a Neural Machine Translation model 
# Using Sockeye and CMA-ES algorithm

# source hyperparameters.txt
source $1

#root directory of the auto-tuning scripts
autotunedir=$2

# options for cpu vs gpu training
device=$3

# path to the current generation folder
generation_path=$4

# current population 
n_population=$5

# path to the gene file
gene=$6

# current generation
n_generation=$7


if [ $device == "cpu" ]; then
    source activate sockeye_cpu_dev
    device="--use-cpu"
else
    source activate sockeye_gpu_dev
    module load cuda80/toolkit
    gpu_id=`$rootdir/scripts/get-gpu.sh`
    device="--device-id $gpu_id"
fi

model_path="${generation_path}model_$(printf "%02d" "$n_population")/"
# path to evaluation score path
eval_scr="${model_path}multibleu.valid_bpe.result"

mkdir $model_path

# update the tuned hyperparameters
source $(printf ${gene} $(printf "%02d" ${n_population}))

# train the model
$py_cmd -m sockeye.train -s ${train_bpe}.$src \
                        -t ${train_bpe}.$trg \
                        -vs ${valid_bpe}.$src \
                        -vt ${valid_bpe}.$trg \
                        --num-embed $num_embed \
                        --rnn-num-hidden $rnn_num_hidden \
                        --rnn-attention-type $attention_type \
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
                        --rnn-dropout-inputs $rnn_dropout_inputs \
                        --rnn-dropout-states $rnn_dropout_states \
                        --rnn-decoder-hidden-dropout $rnn_decoder_hidden_dropout \
                        --initial-learning-rate $initial_learning_rate \
                        --keep-last-params $keep_last_params \
                        --use-tensorboard \
                        $device \
                        -o $model_path
#                       --decode-and-evaluate -1 \
#                       --decode-and-evaluate-use-cpu \

# compute bleu on validation set
# basic settings
multibleu=$rootdir/tools/multi-bleu.perl
resultlog=${model_path}multibleu.valid_bpe.result
# use the checkpoint that has the best params
output = ${model_path}out.valid_bpe.best
if [ ! -f $output ]; then
python -m sockeye.translate --models ${model_path} $device < $valid_bpe.$src > $output
fi
# compute bleu
$multibleu $valid_bpe.$trg < $output >> $resultlog

# report the score
$py_cmd $autotunedir/reporter.py \
        --trg ${generation_path}genes.scr \
        --scr $eval_scr \
        --pop $population \
        --n-pop $n_population \
        --n-gen $n_generation \
        --model-path $model_path