#!/bin/bash
#
# Auto-tune a Neural Machine Translation model 
# Using Sockeye and CMA-ES algorithm

if [ $# -ne 2 ]; then
    echo "Usage: auto-tune.sh hyperparams.txt device(gpu/cpu)"
    exit
fi

# export all variables
set -a 

###########################################
# (0) Hyperparameter settings
# source hyperparams.txt to get text files and all training hyperparameters
source $1

# options for cpu vs gpu training (may need to modify for different grids)
device=$2

###########################################
# (1) Hyperparameter auto-tuning
# exit when max generation reached
for ((n_generation=$n_generation;n_generation<$generation;n_generation++))
    do
        ###########################################
        # (1.1) set path and create folders
        if [ ! -d $checkpoint_path ]; then
          mkdir $checkpoint_path
        fi

        # path to current generation folder
        generation_path="${generation_dir}generation_$(printf "%02d" "$n_generation")/"
        
        # path to previous generation folder
        prev_generation_path="${prev_generation_dir}generation_$(printf "%02d" "`expr $n_generation - 1`")/"
        
        # path to current genes folder
        gene_path="${generation_path}genes/"

        # path to gene files
        gene="${gene_path}/%s.gene"

        mkdir $generation_path
        mkdir $gene_path

        ###########################################
        # (1.2) generate and record genes for current generation
        # save current generation information as a checkpoint
        $py_cmd evo_single.py \
        --checkpoint $checkpoint \
        --gene $gene \
        --params ${params} \
        --map-func $map_func \
        --scr ${prev_generation_path}genes.scr \
        --pop $population \
        --n-gen $n_generation 
        
        ###########################################
        # (1.3) train models described by model description file in current generation 
        $pycmd parallel.py \
        --pop $population \
        --num-devices $num-devices

        done

    done
        # Finished
