#!/bin/bash

# query for gpu with 0% utilization
module load cuda80/toolkit
numgpu=$1

if [ $CUDA_VISIBLE_DEVICES ]; then
    # if CUDA_VISIBLE_DEVICES is already set, just return 0
    echo "-$numgpu"
else

    # else, look for a single free GPU (Doesn't support multiGPU in this case)
    freegpu=( $( nvidia-smi --format=noheader,csv --query-gpu=index,utilization.gpu | grep ', 0 %' | awk -F ',' '{print $1}' ) )

    numfree=${#freegpu[@]}

    if [ $numfree -gt 0 ]; then
	# randomly pick one gpu id to return
	pick=$(expr $RANDOM % $numfree )
	echo ${freegpu[$pick]}
    else
	# return -1 if no gpu is free
	echo -1
    fi
fi
