#!/bin/bash
# get gpu device id

module load cuda90/toolkit

if [ $CUDA_VISIBLE_DEVICES ]; then
    # if CUDA_VISIBLE_DEVICES is already set, just return 0
    echo 0
    echo "get-gpu.sh: On $HOSTNAME CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES found, so set device_id = 0." `date`
else

    # else, look for a free GPU
    freegpu=( $( nvidia-smi --format=noheader,csv --query-gpu=index,utilization.gpu | grep ', 0 %' | awk -F ',' '{print $1}' ) )

    numfree=${#freegpu[@]}

    if [ $numfree -gt 0 ]; then
	# randomly pick one gpu id to return
	pick=$(expr $RANDOM % $numfree )
	echo ${freegpu[$pick]}
	echo "get-gpu.sh: Picking device_id = ${freegpu[$pick]} from free GPUs: ${freegpu[@]}." `date`
    else
	# return -1 if no gpu is free
	echo -1
	echo "get-gpu.sh: No free GPUs." `date`
    fi
fi
