#!/usr/bin/env bash

# Sets $device variable to --use-cpu if CPU or --use-device-id X if GPU
#
# The default is to infer whether to use CPU or GPU based on
# the absence or presence of mxnet-cu* library. 
# Optional 1st argument (DESIRED_DEVICE) can overide this auto-check.
#
# If GPU is desired, the script then picks the proper device id based 
# first on $CUDA_VISIBLE_DEVICES if set, and if not it will choose 
# a free GPU randomly based on nvidia-smi result
# All this assumes we use --disable-device-locking in Sockeye

DESIRED_DEVICE=$1
if [[ -z $DESIRED_DEVICE ]]; then
    # Assume that a conda env is enabled;
    # Infer device from env
    pip freeze 2> /dev/null | grep mxnet-cu* &> /dev/null
    if [ $? -eq 0 ]; then
	DEVICE="gpu"
    else
	DEVICE="cpu"
    fi
else
    DEVICE=$DESIRED_DEVICE
fi

# Select device to use
if [ "$DEVICE" == "cpu" ]; then
    # Use CPU
    device="--use-cpu"
    devicelog="get-device.sh: On $HOSTNAME Using CPU."
    
elif [ "$DEVICE" == "gpu" ]; then
    # Use GPU
    module load cuda90/toolkit
    
    if [ $CUDA_VISIBLE_DEVICES ]; then
	# if CUDA_VISIBLE_DEVICES is already set, just return 0,1,...,#maxid
	visible=(${CUDA_VISIBLE_DEVICES//,/ })
	maxid=$(expr "${#visible[@]}" - 1)
	visible_mapping=$(seq -s ' ' 0 $maxid)
	device="--device-ids $visible_mapping"
	devicelog="get-device.sh: On $HOSTNAME CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES found, map to device-ids = $visible_mapping. `nvidia-smi`"
	
    else
	# else, look for a free GPU
	freegpu=( $( nvidia-smi --format=noheader,csv --query-gpu=index,utilization.gpu | grep ', 0 %' | awk -F ',' '{print $1}' ) )
	
	numfree=${#freegpu[@]}
	
	if [ $numfree -gt 0 ]; then
            # randomly pick one gpu id to return
            pick=$(expr $RANDOM % $numfree )
            device="--device-ids ${freegpu[$pick]}"
            echo "get-gpu.sh: Picking device-id = ${freegpu[$pick]} from free GPUs: ${freegpu[@]}. `nvidia-smi`"
	else
            # No GPUs, default back to CPU
            device="--use-cpu"
            devicelog="get-gpu.sh: On $HOSTNAME Using CPU because no free GPUs. `nvidia-smi`"
	fi
    fi
else
    echo "Invalid device name specified; must be one of cpu or gpu"
    exit 1
fi

echo "$device"
echo "$devicelog"
