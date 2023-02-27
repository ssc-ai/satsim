#!/bin/bash

# usage: source gpu_select.sh && gpu_app

OUTPUT="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"

#echo "${OUTPUT}"

FREE_GPU_LIST=()
INDEX=0
for mb in $OUTPUT
do
    if [[ $mb -gt 5000 ]] ; then
        FREE_GPU_LIST+=($INDEX)
    fi
    INDEX=$(($INDEX+1))
done

if [ -z "$FREE_GPU_LIST" ]
then
    echo "Warning: No GPUs available. Some tests may fail."
fi

printf -v FREE_GPU_LIST '%s,' "${FREE_GPU_LIST[@]}"
FREE_GPU_LIST=${FREE_GPU_LIST%,}

echo "Using GPU devices: ${FREE_GPU_LIST}"

export CUDA_VISIBLE_DEVICES="${FREE_GPU_LIST}"
