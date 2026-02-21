#!/bin/bash

# Run inference on all tasks in parallel (requires 4 GPUs)

CKPT_PATH=$1

if [ -z "$CKPT_PATH" ]; then
    echo "Usage: bash scripts/inference.sh /path/to/your/checkpoint"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m inference.vigia.single_cvmr --ckpt-path $CKPT_PATH --device_id 0 &
CUDA_VISIBLE_DEVICES=1 python -m inference.vigia.vsg --ckpt-path $CKPT_PATH --device_id 0 &
CUDA_VISIBLE_DEVICES=2 python -m inference.vigia.vqa --ckpt-path $CKPT_PATH --device_id 0 &
CUDA_VISIBLE_DEVICES=3 python -m inference.vigia.text_gen --ckpt-path $CKPT_PATH --device_id 0 &
wait
