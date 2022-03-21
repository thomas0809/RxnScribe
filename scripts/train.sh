#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=2

BATCH_SIZE=16
ACCUM_STEP=1

DATESTR=$(date +"%m-%d-%H-%M")
SAVE_PATH=output/swin_base
mkdir -p ${SAVE_PATH}

set -x
#NCCL_P2P_DISABLE=1
#PL_TORCH_DISTRIBUTED_BACKEND=gloo
NCCL_P2P_DISABLE=1 python main.py \
    --data_path data/diagram \
    --image_path data/diagram/images \
    --save_path $SAVE_PATH \
    --train_file train.json \
    --valid_file train_50.json \
    --test_file dev.json \
    --formats reaction \
    --input_size 512 \
    --encoder swin_base \
    --decoder transformer \
    --lr 4e-4 \
    --epochs 2000 \
    --warmup 0.05 \
    --label_smoothing 0.1 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --do_test \
    --gpus $NUM_GPUS_PER_NODE  #  2>&1  | tee $SAVE_PATH/log_${DATESTR}.txt
