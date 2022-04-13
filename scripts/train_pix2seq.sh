#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4

BATCH_SIZE=16
ACCUM_STEP=1

DATESTR=$(date +"%m-%d-%H-%M")
PIX2SEQ_CKPT=./ckpts/checkpoint_e299_ap370.pth
SAVE_PATH=output/pix2seq_reaction_aug
mkdir -p ${SAVE_PATH}

set -x
#NCCL_P2P_DISABLE=1
#PL_TORCH_DISTRIBUTED_BACKEND=gloo
NCCL_P2P_DISABLE=1 python main.py \
    --data_path data/parse \
    --image_path data/parse/images \
    --save_path $SAVE_PATH \
    --train_file train.json \
    --valid_file dev.json \
    --test_file dev.json \
    --formats reaction \
    --input_size 1333 \
    --pix2seq \
    --pix2seq_ckpt ${PIX2SEQ_CKPT} \
    --pred_eos \
    --augment \
    --lr 1e-4 \
    --epochs 200 \
    --warmup 0.05 \
    --label_smoothing 0.1 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --do_test \
    --gpus $NUM_GPUS_PER_NODE  #  2>&1  | tee $SAVE_PATH/log_${DATESTR}.txt
