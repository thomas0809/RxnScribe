#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4

BATCH_SIZE=32
ACCUM_STEP=2

PIX2SEQ_CKPT=./ckpts/checkpoint_e299_ap370.pth
SAVE_PATH=output/pix2seq_reaction_full

set -x
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 python main.py \
    --data_path data/parse \
    --image_path data/parse/images \
    --save_path $SAVE_PATH \
    --train_file train.json \
    --valid_file dev.json \
    --test_file dev.json \
    --format reaction \
    --input_size 1333 \
    --pix2seq \
    --pix2seq_ckpt ${PIX2SEQ_CKPT} \
    --pred_eos \
    --augment --composite_augment \
    --lr 4e-4 \
    --epochs 600 --eval_per_epoch 30 \
    --warmup 0.02 \
    --label_smoothing 0. \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_train --do_valid --do_test \
    --gpus $NUM_GPUS_PER_NODE
