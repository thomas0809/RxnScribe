#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4

BATCH_SIZE=32
ACCUM_STEP=2

PIX2SEQ_CKPT=./ckpts/checkpoint_e299_ap370.pth
SAVE_PATH=output/5e-5_300epoch_coco

set -x
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 python main.py \
    --data_path data/detect/splits/annotations \
    --image_path data/detect/images \
    --save_path $SAVE_PATH \
    --train_file detect_train.json\
    --valid_file detect_val.json\
    --test_file detect_test.json\
    --format bbox \
    --input_size 1333 \
    --pix2seq \
    --pix2seq_ckpt ${PIX2SEQ_CKPT} \
    --pred_eos \
    --lr 4e-4 \
    --epochs 9 --eval_per_epoch 3 \
    --warmup 0.02 \
    --label_smoothing 0. \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_train --do_test \
    --gpus $NUM_GPUS_PER_NODE
