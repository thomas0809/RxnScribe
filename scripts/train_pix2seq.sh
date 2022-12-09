#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4

BATCH_SIZE=32
ACCUM_STEP=2

DATESTR=$(date +"%m-%d-%H-%M")
PIX2SEQ_CKPT=./ckpts/checkpoint_e299_ap370.pth
SAVE_PATH=output/pix2seq_reaction_nov_cv

set -x
for i in 0 1 2 3 4
do
save_path=${SAVE_PATH}/${i}
mkdir -p ${save_path}
NCCL_P2P_DISABLE=1 python main.py \
    --data_path data/parse/splits \
    --image_path data/parse/images \
    --save_path $save_path \
    --train_file train${i}.json \
    --valid_file dev${i}.json \
    --test_file test${i}.json \
    --formats reaction \
    --input_size 1333 \
    --pix2seq \
    --pix2seq_ckpt ${PIX2SEQ_CKPT} \
    --pred_eos \
    --augment --composite_augment \
    --lr 1e-4 \
    --epochs 300 --eval_per_epoch 10 \
    --warmup 0.02 \
    --label_smoothing 0. \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_train --do_valid --do_test \
    --gpus $NUM_GPUS_PER_NODE  #  2>&1  | tee $SAVE_PATH/log_${DATESTR}.txt
done