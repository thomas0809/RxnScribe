#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4

BATCH_SIZE=32
ACCUM_STEP=2

PIX2SEQ_CKPT=./output/1e-4_300epoch_coco/checkpoints/best.ckpt
SAVE_PATH=output/pix2seq_reaction_cv_ep1000_1e-4

set -x
for i in 0 1 2 3 4
do
save_path=${SAVE_PATH}/${i}
mkdir -p ${save_path}
NCCL_P2P_DISABLE=1 python main.py \
    --data_path /Mounts/rbg-storage1/users/yujieq/RxnScribe/data/parse/splits \
    --image_path /Mounts/rbg-storage1/users/yujieq/RxnScribe/data/parse/images \
    --save_path $save_path \
    --train_file train${i}.json \
    --valid_file dev${i}.json \
    --test_file test${i}.json \
    --format reaction \
    --input_size 1333 \
    --use_hf_transformer \
    --pix2seq \
    --pix2seq_ckpt ${PIX2SEQ_CKPT} \
    --pred_eos \
    --augment --composite_augment \
    --lr 1e-4 \
    --epochs 1000 --eval_per_epoch 50 \
    --warmup 0.02 \
    --label_smoothing 0. \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_train --do_valid --do_test \
    --gpus $NUM_GPUS_PER_NODE
done
