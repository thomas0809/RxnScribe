#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4

BATCH_SIZE=32
ACCUM_STEP=2

PIX2SEQ_CKPT=./output/1e-4_300epoch_coco/checkpoints/best.ckpt
SAVE_PATH=output/1e-4_detect_hf_pretrained0.28_2000epoch_both_augment_mol

set -x
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 python main.py \
    --data_path data/detect/splits/annotations \
    --image_path data/detect/images \
    --save_path $SAVE_PATH \
    --train_file mol_detect_train.json\
    --valid_file mol_detect_val.json\
    --test_file mol_detect_test.json\
    --format bbox \
    --input_size 1333 \
    --pix2seq \
    --pix2seq_ckpt ${PIX2SEQ_CKPT} \
    --pred_eos \
    --augment \
    --split_heuristic \
    --composite_augment \
    --use_hf_transformer \
    --lr 1e-4 \
    --epochs 1000 --eval_per_epoch 100 \
    --warmup 0.02 \
    --label_smoothing 0. \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_val --do_test \
    --gpus $NUM_GPUS_PER_NODE
