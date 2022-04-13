#!/bin/bash

set -e
set -x

DATA_DIR=../../data/detect/splits
PIX2SEQ_CKPT=./ckpts/checkpoint_e299_ap370.pth
OUTPUT_DIR=outputs/finetune

NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --pix2seq_lr --large_scale_jitter --model pix2seq --rand_target \
    --dataset_file rxn_bbox \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 4 \
    --coco_path ${DATA_DIR} \
    --resume ${PIX2SEQ_CKPT} \
    --output_dir ${OUTPUT_DIR} --pred_eos

