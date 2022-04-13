#!/bin/bash

set -e
set -x

DATA_DIR=../../data/detect/splits
PIX2SEQ_CKPT=./ckpts/checkpoint_e299_ap370.pth
OUTPUT_DIR=outputs/finetune
FINETUNE_CKPT=${OUTPUT_DIR}/checkpoint_best.pth

NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --pix2seq_lr --large_scale_jitter \
    --dataset_file rxn_bbox \
    --coco_path ${DATA_DIR} \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 4 \
    --num_workers 2 \
    --output_dir ${OUTPUT_DIR} \
    --resume ${FINETUNE_CKPT} --eval --pred_eos
