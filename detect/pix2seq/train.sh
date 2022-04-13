#!/bin/bash

set -e
set -x

OUTPUT_DIR=outputs/coco

NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --coco_path ./coco --model pix2seq --pix2seq_lr --large_scale_jitter --rand_target \
    --output_dir ${OUTPUT_DIR} --sep_xy --pred_eos \
    --epochs 300 --lr 1e-3 --batch_size 4 --resume ${OUTPUT_DIR}/checkpoint.pth
