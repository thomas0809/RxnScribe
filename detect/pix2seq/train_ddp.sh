#!/bin/bash

set -e
set -x

NUM_NODES=2
NUM_GPUS=8
JOB_ID=12345
HOST_NODE_ADDR=rosetta4.csail.mit.edu:10000

OUTPUT_DIR=outputs/coco

NCCL_P2P_DISABLE=1 torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS --max_restarts=3 \
    --rdzv_id=$JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR main.py \
    --coco_path ./coco --model pix2seq --pix2seq_lr --large_scale_jitter --rand_target \
    --output_dir ${OUTPUT_DIR} --sep_xy --pred_eos \
    --epochs 300 --lr 1e-3 --batch_size 4 --resume ${OUTPUT_DIR}/checkpoint.pth
