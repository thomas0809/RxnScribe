set -e
set -x

DATA_DIR=../../data/detect/splits
DETR_CKPT=./ckpts/detr-r50_no-class-head.pth
OUTPUT_DIR=./outputs/finetune200
FINETUNE_CKPT=${OUTPUT_DIR}/checkpoint.pth

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --dataset_file rxn_diagram \
    --coco_path ${DATA_DIR} \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 4 \
    --num_workers 2 \
    --output_dir ${OUTPUT_DIR} \
    --resume ${FINETUNE_CKPT} --eval

