set -e
set -x

# download pretrained DETR checkpoints and remove classification heads
#python ./download_detr_checkpoint.py

DATA_DIR=../../data/detect/splits
DETR_CKPT=./ckpts/detr-r50_no-class-head.pth

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --dataset_file rxn_diagram \
    --coco_path ${DATA_DIR} \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 4 \
    --num_workers 2 \
    --output_dir ./outputs/finetune200 \
    --resume ${DETR_CKPT}

