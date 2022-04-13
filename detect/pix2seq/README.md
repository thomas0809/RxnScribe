# Pretrained-Pix2Seq
We provide the pre-trained model of Pix2Seq.

## Installation

Install PyTorch 1.5+ and torchvision 0.6+ (recommend torch1.8.1 torchvision 0.8.0)

Install pycocotools (for evaluation on COCO):

```
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

That's it, should be good to train and evaluate detection models.

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Scripts

Training
```
bash train.sh
```

Finetune
```
bash finetune.sh
```

Evaluation
```
cp -r /Mounts/rbg-storage1/users/yujieq/chem-diagram-parsing/detect/pix2seq/outputs/finetune outputs/finetune
bash eval.sh
```
