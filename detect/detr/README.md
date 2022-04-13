## Finetune DETR for Molecular Structure Detection

To finetune a pretrained DETR checkpoint using molecular structure detection datasets, first ensure the dataset is organized in the COCO format:
```
cd ../../data/detect
sh prepare_coco.sh
```
Then run:
```
sh finetune.sh
```

To examine the performance of a finetuned model, refer to our notebook: `finetune_detr.ipynb`.

