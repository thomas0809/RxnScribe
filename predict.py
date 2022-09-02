import os
import sys
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from reaction.pix2seq import build_pix2seq_model
from reaction.tokenizer import get_tokenizer
from reaction.dataset import ReactionDataset, get_collate_fn
import reaction.utils as utils
from main import ReactionExtractorPix2Seq, get_args


def predict_images(trainer, model, dataset, batch_size=8):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=1, collate_fn=get_collate_fn(dataset.pad_id))
    results = trainer.predict(model, dataloader)
    predictions = utils.merge_predictions(results)
    return predictions


def main():

    args = get_args()

    args.pix2seq = True
    tokenizer = get_tokenizer(args)

    dataset = ReactionDataset(args, tokenizer, image_files=args.images, split='test')

    model = ReactionExtractorPix2Seq.load_from_checkpoint(
        os.path.join(args.save_path, 'checkpoints/best.ckpt'), strict=False, args=args, tokenizer=tokenizer)

    trainer = pl.Trainer(
        gpus=1,
        default_root_dir='tmp',
        deterministic=True)

    print(predict_images(trainer, model, dataset))


if __name__ == "__main__":
    main()
