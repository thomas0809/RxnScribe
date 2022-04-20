import os
import sys
import time
import json
import random
import argparse
from PIL import Image
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from reaction.pix2seq import build_pix2seq_model
from reaction.tokenizer import BboxTokenizer
from reaction.dataset import make_transforms
from reaction.evaluate import CocoEvaluator, ReactionEvaluator


def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    # Pix2Seq
    parser.add_argument('--pix2seq', action='store_true', help="specify the model from playground")
    parser.add_argument('--pix2seq_ckpt', type=str, default=None)
    parser.add_argument('--large_scale_jitter', action='store_true', help='large scale jitter')
    parser.add_argument('--rand_target', action='store_true', help="randomly permute the sequence of input targets")
    parser.add_argument('--pred_eos', action='store_true', help='use eos token instead of predicting 100 objects')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    # Data
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--formats', type=str, default='bbox')
    args = parser.parse_args([]) if notebook else parser.parse_args()

    args.formats = args.formats.split(',')

    return args


class ReactionExtractorPix2Seq(LightningModule):

    def __init__(self, args, tokenizer):
        super(ReactionExtractorPix2Seq, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = build_pix2seq_model(args)
        self.transform = make_transforms('test', augment=False, debug=False)

    def predict(self, images):
        # images: a list of PIL images
        data = [self.transform(image) for image in images]
        images = torch.stack([image for image, target in data]).cuda()
        targets = [target for image, target in data]
        pred_logits = self.model(images)
        predictions = []
        for i, logits in enumerate(pred_logits):
            probs = F.softmax(logits, dim=-1)
            scores, preds = probs.max(dim=-1)
            predictions.append(
                self.tokenizer.sequence_to_data(preds.tolist(), scores.tolist(), scale=targets[i]['scale']))
        return predictions


def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    return model.predict([image])[0]


def main():

    args = get_args()
    tokenizer = BboxTokenizer(input_size=2000, sep_xy=False, pix2seq=True, rand_target=False)

    model = ReactionExtractorPix2Seq.load_from_checkpoint(
        os.path.join(args.save_path, 'checkpoints/best.ckpt'), strict=False, args=args, tokenizer=tokenizer)
    model.eval()
    model.cuda()

    print(predict_image(model, args.image_path))


if __name__ == "__main__":
    main()
