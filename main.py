import os
import sys
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info
from transformers import get_scheduler

from reaction.model import Encoder, Decoder
from reaction.pix2seq import build_pix2seq_model
from reaction.loss import Criterion
from reaction.tokenizer import get_tokenizer
from reaction.dataset import ReactionDataset, get_collate_fn
from reaction.utils import print_rank_0
from reaction.evaluate import CocoEvaluator, ReactionEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    # Model
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--decoder', type=str, default='lstm')
    parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--lstm_dropout', type=float, default=0.5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--enc_pos_emb', action='store_true')
    group = parser.add_argument_group("lstm_options")
    group.add_argument('--decoder_dim', type=int, default=512)
    group.add_argument('--decoder_layer', type=int, default=1)
    group.add_argument('--attention_dim', type=int, default=256)
    group = parser.add_argument_group("transformer_options")
    group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
    group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
    group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
    group.add_argument("--dec_num_queries", type=int, default=128)
    group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
    group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
    group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
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
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--formats', type=str, default='reaction')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--coord_bins', type=int, default=100)
    parser.add_argument('--sep_xy', action='store_true')
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_encoder_only', action='store_true')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--eval_per_epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init_scheduler', action='store_true')
    parser.add_argument('--trunc_train', type=int, default=None)
    parser.add_argument('--trunc_valid', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--save_image', action='store_true')
    # Inference
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--n_best', type=int, default=1)
    args = parser.parse_args()

    args.formats = args.formats.split(',')

    return args


class ReactionExtractor(LightningModule):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.encoder = Encoder(args, pretrained=(not args.no_pretrained))
        args.encoder_dim = self.encoder.n_features
        self.decoder = Decoder(args, tokenizer)
        self.criterion = Criterion(args, tokenizer)

    def training_step(self, batch, batch_idx):
        indices, images, refs = batch
        features, hiddens = self.encoder(images, refs)
        results = self.decoder(features, hiddens, refs)
        losses = self.criterion(results, refs)
        loss = sum(losses.values())
        self.log('train/loss', loss)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        indices, images, refs = batch
        features, hiddens = self.encoder(images, refs)
        batch_preds, batch_beam_preds = self.decoder.decode(
            features, hiddens, refs,
            beam_size=self.args.beam_size, n_best=self.args.n_best)
        return indices, batch_preds

    def validation_epoch_end(self, outputs, phase='val'):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, outputs)
            gathered_outputs = sum(gathered_outputs, [])
        else:
            gathered_outputs = outputs

        formats = self.args.formats
        predictions = {format_: {} for format_ in formats}
        for format_ in formats:
            for indices, batch_preds in gathered_outputs:
                for idx, preds in zip(indices, batch_preds[format_]):
                    predictions[format_][idx] = preds
            predictions[format_] = [predictions[format_][i] for i in range(len(predictions[format_]))]

        name = self.eval_dataset.name
        if 'bbox' in formats:
            coco_evaluator = CocoEvaluator(self.eval_dataset.coco)
            stats = coco_evaluator.evaluate(predictions['bbox'])
            self.log(f'{phase}/score', stats[0], prog_bar=True, rank_zero_only=True)
        if 'reaction' in formats:
            evaluator = ReactionEvaluator()
            precision, recall, f1 = evaluator.evaluate(self.eval_dataset.data, predictions['reaction'])
            self.print(f'Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}')
            self.log(f'{phase}/score', f1, prog_bar=True, rank_zero_only=True)

        if self.trainer.is_global_zero:
            with open(os.path.join(self.trainer.default_root_dir, f'prediction_{name}.json'), 'w') as f:
                json.dump(predictions, f)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, phase='test')

    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        print_rank_0(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        # parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


class ReactionExtractorPix2Seq(ReactionExtractor):

    def __init__(self, args, tokenizer):
        super(ReactionExtractor, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = build_pix2seq_model(args)
        self.criterion = Criterion(args, tokenizer)

    def training_step(self, batch, batch_idx):
        indices, images, refs = batch
        results = {format_: self.model(images, refs[format_]) for format_ in self.args.formats}
        losses = self.criterion(results, refs)
        loss = sum(losses.values())
        self.log('train/loss', loss)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        indices, images, refs = batch
        batch_preds = {}
        for format_ in self.args.formats:
            batch_preds[format_] = []
            pred_logits = self.model(images)
            for i, logits in enumerate(pred_logits):
                probs = F.softmax(logits, dim=-1)
                scores, preds = probs.max(dim=-1)
                batch_preds[format_].append(
                    self.tokenizer[format_].sequence_to_data(preds.tolist(), scores.tolist()))
        return indices, batch_preds


class ReactionDataModule(LightningDataModule):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.collate_fn = get_collate_fn(args, tokenizer)

    def prepare_data(self):
        if self.args.do_train:
            self.train_dataset = ReactionDataset(self.args, self.args.train_file, self.tokenizer, split='train')
            print(f'Train dataset: {len(self.train_dataset)}')
        if self.args.do_train or self.args.do_valid:
            self.val_dataset = ReactionDataset(self.args, self.args.valid_file, self.tokenizer, split='valid')
            print(f'Valid dataset: {len(self.val_dataset)}')
        if self.args.do_test:
            self.test_dataset = ReactionDataset(self.args, self.args.test_file, self.tokenizer, split='test')
            print(f'Test dataset: {len(self.test_dataset)}')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, drop_last=False,
            collate_fn=self.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)


def main():

    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    if args.debug:
        args.save_path = "output/debug"

    tokenizer = get_tokenizer(args)

    MODEL = ReactionExtractorPix2Seq if args.pix2seq else ReactionExtractor
    if args.do_train:
        model = MODEL(args, tokenizer)
    else:
        model = MODEL.load_from_checkpoint(
            os.path.join(args.save_path, 'checkpoints/best.ckpt'), args=args, tokenizer=tokenizer)

    dm = ReactionDataModule(args, tokenizer)
    dm.prepare_data()

    checkpoint = ModelCheckpoint(monitor='val/score', mode='max', save_top_k=1, filename='best', save_last=True)
    # checkpoint = ModelCheckpoint(monitor=None, save_top_k=0, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = pl.loggers.TensorBoardLogger(args.save_path, name='', version='')

    trainer = pl.Trainer(
        strategy="ddp",
        gpus=args.gpus,
        logger=logger,
        default_root_dir=args.save_path,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        check_val_every_n_epoch=args.eval_per_epoch,
        log_every_n_steps=20,
        deterministic=True)

    if args.do_train:
        trainer.num_training_steps = len(dm.train_dataset) // (args.batch_size * args.gpus) * args.epochs
        model.eval_dataset = dm.val_dataset
        trainer.fit(model, datamodule=dm)
        model = MODEL.load_from_checkpoint(checkpoint.best_model_path, args=args, tokenizer=tokenizer)

    if args.do_valid:
        model.eval_dataset = dm.val_dataset
        trainer.validate(model, datamodule=dm)

    if args.do_test:
        model.eval_dataset = dm.test_dataset
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()