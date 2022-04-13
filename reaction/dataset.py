import os
import cv2
import copy
import random
import json
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from reaction.tokenizer import FORMAT_INFO
import reaction.transforms as T

from pycocotools.coco import COCO
from PIL import Image


class ReactionDataset(Dataset):
    def __init__(self, args, data_file, tokenizer, split='train'):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        data_file = os.path.join(args.data_path, data_file)
        with open(data_file) as f:
            self.data = json.load(f)['images']
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                self.coco = COCO(data_file)
        self.name = os.path.basename(data_file).split('.')[0]
        self.image_path = args.image_path
        self.split = split
        self.formats = args.formats
        self.is_train = (split == 'train')
        self.transform = make_coco_transforms(split, args)

    def __len__(self):
        return len(self.data)

    def generate_sample(self, image, target):
        ref = {}
        image, target = self.transform(image, target)
        if self.is_train:
            if 'reaction' in self.formats:
                max_len = FORMAT_INFO['reaction']['max_len']
                label = self.tokenizer['reaction'].data_to_sequence(target)
                ref['reaction'] = torch.LongTensor(label[:max_len])
            if 'bbox' in self.formats:
                max_len = FORMAT_INFO['bbox']['max_len']
                label = self.tokenizer['bbox'].data_to_sequence(target)
                ref['bbox'] = torch.LongTensor(label[:max_len])
        return image, ref

    def __getitem__(self, idx):
        target = self.data[idx]
        path = os.path.join(self.image_path, target['file_name'])
        if not os.path.exists(path):
            print(path, "doesn't exists.", flush=True)
        image = Image.open(path).convert("RGB")
        image, target = self.prepare(image, target)
        if self.is_train and self.args.augment:
            image1, ref1 = self.generate_sample(image, target)
            image2, ref2 = self.generate_sample(image, target)
            return [[idx, image1, ref1], [idx, image2, ref2]]
        else:
            image, ref = self.generate_sample(image, target)
            return [[idx, image, ref]]

    def prepare(self, image, target):
        w, h = target['width'], target['height']

        image_id = target["id"]
        image_id = torch.tensor([image_id])

        anno = target["bboxes"]

        boxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = copy.deepcopy(target)
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["bbox"][2] * obj['bbox'][3] for obj in anno])
        target["area"] = area[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):
    normalize = T.Compose([
        T.Resize((1333, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train' and args.augment:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.LargeScaleJitter(output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0),
            T.RandomDistortion(0.5, 0.5, 0.5, 0.5),
            normalize])
    else:
        return T.Compose([
            T.LargeScaleJitter(output_size=1333, aug_scale_min=1.0, aug_scale_max=1.0),
            normalize])


def pad_images(imgs):
    # B, C, H, W
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1-i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1-i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)


def get_collate_fn(args, tokenizer):

    seq_formats = args.formats
    pad_id = tokenizer[seq_formats[0]].PAD_ID

    def rxn_collate(batch):
        ids = []
        imgs = []
        batch = [ex for seq in batch for ex in seq]
        seq_formats = list(batch[0][2].keys())
        refs = {key: [[], []] for key in seq_formats}
        for ex in batch:
            ids.append(ex[0])
            imgs.append(ex[1])
            ref = ex[2]
            for key in seq_formats:
                refs[key][0].append(ref[key])
                refs[key][1].append(torch.LongTensor([len(ref[key])]))
        # Sequence
        for key in seq_formats:
            refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=pad_id)
            refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
        return ids, pad_images(imgs), refs

    return rxn_collate