import os
import cv2
import time
import random
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

from reaction.augment import SafeRotate, CropWhite, NormalizedGridDistortion, PadWhite, SaltAndPepperNoise
from reaction.tokenizer import PAD_ID, FORMAT_INFO

cv2.setNumThreads(1)


def get_transforms(input_size, augment=True, debug=False):
    trans_list = []
    if augment:
        trans_list.append(SafeRotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    trans_list.append(CropWhite(pad=5))
    if augment:
        trans_list += [
            # NormalizedGridDistortion(num_steps=10, distort_limit=0.3),
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            PadWhite(pad_ratio=0.4, p=0.2),
            A.Downscale(scale_min=0.15, scale_max=0.3, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5)
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list)


class ReactionDataset(Dataset):
    def __init__(self, args, data_file, tokenizer, split='train'):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        with open(os.path.join(args.data_path, data_file)) as f:
            self.image_data = json.load(f)['images']
        self.name = os.path.basename(data_file).split('.')[0]
        self.image_path = args.image_path
        self.split = split
        self.formats = args.formats
        self.labelled = (split == 'train')
        self.transform = get_transforms(args.input_size, augment=(self.labelled and args.augment))
        # if args.debug:
        #     self.image_data = self.image_data[:16]
        
    def __len__(self):
        return len(self.image_data)

    def image_transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
        return image

    def __getitem__(self, idx):
        ref = {}
        data = self.image_data[idx]
        image = cv2.imread(os.path.join(self.image_path, data['file_name']))
        if image is None:
            print(data['file_name'], "doesn't exists.", flush=True)
        image = self.image_transform(image)
        if self.labelled:
            if 'reaction' in self.formats:
                max_len = FORMAT_INFO['reaction']['max_len']
                label = self.tokenizer['reaction'].data_to_sequence(data)
                ref['reaction'] = torch.LongTensor(label[:max_len])
            if 'bbox' in self.formats:
                max_len = FORMAT_INFO['bbox']['max_len']
                label = self.tokenizer['bbox'].data_to_sequence(data)
                ref['bbox'] = torch.LongTensor(label[:max_len])
        return idx, image, ref


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


def rxn_collate(batch):
    ids = []
    imgs = []
    batch = [ex for ex in batch if ex[1] is not None]
    formats = list(batch[0][2].keys())
    seq_formats = formats
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
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    return ids, pad_images(imgs), refs
