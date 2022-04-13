# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .pix2seq import build


def build_pix2seq_model(args):
    return build(args)
