# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .misc import nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer


class Pix2Seq(nn.Module):
    """ This is the Pix2Seq module that performs object detection """
    def __init__(self, backbone, transformer):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_bins: number of bins for each side of the input image
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.input_proj = nn.Sequential(
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=(1, 1)),
            nn.GroupNorm(32, hidden_dim))
        self.backbone = backbone

    def forward(self, image_tensor, targets=None, max_len=500):
        """ 
        image_tensor:
        The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all vocabulary.
                            Shape= [batch_size, num_sequence, num_vocal]
        """
        if isinstance(image_tensor, (list, torch.Tensor)):
            image_tensor = nested_tensor_from_tensor_list(image_tensor)
        features, pos = self.backbone(image_tensor)

        src, mask = features[-1].decompose()
        assert mask is not None
        mask = torch.zeros_like(mask).bool()

        src = self.input_proj(src)
        if targets is not None:
            input_seq, input_len = targets
            output_logits = self.transformer(src, input_seq[:, 1:], mask, pos[-1])
            return output_logits[:, :-1]
        else:
            output_seqs, output_scores = self.transformer(src, None, mask, pos[-1], max_len=max_len)
            return output_seqs, output_scores


def build_pix2seq_model(args, tokenizer):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    backbone = build_backbone(args)
    transformer = build_transformer(args, tokenizer)

    model = Pix2Seq(backbone, transformer)

    if args.pix2seq_ckpt is not None:
        checkpoint = torch.load(args.pix2seq_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    return model
