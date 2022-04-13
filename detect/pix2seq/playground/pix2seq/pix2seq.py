# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer
from util.box_ops import box_cxcywh_to_xyxy


class Pix2Seq(nn.Module):
    """ This is the Pix2Seq module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_bins=2000, max_objects=100, sep_xy=False):
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
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.sep_xy = sep_xy
        self.max_objects = max_objects
        self.input_proj = nn.Sequential(
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=(1, 1)),
            nn.GroupNorm(32, hidden_dim))
        self.backbone = backbone

    def forward(self, samples):
        """ 
            samples[0]:
            The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            samples[1]:
                targets
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all vocabulary.
                                Shape= [batch_size, num_sequence, num_vocal]
        """
        image_tensor, targets = samples[0], samples[1]
        if isinstance(image_tensor, (list, torch.Tensor)):
            image_tensor = nested_tensor_from_tensor_list(image_tensor)
        features, pos = self.backbone(image_tensor)

        src, mask = features[-1].decompose()
        assert mask is not None
        mask = torch.zeros_like(mask).bool()

        src = self.input_proj(src)
        if self.training:
            out = self.forward_train(src, targets, mask, pos[-1])
        else:
            out = self.forward_inference(src, mask, pos[-1])

        out = {'pred_seq_logits': out}
        return out

    def forward_train(self, src, targets, mask, pos):
        input_seq = self.build_input_seq(targets)
        similarity = self.transformer(src, input_seq, mask, pos)
        return similarity

    def forward_inference(self, src, mask, pos):
        out_seq = self.transformer(src, -1, mask, pos)
        return out_seq

    def build_input_seq(self, targets):
        device = targets[0]["labels"].device
        input_seq_list = []
        for b_i, target in enumerate(targets):
            box = target["boxes"]
            label = target["labels"]
            img_size = target["size"]
            h, w = img_size[0], img_size[1]
            scale_factor = torch.stack([w, h, w, h], dim=0)

            label_token = label.unsqueeze(1) + self.num_bins + 1
            scaled_box = box * scale_factor
            scaled_box = box_cxcywh_to_xyxy(scaled_box)
            box_tokens = discretize_box(scaled_box, self.num_bins, self.sep_xy)
            input_tokens = torch.cat([box_tokens, label_token], dim=1)

            num_objects = input_tokens.shape[0]
            num_noise = self.max_objects - num_objects

            random_class = torch.randint(0, self.num_classes, (num_noise, 1), device=device) + self.num_bins + 1
            random_box_x0y0 = torch.rand(num_noise, 2, device=device)
            random_box_wh = torch.rand(num_noise, 2, device=device)
            random_box_x1y1 = (random_box_x0y0 + random_box_wh).clamp(min=0, max=1)
            random_scaled_box = torch.cat([random_box_x0y0, random_box_x1y1], dim=1) * scale_factor
            random_box_tokens = discretize_box(random_scaled_box, self.num_bins, self.sep_xy)
            random_tokens = torch.cat([random_box_tokens, random_class], dim=1)

            if num_objects > 0:
                jitter_box_idx = torch.randint(0, num_objects, (num_noise,), device=device)
                jitter_class = label_token[jitter_box_idx]
                jitter_box = box[jitter_box_idx]
                jitter_box_wh = jitter_box[:, 2:].repeat(1, 2)
                jitter_box = box_cxcywh_to_xyxy(jitter_box)
                jitter_box = (torch.rand((num_noise, 4), device=device) - 0.5) * 2 * 0.2 * jitter_box_wh + jitter_box
                scaled_jitter_box = jitter_box.clamp(min=0, max=1.0) * scale_factor
                jitter_box_tokens = discretize_box(scaled_jitter_box, self.num_bins, self.sep_xy)
                jitter_tokens = torch.cat([jitter_box_tokens, jitter_class], dim=1)

                fake_tokens = torch.stack([random_tokens, jitter_tokens], dim=1)
                select_idx = torch.randint(0, 2, (num_noise,), device=device)
                fake_tokens = fake_tokens[range(num_noise), select_idx]
            else:
                fake_tokens = random_tokens

            input_seq = torch.cat([input_tokens, fake_tokens], dim=0).flatten()
            input_seq_list.append(input_seq)
        return torch.stack(input_seq_list, dim=0)


class SetCriterion(nn.Module):
    """
    This class computes the loss for Pix2Seq.
    """
    def __init__(self, num_classes, weight_dict, eos_coef, num_bins, num_vocal, max_objects, sep_xy):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_bins = num_bins
        self.sep_xy = sep_xy
        self.num_vocal = num_vocal
        self.max_objects = max_objects
        self.num_classes = num_classes
        empty_weight = torch.ones(self.num_vocal)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.weight_dict = weight_dict

    def build_target_seq(self, targets):
        device = targets[0]["labels"].device
        target_seq_list = []
        for target in targets:
            label = target["labels"]
            box = target["boxes"]
            img_size = target["size"]
            h, w = img_size[0], img_size[1]

            label = label.unsqueeze(1) + self.num_bins + 1
            box = box * torch.stack([w, h, w, h], dim=0)
            box = box_cxcywh_to_xyxy(box)
            box = discretize_box(box, self.num_bins, self.sep_xy)
            target_tokens = torch.cat([box, label], dim=1).flatten()

            end_token = torch.tensor([self.num_vocal - 2], dtype=torch.int64).to(device)

            num_noise = self.max_objects - len(label)
            fake_target_tokens = torch.zeros((num_noise, 5), dtype=torch.int64).to(device)
            fake_target_tokens[:, :3] = -100
            fake_target_tokens[:, 3] = self.num_vocal - 1  # noise class
            fake_target_tokens[:, 4] = self.num_vocal - 2  # eos
            fake_target_tokens = fake_target_tokens.flatten()

            target_seq = torch.cat([target_tokens, end_token, fake_target_tokens], dim=0)
            target_seq_list.append(target_seq)

        return torch.stack(target_seq_list, dim=0)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        target_seq = self.build_target_seq(targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_pos = (target_seq > -1).sum()
        num_pos = torch.as_tensor([num_pos], dtype=torch.float, device=target_seq.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pos)
        num_pos = torch.clamp(num_pos / get_world_size(), min=1).item()

        pred_seq_logits = outputs['pred_seq_logits']

        if isinstance(pred_seq_logits, list) and not self.training:
            num_pred_seq = [len(seq) for seq in pred_seq_logits]
            pred_seq_logits = torch.cat(pred_seq_logits, dim=0).reshape(-1, self.num_vocal)
            target_seq = torch.cat(
                [t_seq[:p_seq] for t_seq, p_seq in zip(target_seq, num_pred_seq)], dim=0)
        else:
            pred_seq_logits = pred_seq_logits.reshape(-1, self.num_vocal)
            target_seq = target_seq.flatten()

        loss_seq = F.cross_entropy(pred_seq_logits, target_seq, weight=self.empty_weight, reduction='sum') / num_pos

        # Compute all the requested losses
        losses = dict()
        losses["loss_seq"] = loss_seq
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_bins, num_classes, sep_xy):
        super().__init__()
        self.num_bins = num_bins
        self.sep_xy = sep_xy
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            targets:
            # target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
            #               For evaluation, this must be the original image size (before any data augmentation)
            #               For visualization, this should be the image size after data augment, but before padding
        """
        origin_img_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        input_img_sizes = torch.stack([t["size"] for t in targets], dim=0)
        out_seq_logits = outputs['pred_seq_logits']

        assert len(out_seq_logits) == len(origin_img_sizes)

        # and from relative [0, 1] to absolute [0, height] coordinates
        ori_img_h, ori_img_w = origin_img_sizes.unbind(1)
        inp_img_h, inp_img_w = input_img_sizes.unbind(1)
        scale_fct = torch.stack(
            [ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h], dim=1).unsqueeze(1)

        results = []
        for b_i, pred_seq_logits in enumerate(out_seq_logits):
            seq_len = pred_seq_logits.shape[0]
            if seq_len < 5:
                results.append(dict())
                continue
            pred_seq_logits = pred_seq_logits.softmax(dim=-1)
            num_objects = seq_len // 5
            pred_seq_logits = pred_seq_logits[:int(num_objects * 5)].reshape(num_objects, 5, -1)
            pred_boxes_logits = pred_seq_logits[:, :4, :self.num_bins + 1]
            pred_class_logits = pred_seq_logits[:, 4, self.num_bins + 1: self.num_bins + 1 + self.num_classes]
            scores_per_image, labels_per_image = torch.max(pred_class_logits, dim=1)
            boxes_per_image = continuous_box(pred_boxes_logits.argmax(dim=2), self.num_bins, self.sep_xy)
            boxes_per_image = boxes_per_image * scale_fct[b_i]
            result = dict()
            result['scores'] = scores_per_image
            result['labels'] = labels_per_image
            result['boxes'] = boxes_per_image
            results.append(result)

        return results


def discretize_box(x, num_bins=2000, sep_xy=False):
    if sep_xy:
        num_bins = num_bins // 2
        x = (x / 1333 * (num_bins - 1)).floor().long().clamp(min=0, max=num_bins-1)
        x[..., 1] += num_bins
        x[..., 3] += num_bins
    else:
        x = (x / 1333 * num_bins).floor().long().clamp(min=0, max=num_bins)
    return x


def continuous_box(x, num_bins=2000, sep_xy=False):
    if sep_xy:
        num_bins = num_bins // 2
        x[..., 1] -= num_bins
        x[..., 3] -= num_bins
        x = x * 1333 / (num_bins - 1)
    else:
        x = x * 1333 / num_bins
    return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20
    max_objects = 100
    if args.dataset_file == 'coco':
        num_classes = 91
    elif args.dataset_file == "coco_panoptic":
        num_classes = 250
    elif args.dataset_file == 'rxn_bbox':
        num_classes = 5
        max_objects = 100
    device = torch.device(args.device)

    backbone = build_backbone(args)

    num_bins = 2000
    # 0 - num_bin coordinate, num_bin+1 - num_bin+num_class class,
    # num_bin+num_class+1 end, num_bin+num_class+2 noise
    num_vocal = num_bins + 1 + 91 + 2

    transformer = build_transformer(args, num_vocal)

    model = Pix2Seq(
        backbone,
        transformer,
        num_classes=num_classes,
        num_bins=num_bins,
        max_objects=max_objects,
        sep_xy=args.sep_xy)

    weight_dict = {'loss_seq': 1}
    criterion = SetCriterion(
        num_classes,
        weight_dict,
        eos_coef=args.eos_coef,
        num_bins=num_bins,
        num_vocal=num_vocal,
        max_objects=max_objects,
        sep_xy=args.sep_xy)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_bins, num_classes, args.sep_xy)}

    return model, criterion, postprocessors
