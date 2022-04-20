# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from .data import ImageData


class CocoEvaluator(object):

    def __init__(self, coco_gt):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

    def evaluate(self, predictions):
        img_ids, results = self.prepare(predictions, 'bbox')
        if len(results) == 0:
            return np.zeros((12,))
        coco_dt = self.coco_gt.loadRes(results)
        cocoEval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        self.cocoEval = cocoEval
        return cocoEval.stats

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        img_ids = []
        coco_results = []
        for idx, prediction in enumerate(predictions):
            if len(prediction) == 0:
                continue

            image = self.coco_gt.dataset['images'][idx]
            img_ids.append(image['id'])
            width = image['width']
            height = image['height']

            coco_results.extend(
                [
                    {
                        "image_id": image['id'],
                        "category_id": pred['category_id'],
                        "bbox": convert_to_xywh(pred['bbox'], width, height),
                        "score": pred['score'],
                    }
                    for pred in prediction
                ]
            )
        return img_ids, coco_results


def convert_to_xywh(box, width, height):
    xmin, ymin, xmax, ymax = box
    return [xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height]


class ReactionEvaluator(object):

    def evaluate_image(self, gold_image, pred_image):
        data = ImageData(gold_image, pred_image)
        return data.evaluate()

    def evaluate(self, groundtruths, predictions):
        gold_hits, gold_total, pred_hits, pred_total = 0, 0, 0, 0
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image)
            gold_hits += sum(gh)
            gold_total += len(gh)
            pred_hits += sum(ph)
            pred_total += len(ph)
        precision = pred_hits / max(pred_total, 1)
        recall = gold_hits / gold_total
        f1 = precision * recall * 2 / max(precision + recall, 1e-6)
        return precision, recall, f1

    def evaluate_by_size(self, groundtruths, predictions):
        gold_groups = {}
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image)
            gtotal = len(gh)
            if gtotal not in gold_groups:
                gold_groups[gtotal] = {'hit': 0, 'reaction': 0, 'image': 0}
            gold_groups[gtotal]['hit'] += sum(gh)
            gold_groups[gtotal]['reaction'] += len(gh)
            gold_groups[gtotal]['image'] += 1
        for gtotal, stats in gold_groups.items():
            gold_groups[gtotal]['recall'] = stats['hit'] / stats['reaction']
        return gold_groups
