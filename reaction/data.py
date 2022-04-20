import os
import cv2
import numpy as np
import matplotlib.patches as patches


class BBox(object):

    def __init__(self, bbox, width, height, xyxy=False, normalized=False):
        """
        :param bbox: {'catrgory_id', 'bbox'}
        :param width: width of the image
        :param height: height of the image
        :param xyxy:
        :param normalized:
        """
        self.width = width
        self.height = height
        self.category_id = bbox['category_id']
        if xyxy:
            x1, y1, x2, y2 = bbox['bbox']
        else:
            x1, y1, w, h = bbox['bbox']
            x2, y2 = x1 + w, y1 + h
        if not normalized:
            x1, y1, x2, y2 = x1 / width, y1 / height, x2 /width, y2 / height
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def unnormalize(self):
        return self.x1 * self.width, self.y1 * self.height, self.x2 * self.width, self.y2 * self.height

    COLOR = {1: 'r', 2: 'g', 3: 'b', 4: 'y'}

    def draw(self, ax, color=None):
        x1, y1, x2, y2 = self.unnormalize()
        if color is None:
            color = self.COLOR[self.category_id]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
        text = f'{self.category_id}'
        ax.text(x1, y1, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
        ax.add_patch(rect)
        return


class Reaction(object):

    def __init__(self, reaction=None, bboxes=None):
        '''
        if image_data is None, create from prediction
        if image_data is not None, create from groundtruth
        '''
        self.reactants = []
        self.conditions = []
        self.products = []
        self.bboxes = []
        if reaction is not None:
            for x in reaction['reactants']:
                bbox = bboxes[x] if type(x) is int else x
                self.bboxes.append(bbox)
                self.reactants.append(len(self.bboxes) - 1)
            for x in reaction['conditions']:
                bbox = bboxes[x] if type(x) is int else x
                self.bboxes.append(bbox)
                self.conditions.append(len(self.bboxes) - 1)
            for x in reaction['products']:
                bbox = bboxes[x] if type(x) is int else x
                self.bboxes.append(bbox)
                self.products.append(len(self.bboxes) - 1)

    def __eq__(self, other):
        # Exact matching of two reactions
        if len(self.bboxes) != len(other.bboxes):
            return False
        match1, match2, scores = get_bboxes_match(self.bboxes, other.bboxes)
        if any([m == -1 for m in match1]):
            return False
        m_reactants = [match1[i] for i in self.reactants]
        m_conditions = [match1[i] for i in self.conditions]
        m_products = [match1[i] for i in self.products]
        return sorted(m_reactants) == sorted(other.reactants) and sorted(m_conditions) == sorted(other.conditions) \
            and sorted(m_products) == sorted(other.products)

    def draw(self, ax):
        for i in self.reactants:
            self.bboxes[i].draw(ax, color='r')
        for i in self.conditions:
            self.bboxes[i].draw(ax, color='g')
        for i in self.products:
            self.bboxes[i].draw(ax, color='b')
        return


class ImageData(object):

    def __init__(self, image_data, predictions=None):
        self.file_name = image_data['file_name']
        self.width = image_data['width']
        self.height = image_data['height']
        self.gold_bboxes = [BBox(bbox, self.width, self.height, xyxy=False, normalized=False)
                       for bbox in image_data['bboxes']]
        self.reaction = ('reactions' in image_data)
        if self.reaction:
            self.gold_reactions = [Reaction(reaction, self.gold_bboxes) for reaction in image_data['reactions']]
        if predictions is not None:
            if self.reaction:
                self.pred_reactions = [
                    Reaction({
                        key: [BBox(bbox, self.width, self.height, xyxy=True, normalized=True) for bbox in reaction[key]]
                        for key in ['reactants', 'conditions', 'products']
                    })
                    for reaction in predictions
                ]
            else:
                self.pred_bboxes = [
                    BBox(bbox, self.width, self.height, xyxy=True, normalized=True) for bbox in predictions
                ]

    def evaluate(self):
        gold_total = len(self.gold_reactions)
        gold_hit = [False] * gold_total
        pred_total = len(self.pred_reactions)
        pred_hit = [False] * pred_total
        for i, ri in enumerate(self.gold_reactions):
            for j, rj in enumerate(self.pred_reactions):
                if gold_hit[i] and pred_hit[j]:
                    continue
                if ri == rj:
                    gold_hit[i] = True
                    pred_hit[j] = True
        return gold_hit, pred_hit

    def draw_gold(self, ax, image=None):
        if image is not None:
            ax.imshow(image)
        if self.reaction:
            for r in self.gold_reactions:
                r.draw(ax)
        else:
            for b in self.gold_bboxes:
                b.draw(ax)
        return

    def draw_prediction(self, ax, image=None):
        if image is not None:
            ax.imshow(image)
        if self.reaction:
            for r in self.pred_reactions:
                r.draw(ax)
        else:
            for b in self.pred_bboxes:
                b.draw(ax)
        return


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    bb1 = {'x1': bb1.x1, 'y1': bb1.y1, 'x2': bb1.x2, 'y2': bb1.y2}
    bb2 = {'x1': bb2.x1, 'y1': bb2.y1, 'x2': bb2.x2, 'y2': bb2.y2}

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_bboxes_match(bboxes1, bboxes2, iou_thres=0.5, match_category=False):
    scores = np.zeros((len(bboxes1), len(bboxes2)))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            if match_category and bbox1.category_id != bbox2.category_id:
                scores[i, j] = 0
            else:
                scores[i, j] = get_iou(bbox1, bbox2)
    match1 = scores.argmax(axis=1)
    for i in range(len(match1)):
        if scores[i, match1[i]] < iou_thres:
            match1[i] = -1
    match2 = scores.argmax(axis=0)
    for j in range(len(match2)):
        if scores[match2[j], j] < iou_thres:
            match2[j] = -1
    return match1, match2, scores
