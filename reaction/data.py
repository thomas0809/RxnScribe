import os
import cv2
import numpy as np
import matplotlib.patches as patches


class BBox(object):

    def __init__(self, bbox, image_data=None, xyxy=False, normalized=False):
        """
        :param bbox: {'catrgory_id', 'bbox'}
        :param input_image: ImageData object
        :param xyxy:
        :param normalized:
        """
        self.data = bbox
        self.image_data = image_data
        if image_data is not None:
            self.width = image_data.width
            self.height = image_data.height
        self.category_id = bbox['category_id']
        if xyxy:
            x1, y1, x2, y2 = bbox['bbox']
        else:
            x1, y1, w, h = bbox['bbox']
            x2, y2 = x1 + w, y1 + h
        if not normalized:
            x1, y1, x2, y2 = x1 / self.width, y1 / self.height, x2 / self.width, y2 / self.height
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def is_mol(self):
        return self.category_id == 1

    def unnormalize(self):
        return self.x1 * self.width, self.y1 * self.height, self.x2 * self.width, self.y2 * self.height

    def image(self):
        x1, y1, x2, y2 = self.unnormalize()
        x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), self.width), min(int(y2), self.height)
        return self.image_data.image[y1:y2, x1:x2]

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

    def set_smiles(self, smiles):
        self.data['smiles'] = smiles

    def to_json(self):
        return self.data


class Reaction(object):

    def __init__(self, reaction=None, bboxes=None, image_data=None):
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
                bbox = bboxes[x] if type(x) is int else BBox(x, image_data, xyxy=True, normalized=True)
                self.bboxes.append(bbox)
                self.reactants.append(len(self.bboxes) - 1)
            for x in reaction['conditions']:
                bbox = bboxes[x] if type(x) is int else BBox(x, image_data, xyxy=True, normalized=True)
                self.bboxes.append(bbox)
                self.conditions.append(len(self.bboxes) - 1)
            for x in reaction['products']:
                bbox = bboxes[x] if type(x) is int else BBox(x, image_data, xyxy=True, normalized=True)
                self.bboxes.append(bbox)
                self.products.append(len(self.bboxes) - 1)

    def to_json(self):
        return {
            'reactants': [self.bboxes[i].to_json() for i in self.reactants],
            'conditions': [self.bboxes[i].to_json() for i in self.conditions],
            'products': [self.bboxes[i].to_json() for i in self.products]
        }

    def schema(self, mol_only=False):
        # Return reactants, conditions, and products. If mol_only is True, only include bboxes that are mol structures.
        if mol_only:
            return [
                [idx for idx in indices if self.bboxes[idx].is_mol]
                for indices in [self.reactants, self.conditions, self.products]
            ]
        else:
            return self.reactants, self.conditions, self.products

    def compare(self, other, mol_only=False, merge_condition=False, debug=False):
        reactants1, conditions1, products1 = self.schema(mol_only)
        reactants2, conditions2, products2 = other.schema(mol_only)
        if debug:
            print(reactants1, conditions1, products1, ';', reactants2, conditions2, products2)
        if len(reactants1) + len(conditions1) + len(products1) == 0:
            # schema is empty, always return False
            return False
        if len(reactants1) + len(conditions1) + len(products1) != len(reactants2) + len(conditions2) + len(products2):
            return False
        # Match use original index
        match1, match2, scores = get_bboxes_match(self.bboxes, other.bboxes)
        m_reactants, m_conditions, m_products = [[match1[i] for i in x] for x in [reactants1, conditions1, products1]]
        if any([m == -1 for m in m_reactants + m_conditions + m_products]):
            return False
        if debug:
            print(m_reactants, m_conditions, m_products, ';', reactants2, conditions2, products2)
        if merge_condition:
            return sorted(m_reactants + m_conditions) == sorted(reactants2 + conditions2) \
                   and sorted(m_products) == sorted(products2)
        else:
            return sorted(m_reactants) == sorted(reactants2) and sorted(m_conditions) == sorted(conditions2) \
                   and sorted(m_products) == sorted(products2)

    def __eq__(self, other):
        # Exact matching of two reactions
        return self.compare(other)

    def draw(self, ax):
        for i in self.reactants:
            self.bboxes[i].draw(ax, color='r')
        for i in self.conditions:
            self.bboxes[i].draw(ax, color='g')
        for i in self.products:
            self.bboxes[i].draw(ax, color='b')
        return


class ImageData(object):

    def __init__(self, image_data, predictions=None, load_image=False, image_path=None):
        self.file_name = image_data['file_name']
        self.width = image_data['width']
        self.height = image_data['height']
        if load_image:
            assert image_path is not None
            self.image = cv2.imread(os.path.join(image_path, self.file_name))
        if 'bboxes' in image_data:
            self.gold_bboxes = [BBox(bbox, self, xyxy=False, normalized=False) for bbox in image_data['bboxes']]
        else:
            self.gold_bboxes = []
        self.reaction = ('reactions' in image_data)
        if self.reaction:
            self.gold_reactions = [Reaction(reaction, self.gold_bboxes) for reaction in image_data['reactions']]
        if predictions is not None:
            if self.reaction:
                self.pred_reactions = [Reaction(reaction, image_data=self) for reaction in predictions]
            else:
                self.pred_bboxes = [BBox(bbox, self, xyxy=True, normalized=True) for bbox in predictions]

    def evaluate(self, mol_only=False, merge_condition=False, debug=False):
        gold_total = len(self.gold_reactions)
        gold_hit = [False] * gold_total
        pred_total = len(self.pred_reactions)
        pred_hit = [False] * pred_total
        for i, ri in enumerate(self.gold_reactions):
            for j, rj in enumerate(self.pred_reactions):
                if gold_hit[i] and pred_hit[j]:
                    continue
                if ri.compare(rj, mol_only, merge_condition, debug):
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
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
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
    """Find the match between two sets of bboxes. Each bbox is matched with a bbox with maximum overlap
    (at least above iou_thres). -1 if a bbox does not have a match."""
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
