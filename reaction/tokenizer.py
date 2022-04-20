import json
import random
import numpy as np


FORMAT_INFO = {
    "reaction": {"max_len": 256},
    "bbox": {"max_len": 400}
}

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
MASK = '<mask>'

Rxn = '[Rxn]'  # Reaction
Rct = '[Rct]'  # Reactant
Prd = '[Prd]'  # Product
Cnd = '[Cnd]'  # Condition
Idt = '[Idt]'  # Identifier
Mol = '[Mol]'  # Molecule
Txt = '[Txt]'  # Text
Sup = '[Sup]'  # Supplement
Noise = '[Nos]'


class ReactionTokenizer(object):

    def __init__(self, input_size=100, sep_xy=True, pix2seq=False):
        self.stoi = {}
        self.itos = {}
        self.pix2seq = pix2seq
        self.maxx = input_size  # height
        self.maxy = input_size  # width
        self.sep_xy = sep_xy
        self.special_tokens = [PAD, SOS, EOS, UNK, MASK]
        self.tokens = [Rxn, Rct, Prd, Cnd, Idt, Mol, Txt, Sup]
        self.fit_tokens(self.tokens)

    def __len__(self):
        if self.pix2seq:
            return 2094
        if self.sep_xy:
            return self.offset + self.maxx + self.maxy
        else:
            return self.offset + max(self.maxx, self.maxy)

    @property
    def PAD_ID(self):
        return self.stoi[PAD]

    @property
    def SOS_ID(self):
        return self.stoi[SOS]

    @property
    def EOS_ID(self):
        return self.stoi[EOS]

    @property
    def UNK_ID(self):
        return self.stoi[UNK]

    @property
    def offset(self):
        return 0 if self.pix2seq else len(self.stoi)

    @property
    def output_constraint(self):
        return False

    def fit_tokens(self, tokens):
        vocab = self.special_tokens + tokens
        if self.pix2seq:
            for i, s in enumerate(vocab):
                self.stoi[s] = 2001 + i
            self.stoi[EOS] = len(self) - 2
            self.stoi[Noise] = len(self) - 1
        else:
            for i, s in enumerate(vocab):
                self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        self.bbox_category_to_token = {1: Mol, 2: Txt, 3: Idt, 4: Sup}
        self.token_to_bbox_category = {item[1]: item[0] for item in self.bbox_category_to_token.items()}

    def is_x(self, x):
        return 0 <= x - self.offset < self.maxx

    def is_y(self, y):
        if self.sep_xy:
            return self.maxx <= y - self.offset < self.maxx + self.maxy
        return 0 <= y - self.offset < self.maxy

    def x_to_id(self, x):
        assert 0 <= x <= 1
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y):
        assert 0 <= y <= 1
        if self.sep_xy:
            return self.offset + self.maxx + round(y * (self.maxy - 1))
        return self.offset + round(y * (self.maxy - 1))

    def id_to_x(self, id, scale=1):
        if not self.is_x(id):
            return -1
        return (id - self.offset) / (self.maxx - 1) / scale

    def id_to_y(self, id, scale=1):
        if not self.is_y(id):
            return -1
        if self.sep_xy:
            return (id - self.offset - self.maxx) / (self.maxy - 1) * scale
        return (id - self.offset) / (self.maxy - 1) / scale

    def bbox_to_sequence(self, bbox, category):
        sequence = []
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return []
        sequence.append(self.x_to_id(x1))
        sequence.append(self.y_to_id(y1))
        sequence.append(self.x_to_id(x2))
        sequence.append(self.y_to_id(y2))
        sequence.append(self.stoi[self.bbox_category_to_token[category]])
        return sequence

    def sequence_to_bbox(self, sequence, scale=[1, 1]):
        if len(sequence) < 5:
            return None
        x1, y1 = self.id_to_x(sequence[0], scale[0]), self.id_to_y(sequence[1], scale[1])
        x2, y2 = self.id_to_x(sequence[2], scale[0]), self.id_to_y(sequence[3], scale[1])
        if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1 or x1 >= x2 or y1 >= y2 or sequence[4] not in self.itos:
            return None
        category = self.itos[sequence[4]]
        if category not in [Mol, Txt, Idt, Sup]:
            return None
        return {'category': category, 'bbox': (x1, y1, x2, y2), 'category_id': self.token_to_bbox_category[category]}

    def data_to_sequence(self, data):
        sequence = [self.SOS_ID]
        for reaction in data['reactions']:
            reactants = reaction['reactants']
            conditions = reaction['conditions']
            products = reaction['products']
            if all([data['area'][i] == 0 for i in reactants]) or all([data['area'][i] == 0 for i in products]):
                continue
            sequence.append(self.stoi[Rxn])
            sequence.append(self.stoi[Rct])
            for idx in reactants:
                sequence += self.bbox_to_sequence(data['boxes'][idx].tolist(), data['labels'][idx].item())
            sequence.append(self.stoi[Cnd])
            for idx in conditions:
                sequence += self.bbox_to_sequence(data['boxes'][idx].tolist(), data['labels'][idx].item())
            sequence.append(self.stoi[Prd])
            for idx in products:
                sequence += self.bbox_to_sequence(data['boxes'][idx].tolist(), data['labels'][idx].item())
        sequence.append(self.EOS_ID)
        return sequence

    def sequence_to_data(self, sequence, scores=None, scale=None):
        reactions = []
        i = 0
        flag = None
        if len(sequence) > 0 and sequence[0] == self.SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == self.EOS_ID:
                break
            if sequence[i] in self.itos:
                if self.itos[sequence[i]] == Rxn:
                    reactions.append({'reactants': [], 'conditions': [], 'products': []})
                    flag = None
                elif self.itos[sequence[i]] == Rct:
                    flag = 'reactants'
                elif self.itos[sequence[i]] == Cnd:
                    flag = 'conditions'
                elif self.itos[sequence[i]] == Prd:
                    flag = 'products'
            elif i+4 < len(sequence) and len(reactions) > 0 and flag is not None:
                bbox = self.sequence_to_bbox(sequence[i:i+5], scale)
                if bbox is not None:
                    reactions[-1][flag].append(bbox)
                    i += 4
            i += 1
        return reactions


class BboxTokenizer(ReactionTokenizer):

    def __init__(self, input_size=100, sep_xy=True, pix2seq=False, rand_target=False):
        super(BboxTokenizer, self).__init__(input_size, sep_xy, pix2seq)
        self.rand_target = rand_target

    def data_to_sequence(self, data):
        sequence = [self.SOS_ID]
        if self.rand_target:
            perm = np.random.permutation(len(data['boxes']))
            boxes = data['boxes'][perm].tolist()
            labels = data['labels'][perm].tolist()
        else:
            boxes = data['boxes'].tolist()
            labels = data['labels'].tolist()
        for bbox, category in zip(boxes, labels):
            sequence += self.bbox_to_sequence(bbox, category)
        sequence.append(self.EOS_ID)
        return sequence

    def sequence_to_data(self, sequence, scores=None, scale=None):
        bboxes = []
        i = 0
        if len(sequence) > 0 and sequence[0] == self.SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == self.EOS_ID:
                break
            if i+4 < len(sequence):
                bbox = self.sequence_to_bbox(sequence[i:i+5], scale)
                if bbox is not None:
                    if scores is not None:
                        bbox['score'] = scores[i + 4]
                    bboxes.append(bbox)
                    i += 4
            i += 1
        return bboxes


def get_tokenizer(args):
    tokenizer = {}
    if args.pix2seq:
        args.coord_bins = 2000
        args.sep_xy = False
    for format_ in args.formats:
        if format_ == 'reaction':
            tokenizer[format_] = ReactionTokenizer(args.coord_bins, args.sep_xy, args.pix2seq)
        if format_ == 'bbox':
            tokenizer[format_] = BboxTokenizer(args.coord_bins, args.sep_xy, args.pix2seq, args.rand_target)
    return tokenizer
