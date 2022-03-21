import json
import random
import numpy as np

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
MASK = '<mask>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4

Rxn = '[Rxn]'  # Reaction
Rct = '[Rct]'  # Reactant
Prd = '[Prd]'  # Product
Cnd = '[Cnd]'  # Condition
Idt = '[Idt]'  # Identifier
Mol = '[Mol]'  # Molecule
Txt = '[Txt]'  # Text
Sup = '[Sup]'  # Supplement

FORMAT_INFO = {
    "reaction": {"max_len": 256},
    "bbox": {"max_len": 256}
}


class ReactionTokenizer(object):

    def __init__(self, input_size=100, sep_xy=True, debug=False):
        self.stoi = {}
        self.itos = {}
        self.maxx = input_size  # height
        self.maxy = input_size  # width
        self.sep_xy = sep_xy
        self.special_tokens = [PAD, SOS, EOS, UNK, MASK]
        self.tokens = [Rxn, Rct, Prd, Cnd, Idt, Mol, Txt, Sup]
        self.fit_tokens(self.tokens)
        self.debug = debug

    def __len__(self):
        if self.sep_xy:
            return self.offset + self.maxx + self.maxy
        else:
            return self.offset + max(self.maxx, self.maxy)

    @property
    def offset(self):
        return len(self.stoi)

    @property
    def output_constraint(self):
        return False

    def fit_tokens(self, tokens):
        vocab = self.special_tokens + tokens
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID
        assert self.stoi[MASK] == MASK_ID
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def is_x(self, x):
        return self.offset <= x < self.offset + self.maxx

    def is_y(self, y):
        if self.sep_xy:
            return self.offset + self.maxx <= y
        return self.offset <= y

    def x_to_id(self, x):
        assert 0 <= x <= 1
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y):
        assert 0 <= y <= 1
        if self.sep_xy:
            return self.offset + self.maxx + round(y * (self.maxy - 1))
        return self.offset + round(y * (self.maxy - 1))

    def id_to_x(self, id):
        if not self.is_x(id):
            return -1
        return (id - self.offset) / (self.maxx - 1)

    def id_to_y(self, id):
        if not self.is_y(id):
            return -1
        if self.sep_xy:
            return (id - self.offset - self.maxx) / (self.maxy - 1)
        return (id - self.offset) / (self.maxy - 1)

    def bbox_to_sequence(self, bbox, width, height):
        sequence = []
        if bbox['category_id'] == 1:
            sequence.append(self.stoi[Mol])
        elif bbox['category_id'] == 2:
            sequence.append(self.stoi[Txt])
        elif bbox['category_id'] == 3:
            sequence.append(self.stoi[Idt])
        else:
            sequence.append(self.stoi[Sup])
        x, y, w, h = bbox['bbox']
        sequence.append(self.x_to_id(x / width))
        sequence.append(self.y_to_id(y / height))
        sequence.append(self.x_to_id((x + w) / width))
        sequence.append(self.y_to_id((y + h) / height))
        return sequence

    def sequence_to_bbox(self, sequence):
        category = self.itos[sequence[0]]
        x1, y1 = self.id_to_x(sequence[1]), self.id_to_y(sequence[2])
        x2, y2 = self.id_to_x(sequence[3]), self.id_to_y(sequence[4])
        if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1:
            return None
        return {'category': category, 'bbox': (x1, y1, x2, y2)}

    def data_to_sequence(self, data):
        sequence = [SOS_ID]
        for reaction in data['reactions']:
            reactants = reaction['reactants']
            conditions = reaction['conditions']
            products = reaction['products']
            sequence.append(self.stoi[Rxn])
            sequence.append(self.stoi[Rct])
            for idx in reactants:
                sequence += self.bbox_to_sequence(data['bboxes'][idx], data['width'], data['height'])
            sequence.append(self.stoi[Cnd])
            for idx in conditions:
                sequence += self.bbox_to_sequence(data['bboxes'][idx], data['width'], data['height'])
            sequence.append(self.stoi[Prd])
            for idx in products:
                sequence += self.bbox_to_sequence(data['bboxes'][idx], data['width'], data['height'])
        sequence.append(EOS_ID)
        return sequence

    def sequence_to_data(self, sequence):
        reactions = []
        i = 0
        flag = None
        if sequence[0] == SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == EOS_ID:
                break
            if sequence[i] < self.offset:
                if self.itos[sequence[i]] == Rxn:
                    reactions.append({'reactants': [], 'conditions': [], 'products': []})
                    flag = None
                elif self.itos[sequence[i]] == Rct:
                    flag = 'reactants'
                elif self.itos[sequence[i]] == Cnd:
                    flag = 'conditions'
                elif self.itos[sequence[i]] == Prd:
                    flag = 'products'
                elif i+4 < len(sequence) and self.itos[sequence[i]] in [Mol, Txt, Idt, Sup]:
                    if len(reactions) > 0 and flag is not None:
                        bbox = self.sequence_to_bbox(sequence[i:i+5])
                        if bbox is not None:
                            reactions[-1][flag].append(bbox)
                            i += 4
            i += 1
        return reactions


class BboxTokenizer(ReactionTokenizer):

    def data_to_sequence(self, data):
        sequence = [SOS_ID]
        for bbox in data['bboxes']:
            sequence += self.bbox_to_sequence(bbox, data['width'], data['height'])
        sequence.append(EOS_ID)
        return sequence

    def sequence_to_data(self, sequence):
        bboxes = []
        i = 0
        if sequence[0] == SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == EOS_ID:
                break
            if sequence[i] < self.offset:
                if i+4 < len(sequence) and self.itos[sequence[i]] in [Mol, Txt, Idt, Sup]:
                    bbox = self.sequence_to_bbox(sequence[i:i+5])
                    if bbox is not None:
                        bboxes.append(bbox)
                        i += 4
            i += 1
        return bboxes


def get_tokenizer(args):
    tokenizer = {}
    for format_ in args.formats:
        if format_ == 'reaction':
            tokenizer[format_] = ReactionTokenizer(args.coord_bins)
        if format_ == 'bbox':
            tokenizer[format_] = BboxTokenizer(args.coord_bins)
    return tokenizer
