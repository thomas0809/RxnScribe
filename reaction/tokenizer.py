import json
import copy
import random
import numpy as np


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
    def max_len(self):
        return 256

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
    def NOISE_ID(self):
        return self.stoi[Noise]

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
        if x < -0.001 or x > 1.001:
            print(x)
        else:
            x = min(max(x, 0), 1)
        assert 0 <= x <= 1
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y):
        if y < -0.001 or y > 1.001:
            print(y)
        else:
            y = min(max(y, 0), 1)
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

    def random_category(self):
        return random.choice(list(self.bbox_category_to_token.keys()))
        # return random.choice([random.choice(list(self.bbox_category_to_token.keys())), self.NOISE_ID])

    def random_bbox(self):
        _x1, _y1, _x2, _y2 = random.random(), random.random(), random.random(), random.random()
        x1, y1, x2, y2 = min(_x1, _x2), min(_y1, _y2), max(_x1, _x2), max(_y1, _y2)
        category = self.random_category()
        return [x1, y1, x2, y2], category

    def jitter_bbox(self, bbox, ratio=0.2):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        _x1 = x1 + random.uniform(-w*ratio, w*ratio)
        _y1 = y1 + random.uniform(-h*ratio, h*ratio)
        _x2 = x2 + random.uniform(-w * ratio, w * ratio)
        _y2 = y2 + random.uniform(-h * ratio, h * ratio)
        x1, y1, x2, y2 = min(_x1, _x2), min(_y1, _y2), max(_x1, _x2), max(_y1, _y2)
        category = self.random_category()
        return np.clip([x1, y1, x2, y2], 0, 1), category

    def augment_box(self, bboxes):
        if len(bboxes) == 0:
            return self.random_bbox()
        if random.random() < 0.5:
            return self.random_bbox()
        else:
            return self.jitter_bbox(random.choice(bboxes))

    def bbox_to_sequence(self, bbox, category):
        sequence = []
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return []
        sequence.append(self.x_to_id(x1))
        sequence.append(self.y_to_id(y1))
        sequence.append(self.x_to_id(x2))
        sequence.append(self.y_to_id(y2))
        if category in self.bbox_category_to_token:
            sequence.append(self.stoi[self.bbox_category_to_token[category]])
        else:
            sequence.append(self.stoi[Noise])
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

    def perturb_reaction(self, reaction, boxes):
        reaction = copy.deepcopy(reaction)
        options = []
        options.append(0)  # Option 0: add
        if not(len(reaction['reactants']) == 1 and len(reaction['conditions']) == 0 and len(reaction['products']) == 1):
            options.append(1)  # Option 1: delete
            options.append(2)  # Option 2: move
        choice = random.choice(options)
        if choice == 0:
            key = random.choice(['reactants', 'conditions', 'products'])
            # TODO: insert to a random position
            # We simply add a random box, which may be a duplicate box in this reaction
            reaction[key].append(random.randrange(len(boxes)))
        if choice == 1 or choice == 2:
            options = []
            for key, val in [('reactants', 1), ('conditions', 0), ('products', 1)]:
                if len(reaction[key]) > val:
                    options.append(key)
            key = random.choice(options)
            idx = random.randrange(len(reaction[key]))
            del_box = reaction[key][idx]
            reaction[key] = reaction[key][:idx] + reaction[key][idx+1:]
            if choice == 2:
                options = ['reactants', 'conditions', 'products']
                options.remove(key)
                newkey = random.choice(options)
                reaction[newkey].append(del_box)
        return reaction

    def augment_reaction(self, reactions, data):
        area, boxes, labels = data['area'], data['boxes'], data['labels']
        nonempty_boxes = [i for i in range(len(area)) if area[i] > 0]
        if len(nonempty_boxes) == 0:
            return []
        if random.randrange(100) < 10:
            num_reactants = random.randint(1, 3)
            num_conditions = random.randint(0, 3)
            num_products = random.randint(1, 3)
            reaction = {
                'reactants': random.choices(nonempty_boxes, k=num_reactants),
                'conditions': random.choices(nonempty_boxes, k=num_conditions),
                'products': random.choices(nonempty_boxes, k=num_products)
            }
        else:
            if len(reactions) == 0:
                return []
            reaction = self.perturb_reaction(random.choice(reactions), boxes)
        seq = self.reaction_to_sequence(reaction, data)
        return seq

    def reaction_to_sequence(self, reaction, data):
        area, boxes, labels = data['area'], data['boxes'], data['labels']
        # If reactants or products are empty (because of image cropping), skip the reaction
        if all([area[i] == 0 for i in reaction['reactants']]) or all([area[i] == 0 for i in reaction['products']]):
            return []
        sequence = []
        for idx in reaction['reactants']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        sequence.append(self.stoi[Rct])
        for idx in reaction['conditions']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        sequence.append(self.stoi[Cnd])
        for idx in reaction['products']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        sequence.append(self.stoi[Prd])
        sequence.append(self.stoi[Rxn])
        return sequence

    def data_to_sequence(self, data, add_noise=False, rand_target=False):
        sequence = [self.SOS_ID]
        reactions = copy.deepcopy(data['reactions'])
        if rand_target:
            random.shuffle(reactions)
        for reaction in reactions:
            sequence += self.reaction_to_sequence(reaction, data)
        sequence_out = copy.deepcopy(sequence)
        if add_noise:
            while len(sequence) < self.max_len:
                seq = self.augment_reaction(reactions, data)
                if len(seq) == 0:
                    break
                sequence += seq
                seq_out = [self.PAD_ID] * len(seq)
                seq_out[-1] = self.NOISE_ID
                sequence_out += seq_out
        sequence.append(self.EOS_ID)
        sequence_out.append(self.EOS_ID)
        return sequence, sequence_out

    def sequence_to_data(self, sequence, scores=None, scale=None):
        reactions = []
        i = 0
        cur_reaction = {'reactants': [], 'conditions': [], 'products': []}
        flag = 'reactants'
        if len(sequence) > 0 and sequence[0] == self.SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == self.EOS_ID:
                break
            if sequence[i] in self.itos:
                if self.itos[sequence[i]] == Rxn:
                    if len(cur_reaction['reactants']) > 0 and len(cur_reaction['products']) > 0:
                        reactions.append(cur_reaction)
                    cur_reaction = {'reactants': [], 'conditions': [], 'products': []}
                    flag = 'reactants'
                elif self.itos[sequence[i]] == Rct:
                    flag = 'conditions'
                elif self.itos[sequence[i]] == Cnd:
                    flag = 'products'
                elif self.itos[sequence[i]] == Prd:
                    flag = None
            elif i+5 <= len(sequence) and flag is not None:
                bbox = self.sequence_to_bbox(sequence[i:i+5], scale)
                if bbox is not None:
                    cur_reaction[flag].append(bbox)
                    i += 4
            i += 1
        return reactions

class OldReactionTokenizer(ReactionTokenizer):

    def reaction_to_sequence(self, reaction, data):
        area, boxes, labels = data['area'], data['boxes'], data['labels']
        # If reactants or products are empty (because of image cropping), skip the reaction
        if all([area[i] == 0 for i in reaction['reactants']]) or all([area[i] == 0 for i in reaction['products']]):
            return []
        sequence = [self.stoi[Rxn]]
        sequence.append(self.stoi[Rct])
        for idx in reaction['reactants']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        sequence.append(self.stoi[Cnd])
        for idx in reaction['conditions']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        sequence.append(self.stoi[Prd])
        for idx in reaction['products']:
            if area[idx] == 0:
                continue
            sequence += self.bbox_to_sequence(boxes[idx].tolist(), labels[idx].item())
        return sequence

    def data_to_sequence(self, data, add_noise=False, rand_target=False):
        sequence = [self.SOS_ID]
        reactions = copy.deepcopy(data['reactions'])
        if rand_target:
            random.shuffle(reactions)
        for reaction in reactions:
            sequence += self.reaction_to_sequence(reaction, data)
        sequence_out = copy.deepcopy(sequence)
        if add_noise:
            first = True
            while len(sequence) < self.max_len:
                seq = self.augment_reaction(reactions, data)
                if len(seq) == 0:
                    break
                sequence += seq
                seq_out = [self.PAD_ID] * len(seq)
                if not first:
                    seq_out[0] = self.NOISE_ID
                first = False
                sequence_out += seq_out
        sequence.append(self.EOS_ID)
        sequence_out.append(self.EOS_ID)
        return sequence, sequence_out

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
        reactions = [r for r in reactions if len(r['reactants'] + r['conditions'] + r['products']) > 0]
        return reactions


class BboxTokenizer(ReactionTokenizer):

    def __init__(self, input_size=100, sep_xy=True, pix2seq=False):
        super(BboxTokenizer, self).__init__(input_size, sep_xy, pix2seq)

    @property
    def max_len(self):
        return 500

    def data_to_sequence(self, data, add_noise=False, rand_target=False):
        sequence = [self.SOS_ID]
        sequence_out = [self.SOS_ID]
        if rand_target:
            perm = np.random.permutation(len(data['boxes']))
            boxes = data['boxes'][perm].tolist()
            labels = data['labels'][perm].tolist()
        else:
            boxes = data['boxes'].tolist()
            labels = data['labels'].tolist()
        for bbox, category in zip(boxes, labels):
            seq = self.bbox_to_sequence(bbox, category)
            sequence += seq
            # sequence[-1] = self.random_category()
            sequence_out += seq
        if add_noise:
            while len(sequence) < self.max_len:
                bbox, category = self.augment_box(boxes)
                sequence += self.bbox_to_sequence(bbox, category)
                sequence_out += [self.PAD_ID] * 4 + [self.NOISE_ID]
        sequence.append(self.EOS_ID)
        sequence_out.append(self.EOS_ID)
        return sequence, sequence_out

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
            tokenizer[format_] = BboxTokenizer(args.coord_bins, args.sep_xy, args.pix2seq)
    return tokenizer
