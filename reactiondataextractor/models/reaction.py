# -*- coding: utf-8 -*-
"""
Reaction
=======

This module contains classes for representing reaction elements.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from itertools import product

from .exceptions import NotAnArrowException
from .segments import Panel, PanelMethodsMixin
from .utils import Point, PrettyFrozenSet

log = logging.getLogger('extract.reaction')


class BaseReactionClass(object):
    """
    This is a base.py reaction class placeholder
    """


class Diagram(BaseReactionClass, PanelMethodsMixin):
    """This is a base.py class for chemical structures species found in diagrams (e.g. reactants and products)

    :param panel: bounding box a diagrams
    :type panel: Panel
    :param label: label associated with a diagram
    :type label: Label
    :param smiles: SMILES associated with a diagram
    :type smiles: str
    :param crop: crop containing the diagram
    :type crop: Crop"""

    @classmethod
    def from_coords(cls, left, right, top, bottom, label=None, smiles=None, crop=None):
        """Class method used for instantiation from coordinates, as used within chemschematicresolver"""
        panel = Panel(left, right, top, bottom)
        return cls(panel=panel, label=label, smiles=smiles, crop=crop)

    def __init__(self, panel, label=None, smiles=None, crop=None):
        self._panel = panel
        self._label = label
        self._smiles = smiles
        self._crop = crop
        super().__init__()

    def __eq__(self, other):
        if isinstance(other, Diagram):  # Only compare exact same types
            return self.panel == other.panel and self.label == other.label

    def __hash__(self):
        return hash(self.panel)

    def __repr__(self):
        return f'{self.__class__.__name__}(panel={self.panel}, smiles={self.smiles}, label={self.label})'

    def __str__(self):
        return f'{self.smiles if self.smiles else "???"}, label: {self.label}'

    @property
    def panel(self):
        return self._panel

    @property
    def center(self):
        return self._panel.center

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def smiles(self):
        return self._smiles

    @smiles.setter
    def smiles(self, smiles):
        self._smiles = smiles

    @property
    def crop(self):
        """ Cropped Figure object of the specific diagram"""
        return self._crop

    @crop.setter
    def crop(self, fig):
        self._crop = fig

    def compass_position(self, other):
        """ Determines the compass position (NSEW) of other relative to self"""

        length = other.center[0] - self.center[0]
        height = other.center[1] - self.center[1]

        if abs(length) > abs(height):
            if length > 0:
                return 'E'
            else:
                return 'W'
        elif abs(length) < abs(height):
            if height > 0:
                return 'S'
            else:
                return 'N'

        else:
            return None


class ReactionStep(BaseReactionClass):
    """
    This class describes elementary steps in a reaction.

    :param reactants: reactants of a reaction
    :type reactants: frozenset[Diagram]
    :param products: products of a reaction
    :type products: frozenset[Diagram]
    :param conditions: reaction conditions for the step
    :type conditions: Conditions
    """

    def __init__(self, reactants, products, conditions):
        self.reactants = PrettyFrozenSet(reactants)
        self.products = PrettyFrozenSet(products)
        self.conditions = conditions

    def __eq__(self, other):
        return (self.reactants == other.reactants and self.products == other.products and
                self.conditions == other.conditions)

    def __repr__(self):
        return f'ReactionStep(reactants=({self.reactants}),products=({self.products}),{self.conditions})'

    def __str__(self):
        reactant_strings = [elem.smiles if elem.smiles else '???' for elem in self.reactants]
        product_strings = [elem.smiles if elem.smiles else '???' for elem in self.products]
        return ' + '.join(reactant_strings)+'  -->  ' + ' + '.join(product_strings)

    def __hash__(self):
        all_species = [species for group in iter(self) for species in group]
        species_hash = sum([hash(species) for species in all_species])
        return hash(self.conditions) + species_hash

    def __iter__(self):
        return iter ((self.reactants, self.products))


class Conditions:
    """
    This class describes conditions region and associated text

    :param text_lines: extracted text lines containing conditions
    :type text_lines: list[TextLine]
    :param conditions_dct: dictionary with all parsed conditions
    :type conditions_dct: dict
    :param arrow: reaction arrow, around which the search for conditions is performed
    :type arrow: SolidArrow
    :param structure_panels: bounding boxes of all chemical structures found in the region
    :type structure_panels: list[Panel]
    """

    def __init__(self, text_lines, conditions_dct, arrow, structure_panels=None):
        self.arrow = arrow
        self.text_lines = text_lines
        self.conditions_dct = conditions_dct

        if structure_panels is None:
            structure_panels = []
        self._structure_panels = structure_panels
        self.diags = None
        self.text_lines.sort(key=lambda textline: textline.panel.top)

    def __repr__(self):
        return f'Conditions({self.text_lines}, {self.conditions_dct}, {self.arrow})'

    def __str__(self):
        delimiter = '\n------\n'
        return delimiter + 'Step conditions:' + \
               '\n'.join(f'{key} : {value}' for key, value in self.conditions_dct.items() if value)  + delimiter

    def __eq__(self, other):
        if other.__class__ == self.__class__:
            return self.conditions_dct == other.conditions_dct

        else:
            return False

    def __hash__(self):
        return hash(sum(hash(line) for line in self.text_lines))

    @property
    def structure_panels(self):
        return self._structure_panels

    @property
    def anchor(self):
        a_pixels = self.arrow.pixels
        return a_pixels[len(a_pixels)//2]

    @property
    def coreactants(self):
        return self.conditions_dct['coreactants']

    @property
    def catalysts(self):
        return self.conditions_dct['catalysts']

    @property
    def other_species(self):
        return self.conditions_dct['other species']

    @property
    def temperature(self):
        return self.conditions_dct['temperature']

    @property
    def time(self):
        return self.conditions_dct['time']

    @property
    def pressure(self):
        return self.conditions_dct['pressure']

    @property
    def yield_(self):
        return self.conditions_dct['yield']


class Label(PanelMethodsMixin):
    """Describes labels and recgonised text

    :param panel: bounding box a label
    :type panel: Panel
    :param text: label text
    :type text: str
    :param r_group: generic r_groups associated with a label
    :type r_group: str"""

    @classmethod
    def from_coords(cls, left, right, top, bottom, text):
        panel = Panel(left, right, top, bottom)
        return cls(panel, text)

    def __init__(self, panel, text=None, r_group=None):
        if r_group is None:
            r_group = []
        if text is None:
            text = []
        self.panel = panel
        self._text = text
        self.r_group = r_group

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    def __repr__(self):
        return f'Label(panel={self.panel}, text={self.text}, r_group={self.r_group})'

    def __str__(self):
        return f'Label(Text: {", ".join(sent.text.strip() for sent in self.text)})'

    def __hash__(self):
        return hash(self.panel)

    def __eq__(self, other):
        return isinstance(other, Label) and self.panel == other.panel


    def add_r_group_variables(self, var_value_label_tuples):
        """ Updates the R-groups for this label."""

        self.r_group.append(var_value_label_tuples)


class BaseArrow(PanelMethodsMixin):
    """Base arrow class common to all arrows

    :param pixels: pixels forming the arrows
    :type pixels: list[Point]
    :param line: line found by Hough transform, underlying primitive,
    :type line: Line
    :param panel: bounding box of an arrow
    :type panel: Panel"""

    def __init__(self, pixels, line, panel):
        if not all(isinstance(pixel, Point) for pixel in pixels):
            self.pixels = [Point(row=coords[0], col=coords[1]) for coords in pixels]
        else:
            self.pixels = pixels

        self.line = line
        self._panel = panel
        slope = self.line.slope
        self.sort_pixels()
        self._center_px = None

    @property
    def panel(self):
        return self._panel

    @property
    def is_vertical(self):
        return self.line.is_vertical

    @property
    def center_px(self):
        """
        Based on a geometric centre of an arrow panel, looks for a pixel nearby that belongs to the arrow.

        :return: coordinates of the pixel that is closest to geometric centre and belongs to the object.
        If multiple pairs found, return the floor average.
        :rtype: Point
        """
        if self._center_px is not None:
            return self._center_px

        log.debug('Finding center of an arrow...')
        x, y = self.panel.geometric_centre

        log.debug('Found an arrow with geometric center at (%s, %s)' % (y, x))

        # Look at pixels neighbouring center to check which actually belong to the arrow
        x_candidates = [x+i for i in range(-3, 4)]
        y_candidates = [y+i for i in range(-3, 4)]
        center_candidates = [candidate for candidate in product(x_candidates, y_candidates) if
                             Point(row=candidate[1], col=candidate[0]) in self.pixels]

        log.debug('Possible center pixels: %s', center_candidates)
        if center_candidates:
            self._center_px = np.mean(center_candidates, axis=0, dtype=int)
            self._center_px = Point(row=self._center_px[1], col=self._center_px[0])
        else:
            raise NotAnArrowException('No component pixel lies on the geometric centre')
        log.debug('Center pixel found: %s' % self._center_px)

        return self._center_px

    def sort_pixels(self):
        """
        Simple pixel sort.

        Sorts pixels by row in vertical arrows and by column in all other arrows
        :return:
        """
        if self.is_vertical:
            self.pixels.sort(key=lambda pixel: pixel.row)
        else:
            self.pixels.sort(key=lambda pixel: pixel.col)


class SolidArrow(BaseArrow):
    """
    Class used to represent simple reaction arrows.

    :param pixels: pixels forming the arrows
    :type pixels: list[Point]
    :param line: line found by Hough transform, underlying primitive,
    :type line: Line
    :param panel: bounding box of an arrow
    :type panel: Panel"""

    def __init__(self, pixels, line, panel):
        super(SolidArrow, self).__init__(pixels, line, panel)
        self.react_side = None
        self.prod_side = None
        a_ratio = self.panel.aspect_ratio
        a_ratio = 1/a_ratio if a_ratio < 1 else a_ratio
        if a_ratio < 3:
            raise NotAnArrowException('aspect ratio is not within the accepted range')

        self.react_side, self.prod_side = self.get_direction()
        pixel_majority = len(self.prod_side) - len(self.react_side)
        num_pixels = len(self.pixels)
        min_pixels = min(int(0.1 * num_pixels), 15)
        if pixel_majority < min_pixels:
            raise NotAnArrowException('insufficient pixel majority')
        elif pixel_majority < 2 * min_pixels:
            log.warning('Difficulty detecting arrow sides - low pixel majority')

        log.debug('Arrow accepted!')

    def __repr__(self):
        return f'SolidArrow(pixels={self.pixels}, line={self.line}, panel={self.panel})'

    def __str__(self):
        left, right, top, bottom = self.panel
        return f'SolidArrow({left, right, top, bottom})'

    def __eq__(self, other):
        return self.panel == other.panel

    def __hash__(self):
        return hash(pixel for pixel in self.pixels)

    @property
    def hook(self):
        """
        Returns the last pixel of an arrow hook.
        :return:
        """
        if self.is_vertical:
            prod_side_lhs = True if self.prod_side[0].row < self.react_side[0].row else False
        else:
            prod_side_lhs = True if self.prod_side[0].col < self.react_side[0].col else False
        return self.prod_side[0] if prod_side_lhs else self.prod_side[-1]

    def get_direction(self):
        """Retrieves the direction of an arrow by looking at the number of pixels on each side.

        Splits an arrow in the middle depending on its slope and calculated the number of pixels in each part."""
        center = self.center
        center = Point(center[1], center[0])
        if self.is_vertical:
            part_1 = [pixel for pixel in self.pixels if pixel.row < center.row]
            part_2 = [pixel for pixel in self.pixels if pixel.row > center.row]

        elif self.line.slope == 0:
            part_1 = [pixel for pixel in self.pixels if pixel.col < center.col]
            part_2 = [pixel for pixel in self.pixels if pixel.col > center.col]

        else:
            p_slope = -1/self.line.slope
            p_intercept = center.row - center.col*p_slope
            p_line = lambda point: point.col*p_slope + p_intercept
            part_1 = [pixel for pixel in self.pixels if pixel.row < p_line(pixel)]
            part_2 = [pixel for pixel in self.pixels if pixel.row > p_line(pixel)]

        if len(part_1) > len(part_2):
            react_side = part_2
            prod_side = part_1
        else:
            react_side = part_1
            prod_side = part_2

        log.debug('Established reactant and product sides of an arrow.')
        log.debug('Number of pixel on reactants side: %s ', len(react_side))
        log.debug('product side: %s ', len(prod_side))
        return react_side, prod_side
