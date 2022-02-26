# -*- coding: utf-8 -*-
"""
Model
=====

Models created to identify different regions of a chemical schematic diagram.

Module adapted by :-
author: Ed Beard
email: ejb207@cam.ac.uk

from FigureDataExtractor (<CITATION>) :-
author: Matthew Swain
email: m.swain@me.com

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

from . import decorators
import numpy as np

log = logging.getLogger(__name__)


@decorators.python_2_unicode_compatible
class Rect(object):
    """A rectangular region."""

    def __init__(self, left, right, top, bottom):
        """

        :param int left: Left edge of rectangle.
        :param int right: Right edge of rectangle.
        :param int top: Top edge of rectangle.
        :param int bottom: Bottom edge of rectangle.
        """
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    @property
    def width(self):
        """Return width of rectangle in pixels. May be floating point value.

        :rtype: int
        """
        return self.right - self.left

    @property
    def height(self):
        """Return height of rectangle in pixels. May be floating point value.

        :rtype: int
        """
        return self.bottom - self.top

    @property
    def perimeter(self):
        """Return length of the perimeter around rectangle.

        :rtype: int
        """
        return (2 * self.height) + (2 * self.width)

    @property
    def area(self):
        """Return area of rectangle in pixels. May be floating point values.

        :rtype: int
        """
        return self.width * self.height

    @property
    def center(self):
        """Center point of rectangle. May be floating point values.

        :rtype: tuple(int|float, int|float)
        """
        xcenter = (self.left + self.right) / 2
        ycenter = (self.bottom + self.top) / 2
        return xcenter, ycenter

    @property
    def center_px(self):
        """(x, y) coordinates of pixel nearest to center point.

        :rtype: tuple(int, int)
        """
        xcenter, ycenter = self.center
        return np.around(xcenter), np.around(ycenter)

    def contains(self, other_rect):
        """Return true if ``other_rect`` is within this rect.

        :param Rect other_rect: Another rectangle.
        :return: Whether ``other_rect`` is within this rect.
        :rtype: bool
        """
        return (other_rect.left >= self.left and other_rect.right <= self.right and
                other_rect.top >= self.top and other_rect.bottom <= self.bottom)

    def overlaps(self, other_rect):
        """Return true if ``other_rect`` overlaps this rect.

        :param Rect other_rect: Another rectangle.
        :return: Whether ``other_rect`` overlaps this rect.
        :rtype: bool
        """
        return (min(self.right, other_rect.right) > max(self.left, other_rect.left) and
                min(self.bottom, other_rect.bottom) > max(self.top, other_rect.top))

    def separation(self, other_rect):
        """ Returns the distance between the center of each graph

        :param Rect other_rect: Another rectangle
        :return: Distance between centoids of rectangle
        :rtype: float
        """
        length = abs(self.center[0] - other_rect.center[0])
        height = abs(self.center[1] - other_rect.center[1])
        return np.hypot(length, height)


    def __repr__(self):
        return '%s(left=%s, right=%s, top=%s, bottom=%s)' % (
            self.__class__.__name__, self.left, self.right, self.top, self.bottom
        )

    def __str__(self):
        return '<%s (%s, %s, %s, %s)>' % (self.__class__.__name__, self.left, self.right, self.top, self.bottom)

    def __eq__(self, other):
        if self.left == other.left and self.right == other.right \
                and self.top == other.top and self.bottom == other.bottom:
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.left, self.right, self.top, self.bottom))


class Panel(Rect):
    """ Tagged section inside Figure"""

    def __init__(self, left, right, top, bottom, tag=0):
        super(Panel, self).__init__(left, right, top, bottom)
        self.tag = tag
        self._repeating = False
        self._pixel_ratio = None

    @property
    def repeating(self):
        return self._repeating

    @repeating.setter
    def repeating(self, repeating):
        self._repeating = repeating

    @property
    def pixel_ratio(self):
        return self._pixel_ratio

    @pixel_ratio.setter
    def pixel_ratio(self, pixel_ratio):
        self._pixel_ratio = pixel_ratio


class Diagram(Panel):
    """ Chemical Schematic Diagram that is identified"""

    def __init__(self, *args, label=None, smile=None, fig=None):
        self._label = label
        self._smile = smile
        self._fig = fig
        super(Diagram, self).__init__(*args)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def smile(self):
        return self._smile

    @smile.setter
    def smile(self, smile):
        self._smile = smile

    @property
    def fig(self):
        """ Cropped Figure object of the specific diagram"""
        return self._fig

    @fig.setter
    def fig(self, fig):
        self._fig = fig

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

    def __repr__(self):
        if self.label is not None:
            return '%s(label=%s, smile=%s)' % (
                self.__class__.__name__, self.label.tag, self.smile
            )
        else:
            return '%s(label=None, smile=%s)' % (
                self.__class__.__name__, self.smile
            )

    def __str__(self):
        if self.label is not None:
            return '<%s (%s, %s)>' % (self.__class__.__name__, self.label.tag, self.smile)
        else:
            return '<%s (%s, %s)' % (self.__class__.__name__, self.tag, self.smile)


class Label(Panel):
    """ Label used as an identifier for the closest Chemical Schematic Diagram"""

    def __init__(self, *args):
        super(Label, self).__init__(*args)
        self.r_group = []
        self.values = []

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    def r_group(self):
        """ List of lists of tuples containing variable-value-label triplets.
            Each list represents a particular combination of chemicals yielding a unique compound.

            :param : List(str,str,List(str)) : A list of variable-value pairs and their list of candidate labels
        """
        return self.r_group

    def add_r_group_variables(self, var_value_label_tuples):
        """ Updates the R-groups for this label."""

        self.r_group.append(var_value_label_tuples)


class RGroup(object):
    """ Object containing all extracted information for an R-group result"""

    def __init__(self, var, value, label_candidates):
        self.var = var
        self.value = value
        self.label_candidates = label_candidates

    def __repr__(self):
        return '%s(variable=%s, value=%s, label_candidates=%s)' % (
            self.__class__.__name__, self.var, self.value, self.label_candidates
        )

    def __str__(self):
        return '%s(variable=%s, value=%s, label_candidates=%s)' % (
            self.__class__.__name__, self.var, self.value, self.label_candidates
        )

    def convert_to_tuple(self):
        """ Converts the r-group object to a usable a list of variable-value pairs and their list of candidate labels """
        tuple_r_group = (self.var, self.value, self.label_candidates)
        return tuple_r_group


@decorators.python_2_unicode_compatible
class Figure(object):
    """A figure image."""

    def __init__(self, img, panels=None, plots=None, photos=None):
        """

        :param numpy.ndarray img: Figure image.
        :param list[Panel] panels: List of panels.
        :param list[Plot] plots: List of plots.
        :param list[Photo] photos: List of photos.
        """
        self.img = img
        self.width, self.height = img.shape[0], img.shape[1]
        self.center = (int(self.width * 0.5), int(self.height) * 0.5)
        self.panels = panels
        self.plots = plots
        self.photos = photos

        # TODO: Image metadata?

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

    def __str__(self):
        return '<%s>' % self.__class__.__name__

    def get_bounding_box(self):
        """ Returns the Panel object for the extreme bounding box of the image

        :rtype: Panel()"""

        rows = np.any(self.img, axis=1)
        cols = np.any(self.img, axis=0)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(cols)[0][[0, -1]]
        return Panel(left, right, top, bottom)
