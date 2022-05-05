# -*- coding: utf-8 -*-
"""
Utils
=======

This module contains simple, low-level utility classes involving geometry or implementing custom behaviour of
built-in types.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from math import sqrt
from numpy import hypot, inf

log = logging.getLogger(__name__)


class Point:
    """Simple class for representing points in a uniform, image (row, col) manner"""
    def __init__(self, row, col):
        self.row = int(row)
        self.col = int(col)

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.row == other.row and self.col == other.col
        else:
            return self.row == other[1] and self.col == other[0]    # Assume a tuple

    def __hash__(self):
        return hash(self.row + self.col)

    def __str__(self):
        return f'{self.row, self.col}'

    def __repr__(self):
        return self.__str__()   # to de-clutter more complex objects

    def separation(self, other):
        """
        Calculates distance between self and another point
        :param Point other: another Point object
        :return float: distance between two Points
        """

        drow = self.row - other.row
        dcol = self.col - other.col
        return hypot(drow, dcol)


class Line:
    """This is a utility class representing a line in 2D defined by two points
    :param pixels: pixels belonging to a line
    :type pixels: list[Point]"""

    def __init__(self, pixels):
        self.pixels = pixels
        self.is_vertical = None
        self.slope, self.intercept = self.get_line_parameters()

    def __iter__(self):
        return iter(self.pixels)

    def __getitem__(self, index):
        return self.pixels[index]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pixels})'

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = value
        self.is_vertical = True if self.slope == inf or abs(self.slope) > 10 else False

    def get_line_parameters(self):
        """
        Calculates slope and intercept of ``line``
        :return: slope and intercept of the line
        :rtype: tuple
        """
        p1 = self.pixels[0]
        x1, y1 = p1.col, p1.row

        p2 = self.pixels[-1]  # Can be any two points, but non-neighbouring points increase accuracy of calculation
        x2, y2 = p2.col, p2.row

        delta_x = x2 - x1
        delta_y = y2 - y1

        if delta_x == 0:
            slope = inf
        else:
            slope = delta_y / delta_x

        intercept_1 = y1 - slope * x1
        intercept_2 = y2 - slope * x2
        intercept = (intercept_1 + intercept_2) / 2

        return slope, intercept

    def distance_from_point(self, other):
        """Calculates distance between the line and a point
        :param Point other: Point from which the distance is calculated
        :return float: distance between line and a point
        """
        # p1, *_, p2 = self.points
        p1 = self.pixels[0]
        x1, y1 = p1.col, p1.row
        p2 = self.pixels[-1]
        x2, y2 = p2.col, p2.row

        x0, y0 = other.col, other.row

        top = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1-y2*x1)
        bottom = sqrt((y2-y1)**2+(x2-x1)**2)

        return top/bottom


class DisabledNegativeIndices:
    """If a negative index is passed to an underlying sequence, then an empty element of appropriate type is returned.
    Slices including negative start indices are corrected to start at 0

    :param sequence: underlying sequence-type object
    :type sequence: sequence"""
    def __init__(self, sequence):
        self._sequence = sequence

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if idx.start < 0:
                idx = slice(0, idx.stop, idx.step)

        elif isinstance(idx, int):
            if self._sequence:
                type_ = type(self._sequence[0])
                if idx < 0:
                    return type_()

        return self._sequence[idx]


class PrettyFrozenSet(frozenset):
    """Frozenset with a pretty __str__ method; used for depicting output
    :param frozenset_: underlying frozenset
    :type frozenset_: frozenset"""

    def __new__(cls, frozenset_):
        obj = super().__new__(cls, frozenset_)
        return obj

    def __init__(self, frozenset_):
        self._frozenset_ = frozenset_
        super().__init__()

    def __str__(self):
        return ", ".join([str(elem) for elem in self._frozenset_])


class PrettyList(list):
    """list with a pretty __str__ method; used for depicting output
        :param list_: underlying list
        :type list_: list"""
    # def __new__(cls, list_):
    #     obj = super().__new__(cls, list_)
    #     return obj

    def __init__(self, list_):
        self._list = list_
        super().__init__(list_)

    def __str__(self):
        try:
            return '\n'.join([str(elem) for elem in self._list])
        except Exception as e:
            print()
            print()
            print()
            print(self._list)
            print(e)

