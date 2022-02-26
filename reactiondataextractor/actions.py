# -*- coding: utf-8 -*-
"""
Actions
========

This module contains important high level processing routines

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
import os
import logging
import numpy as np

from skimage.transform import probabilistic_hough_line

from .models.segments import Panel, Figure, FigureRoleEnum
from .models.utils import Point, Line
from .utils.processing import isolate_patches, skeletonize
from . import settings

log = logging.getLogger('extract.actions')

formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler(os.path.join(settings.ROOT_DIR, 'actions.log'))
file_handler.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setFormatter(formatter)

log.addHandler(file_handler)
log.addHandler(ch)


def estimate_single_bond(fig):
    """Estimates length of a single bond in an image

    Uses a skeletonise image to find the number of lines of differing lengths. The single bond length is chosen using
    a graph of number of detected lines vs. the length of a line. The optimal value is where the change in number of
    lines as the length varies is greatest.
    :param Figure fig: analysed figure
    :return: approximate length of a single bond
    :rtype: int"""
    ccs = fig.connected_components
    # Get a rough bond length (line length) value from the two largest structures
    ccs = sorted(ccs, key=lambda cc: cc.area, reverse=True)
    estimation_fig = skeletonize(isolate_patches(fig, ccs[:2]))

    length_scan_param = 0.025 * max(fig.width, fig.height)
    length_scan_start = length_scan_param if length_scan_param > 20 else 20
    min_line_lengths = np.linspace(length_scan_start, 3 * length_scan_start, 20)
    num_lines = [(length, len(probabilistic_hough_line(estimation_fig.img, line_length=int(length), threshold=15)))
                 for length in min_line_lengths]

    # Choose the value where the number of lines starts to drop most rapidly and assign it as the ''single_bond''
    (single_bond, _), (_, _) = min(zip(num_lines, num_lines[1:]), key=lambda pair: pair[1][1] - pair[0][1])
    # the key function is difference in number of detected lines between adjacent pairs
    return int(single_bond)


def extend_line(line, extension=None):
    """
    Extends line in both directions. Output is a pair of points, each of which is further from an arrow (closer to
    reactants or products in the context of reactions).
    :param Line line: original Line object
    :param int extension: value dictated how far the new line should extend in each direction
    :return: two endpoints of a new line
    :rtype: tuple[int]
    """

    if line.is_vertical:
        line.pixels.sort(key=lambda point: point.row)
        first_line_pixel = line.pixels[0]
        last_line_pixel = line.pixels[-1]
        if extension is None:
            extension = int((last_line_pixel.separation(first_line_pixel)) * 0.4)

        left_extended_point = Point(row=first_line_pixel.row - extension, col=first_line_pixel.col)
        right_extended_point = Point(row=last_line_pixel.row + extension, col=last_line_pixel.col)

    else:
        line.pixels.sort(key=lambda point: point.col)

        first_line_pixel = line.pixels[0]
        last_line_pixel = line.pixels[-1]
        if extension is None:
            extension = int((last_line_pixel.separation(first_line_pixel)) * 0.4)

        if line.slope == 0:
            left_extended_last_y = line.slope * (first_line_pixel.col - extension) + first_line_pixel.row
            right_extended_last_y = line.slope * (last_line_pixel.col + extension) + first_line_pixel.row

        else:
            left_extended_last_y = line.slope*(first_line_pixel.col-extension) + line.intercept
            right_extended_last_y = line.slope*(last_line_pixel.col+extension) + line.intercept

        left_extended_point = Point(row=left_extended_last_y, col=first_line_pixel.col-extension)
        right_extended_point = Point(row=right_extended_last_y, col=last_line_pixel.col+extension)

    return left_extended_point, right_extended_point


def find_nearby_ccs(start, all_relevant_ccs, distances, role=None, condition=(lambda cc: True)):
    """
    Find all structures close to ``start`` position. All found structures are added to a queue and
    checked again to form a cluster of nearby structures.

    Found ccs are added to a frontier if they satisfy a condition. Elements of the frontier are removed one by one and
    their neighbourhoods checked for new ccs. The search ends when the frontier is empty. Roles of found ccs may also
    be set after the routine is completed.
    :param Point|(x,y)|Panel start: reference object where the search starts
    :param [Panel,...] all_relevant_ccs: list of all found structures
    :param FigureRoleEnum|ReactionRoleEnum role: class specifying role of the ccs in the scheme
                                                                     (e.g. Diagram, Conditions)
    :param (float, lambda) distances: a tuple (maximum_initial_distance, distance_function) which specifies allowed
    distance from the starting point and a function defining cut-off distance for subsequent reference ccs
    :param lambda condition: optional condition to decide whether a connected component should be added to the frontier
                                                                                                                or not.
    :return: List of all nearby structures
    """
    max_initial_distance, distance_fn = distances
    frontier = [start]
    found_ccs = []
    visited = set()
    while frontier:
        reference = frontier.pop()
        visited.add(reference)
        max_distance = distance_fn(reference) if isinstance(reference, Panel) else max_initial_distance
        successors = [cc for cc in all_relevant_ccs if cc.separation(reference) < max_distance
                      and cc not in visited and condition(cc)]
        newly_found = [found_cc for found_cc in successors if found_cc not in found_ccs]
        frontier.extend(successors)
        found_ccs.extend(newly_found)

    if role is not None:
        [setattr(cc, 'role', role) for cc in found_ccs if not getattr(cc, 'role')]

    return found_ccs
