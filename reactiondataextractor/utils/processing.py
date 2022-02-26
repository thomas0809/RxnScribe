# -*- coding: utf-8 -*-
"""
Processing
==========

This module contains low level processing routines

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
from __future__ import absolute_import
from __future__ import division

from collections.abc import Container
import copy
import math
import numpy as np

from scipy import ndimage as ndi
from scipy.ndimage import label, binary_closing, binary_dilation
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.morphology import disk, skeletonize as skeletonize_skimage
from skimage.transform import probabilistic_hough_line
from skimage.util import pad, crop as crop_skimage

from ..models.utils import Line, Point
from ..models.segments import Rect, Panel, Figure, FigureRoleEnum


def convert_greyscale(img):
    """
    Wrapper around skimage `rgb2gray` used for backward compatilibity
    :param np.ndarray img: input image
    :return np.ndarrat: image in grayscale
    """
    return rgb2gray(img)


def crop(img, left=None, right=None, top=None, bottom=None):
    """
    Crop image.

    Automatically limits the crop if bounds are outside the image.

    :param numpy.ndarray img: Input image.
    :param int left: Left crop.
    :param int right: Right crop.
    :param int top: Top crop.
    :param int bottom: Bottom crop.
    :return: Cropped image.
    :rtype: numpy.ndarray
    """

    height, width = img.shape[:2]

    left = max(0, left if left else 0)
    right = min(width, right if right else width)
    top = max(0, top if top else 0)
    bottom = min(height, bottom if bottom else width)
    out_img = img[top: bottom, left: right]
    return {'img': out_img, 'rectangle': Rect(left, right, top, bottom)}


def crop_rect(img, rect_boundary):
    """
    A convenience crop function that crops an image given boundaries as a Rect object
    :param np.ndarray img: input image
    :param Rect rect_boundary: object containing boundaries of the crop
    :return: cropped image
    :rtype: np.ndarray
    """
    left, right = rect_boundary.left, rect_boundary.right
    top, bottom = rect_boundary.top, rect_boundary.bottom
    return crop(img, left, right, top, bottom)


def binary_close(fig, size=5):
    """ Joins unconnected pixel by dilation and erosion"""

    selem = disk(size)

    img = pad(fig.img, size, mode='constant')
    img = binary_closing(img, selem)
    img = crop_skimage(img, size)
    return Figure(img, raw_img=fig.raw_img)


def binary_floodfill(fig):
    """ Converts all pixels inside closed contour to 1"""
    fig.img = ndi.binary_fill_holes(fig.img)
    return fig


def pixel_ratio(fig, diag):
    """ Calculates the ratio of 'on' pixels to bounding box area for binary figure

    :param fig : Input binary Figure
    :param diag : Area to calculate pixel ratio

    :return ratio: Float detailing ('on' pixels / bounding box area)
    """
    cropped_img = crop_rect(fig.img, diag)
    cropped_img = cropped_img['img']
    ones = np.count_nonzero(cropped_img)
    all_pixels = np.size(cropped_img)
    ratio = ones / all_pixels
    return ratio


def get_bounding_box(fig):
    """ Gets the bounding box of each segment

    :param fig: Input Figure
    :returns panels: List of _panel objects
    """
    panels = []
    regions = regionprops(fig.img)
    for region in regions:
        y1, x1, y2, x2 = region.bbox
        panels.append(Panel(x1, x2, y1, y2, region.label - 1))# Sets tags to start from 0
    return set(panels)


def binary_tag(fig):
    """ Tag connected regions with pixel value of 1

    :param fig: Input Figure
    :returns fig: Connected Figure
    """
    fig = copy.deepcopy(fig)
    fig.img, no_tagged = ndi.label(fig.img)
    return fig


def label_and_get_ccs(fig):
    """
    Convenience function that tags ccs in an image and creates their Panels
    :param Figure fig: Input Figure
    :return set: set of Panels of connected components
    """
    labelled = binary_tag(fig)
    return get_bounding_box(labelled)


def erase_elements(fig, elements):
    """
    Erase elements from an image on a pixel-wise basis. if no `pixels` attribute, the function erases the whole
    region inside the bounding box. Automatically assigns roles to ccs in the new figure based on the original.
    :param Figure fig: Figure object containing binarized image
    :param iterable of panels elements: list of elements to erase from image
    :return: copy of the Figure object with elements removed
    """
    temp_fig = copy.deepcopy(fig)

    try:
        flattened = temp_fig.img.flatten()
        for element in elements:
            np.put(flattened, [pixel.row * temp_fig.img.shape[1] + pixel.col for pixel in element.pixels], 0)
        img_no_elements = flattened.reshape(temp_fig.img.shape[0], temp_fig.img.shape[1])
        temp_fig.img = img_no_elements

    except AttributeError:
        for element in elements:
            temp_fig.img[element.top:element.bottom+1, element.left:element.right+1] = 0

    new_fig = Figure(temp_fig.img, fig.raw_img)
    if hasattr(fig, 'kernel_sizes'):
        new_fig.kernel_sizes = fig.kernel_sizes
    for cc1 in new_fig.connected_components:
        for cc2 in fig.connected_components:
            if cc1 == cc2:
                cc1.role = cc2.role   # Copy roles of ccs

    return new_fig


def dilate_fragments(fig, kernel_size):
    """
    Applies binary dilation to `fig.img` using a disk-shaped structuring element of size ''kernel_sizes''.
    :param Figure fig: Processed figure
    :param int kernel_size: size of the structuring element
    :return Figure: new Figure object
    """
    selem = disk(kernel_size)

    return Figure(binary_dilation(fig.img, selem), raw_img=fig.raw_img)


def is_slope_consistent(lines):
    """
    Checks if the slope of multiple lines is the same or similar. Useful when multiple lines found when searching for
    arrows
    :param [((x1,y1), (x2,y2))] lines: iterable of pairs of coordinates
    :return: True if slope is similar amongst the lines, False otherwise
    """
    if not all(isinstance(line, Line) for line in lines):
        pairs = [[Point(*coords) for coords in pair] for pair in lines]
        lines = [Line(pair) for pair in pairs]

    if all(abs(line.slope) > 10 for line in lines):  # very high/low slope == inf
        return True
    if all([line.slope == np.inf or line.slope == -np.inf for line in lines]):
        return True
    slopes = [line.slope for line in lines if abs(line.slope) != np.inf]
    if any([line.slope == np.inf or line.slope == -np.inf for line in lines]):
        slopes = [line.slope for line in lines if abs(line.slope) != np.inf]
    avg_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    abs_tol = 0.15
    rel_tol = 0.15

    tol = abs_tol if abs(avg_slope < 1) else rel_tol * avg_slope
    if std_slope > abs(tol):
        return False

    return True


def approximate_line(point_1, point_2):
    """
    Implementation of a Bresenham's algorithm. Approximates a straight line between ``point_1`` and ``point_2`` with
    pixels. Output is a list representing pixels forming a straight line path from ``point_1`` to ``point_2``
    """

    slope = Line([point_1, point_2]).slope  # Create Line just to get slope between two points

    if not isinstance(point_1, Point) and not isinstance(point_2, Point):
        point_1 = Point(row=point_1[1], col=point_1[0])
        point_2 = Point(row=point_2[1], col=point_2[0])

    if slope is np.inf:
        ordered_points = sorted([point_1, point_2], key=lambda point: point.row)
        return Line([Point(row=row, col=point_1.col) for row in range(ordered_points[0].row, ordered_points[1].row)])

    elif abs(slope) >= 1:
        ordered_points = sorted([point_1, point_2], key=lambda point: point.row)
        return bresenham_line_y_dominant(*ordered_points, slope)

    elif abs(slope) < 1:
        ordered_points = sorted([point_1, point_2], key=lambda point: point.col)
        return bresenham_line_x_dominant(*ordered_points, slope)


def bresenham_line_x_dominant(point_1, point_2, slope):
    """
    bresenham algorithm implementation when change in x is larger than change in y

    :param Point point_1: one endpoint of a line
    :param Point point_2: other endpoint of a line
    :param float slope: pre-calculated slope of the line
    :return: Line formed between the two points
    """
    y1 = point_1.row
    y2 = point_2.row
    deltay = y2 - y1
    domain = range(point_1.col, point_2.col+1)

    deltaerr = abs(slope)
    error = 0
    y = point_1.row
    line = []
    for x in domain:
        line.append((x, y))
        error += deltaerr
        if error >= 0.5:
            deltay_sign = int(math.copysign(1, deltay))
            y += deltay_sign
            error -= 1
    pixels = [Point(row=y, col=x) for x, y in line]

    return Line(pixels=pixels)


def bresenham_line_y_dominant(point_1, point_2, slope):
    """bresenham algorithm implementation when change in y is larger than change in x

    :param Point point_1: one endpoint of a line
    :param Point point_2: other endpoint of a line
    :param float slope: pre-calculated slope of the line
    :return: Line formed between the two points
    """

    x1 = point_1.col
    x2 = point_2.col
    deltax = x2-x1
    domain = range(point_1.row, point_2.row + 1)

    deltaerr = abs(1/slope)
    error = 0
    x = point_1.col
    line = []
    for y in domain:
        line.append((x, y))
        error += deltaerr
        if error >= 0.5:
            deltax_sign = int(math.copysign(1, deltax))
            x += deltax_sign
            error -= 1
    pixels = [Point(row=y, col=x) for x, y in line]

    return Line(pixels=pixels)


def remove_small_fully_contained(connected_components):
    """
    Remove smaller connected components if their bounding boxes are fully enclosed within larger connected components
    :param iterable connected_components: set of all connected components
    :return: a smaller set of ccs without the enclosed ccs
    """
    enclosed_ccs = [small_cc for small_cc in connected_components if any(large_cc.contains(small_cc) for large_cc
                    in remove_connected_component(small_cc, connected_components))]
    # print(enclosed_ccs)
    refined_ccs = connected_components.difference(set(enclosed_ccs))
    return refined_ccs


def merge_rect(rect1, rect2):
    """ Merges rectangle with another, such that the bounding box enclose both

    :param Rect rect1: A rectangle
    :param Rect rect2: Another rectangle
    :return: Merged rectangle
    """

    left = min(rect1.left, rect2.left)
    right = max(rect1.right, rect2.right)
    top = min(rect1.top, rect2.top)
    bottom = max(rect1.bottom, rect2.bottom)
    return Rect(left=left, right=right, top=top, bottom=bottom)


def remove_connected_component(cc, connected_components):
    """
    Attempt to remove connected component and return the smaller set
    :param Panel cc: connected component to remove
    :param iterable connected_components: set of all connected components
    :return: smaller set of connected components
    """
    if not isinstance(connected_components, set):
        connected_components = set(copy.deepcopy(connected_components))
    connected_components.remove(cc)
    return connected_components


def isolate_patches(fig, to_isolate):
    """
    Creates an empty np.ndarray of shape `fig.img.shape` and populates it with pixels from `to_isolate`
    :param Figure|Crop fig: Figure object with binarized image
    :param iterable of Panels to_isolate: a set or a list of connected components to isolate
    :return: np.ndarray of shape `fig.img.shape` populated with only the isolated components
    """
    isolated = np.zeros(shape=fig.img.shape)

    for connected_component in to_isolate:
        top = connected_component.top
        bottom = connected_component.bottom
        left = connected_component.left
        right = connected_component.right
        isolated[top:bottom, left:right] = fig.img[top:bottom, left:right]

    fig = Figure(img=isolated, raw_img=fig.raw_img, )
    return fig


def postprocessing_close_merge(fig, to_close):
    """
    Isolate a set of connected components and close them using a small kernel.
    Find new, larger connected components. Used for dense images, where appropriate
    closing cannot be performed initially.
    :param Figure fig: Figure object with binarized image
    :param iterable of Panels to_close: a set or list of connected components to close
    :return: A smaller set of larger connected components
    """
    isolated = isolate_patches(fig, to_close)
    closed = binary_close(isolated, size=5)
    labelled = binary_tag(closed)
    panels = get_bounding_box(labelled)
    return panels


def preprocessing_remove_long_lines(fig):
    """
    Remove long line separators from an image to improve image closing algorithm
    :param Figure fig: Figure with a binarized img attribute
    :return: Figure without separators
    """
    fig = copy.deepcopy(fig)
    threshold = int(fig.diagonal//2)
    print(threshold)
    long_lines = probabilistic_hough_line(fig.img, threshold=threshold)  # Output is two endpoints per line
    labelled_img, _ = label(fig.img)
    long_lines_list = []
    for line in long_lines:
        points = [Point(row=y, col=x) for x, y in line]
        p1 = points[0]
        line_label = labelled_img[p1.row, p1.col]
        line_pixels = np.nonzero(labelled_img == line_label)
        line_pixels = list(zip(*line_pixels))
        long_lines_list.append(Line(pixels=line_pixels))

    return erase_elements(fig, long_lines_list)


def intersect_rectangles(rect1, rect2):
    """
    Forms a new Rect object in the space shared by the two rectangles. Similar to intersection operation in set theory.
    :param Rect rect1: any Rect object
    :param Rect rect2: any Rect object
    :return: Rect formed by taking intersection of the two initial rectangles
    """
    left = max(rect1.left, rect2.left)
    right = min(rect1.right, rect2.right)
    top = max(rect1.top, rect2.top)
    bottom = min(rect1.bottom, rect2.bottom)
    return Rect(left, right, top, bottom)


def clean_output(text):
    """ Remove whitespace and newline characters from input text."""
    return text.replace('\n', '')


def flatten_list(data):
    """
    Flattens multi-level iterables into a list of elements
    :param [[..]] data: multi-level iterable data structure to flatten
    :return: flattened list of all elements
    """
    if len(data) == 0:
        return data

    if isinstance(data[0], Container):
        return flatten_list(data[0]) + flatten_list(data[1:])

    return data[:1] + flatten_list(data[1:])


def normalize_image(img):
    """
    Normalise image values to fit range between 0 and 1, and ensure it can be further proceseed. Useful e.g. after
    blurring operation
    :param np.ndarray img: analysed image
    :return: np.ndarray - image with values scaled to fit inside the [0,1] range
    """
    min_val = np.min(img)
    max_val = np.max(img)
    img -= min_val
    img /= (max_val - min_val)

    return img


def standardize(data):
    """
    Standardizes data to mean 0 and standard deviation of 1
    :param np.ndarray data: array of data
    :return np.ndarray: standardized data array
    """
    if data.dtype != 'float':
        data = data.astype('float')
    feature_mean = np.mean(data, axis=0)
    feature_std = np.std(data, axis=0)
    data -= feature_mean
    data /= feature_std
    return data


def find_minima_between_peaks(data, peaks):
    """
    Find deepest minima in ``data``, one between each adjacent pair of entries in ``peaks``, where ``data`` is a 2D
    array describing kernel density estimate. Used to cut ``data`` into segments in a way that allows assigning samples
    (used to create the estimate) to specific peaks.
    :param np.ndarray data: analysed data
    :param [int, int...] peaks: indices of peaks in ``data``
    :return: np.ndarray containing the indices of local minima
    """
    pairs = zip(peaks, peaks[1:])
    minima = []
    for pair in pairs:
        start, end = pair
        min_idx = np.argmin(data[1, start:end])+start
        minima.append(min_idx)

    return minima


def is_a_single_line(fig, panel, line_length):
    """
    Checks if the connected component is a single line by checking slope consistency of lines between randomly
    selected pixels
    :return:
    """

    lines = probabilistic_hough_line(isolate_patches(fig, [panel]).img, line_length=line_length)
    if not lines:
        return False

    return is_slope_consistent(lines)


def skeletonize(fig):
    """
    A convenience function operating on Figure objects working similarly to skimage.morphology.skeletonize
    :param fig: analysed figure object
    :return: figure object with a skeletonised image
    """
    img = skeletonize_skimage(fig.img)

    return Figure(img, raw_img=fig.raw_img)


def skeletonize_area_ratio(fig, panel):
    """ Calculates the ratio of skeletonized image pixels to total number of pixels
    :param fig: Input figure
    :param panel: Original _panel object
    :return: Float : Ratio of skeletonized pixels to total area (see pixel_ratio)
    """
    skel_fig = skeletonize(fig)
    return pixel_ratio(skel_fig, panel)


def mark_tiny_ccs(fig):
    """Marks all tiny connected components

    :param Figure fig: Analysed figure"""
    [setattr(cc, 'role', FigureRoleEnum.TINY) for cc in fig.connected_components if
     cc.area < np.percentile([cc.area for cc in fig.connected_components], 4) and cc.role is None]
