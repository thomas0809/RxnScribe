# -*- coding: utf-8 -*-
"""
Image processing utilities
==========================

A toolkit of image processing operations.

author: Ed Beard
email: ejb207@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import copy

from skimage.color import rgb2gray
from skimage.morphology import binary_closing, disk
from skimage.util import pad
from skimage.util import crop as crop_skimage
from skimage.morphology import skeletonize as skeletonize_skimage

from scipy import ndimage as ndi

from .model import Rect

log = logging.getLogger(__name__)


def crop(img, left=None, right=None, top=None, bottom=None):
    """Crop image.

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

    left = max(0, 0 if left is None else left )
    right = min(width, width if right is None else right)
    top = max(0, 0 if top is None else top)
    bottom = min(height, height if bottom is None else bottom)
    out_img = img[top: bottom, left: right]
    return out_img


def binarize(fig, threshold=0.85):
    """ Converts image to binary

    RGB images are converted to greyscale using :class:`skimage.color.rgb2gray` before binarizing.

    :param numpy.ndarray img: Input image
    :param float|numpy.ndarray threshold: Threshold to use.
    :return: Binary image.
    :rtype: numpy.ndarray
    """
    bin_fig = copy.deepcopy(fig)
    img = bin_fig.img

    # Skip if already binary
    if img.ndim <= 2 and img.dtype == bool:
        return img

    img = convert_greyscale(img)

    # Binarize with threshold (default of 0.85 empirically determined)
    binary = img < threshold
    bin_fig.img = binary
    return bin_fig


def binary_close(fig, size=20):
    """ Joins unconnected pixel by dilation and erosion"""
    selem = disk(size)

    fig.img = pad(fig.img, size, mode='constant')
    fig.img = binary_closing(fig.img, selem)
    fig.img = crop_skimage(fig.img, size)
    return fig


def binary_floodfill(fig):
    """ Converts all pixels inside closed contour to 1"""
    log.debug('Binary floodfill initiated...')
    fig.img = ndi.binary_fill_holes(fig.img)
    return fig


def convert_greyscale(img):
    """ Converts to greyscale if RGB"""

    # Convert to greyscale if needed
    if img.ndim == 3 and img.shape[-1] in [3, 4]:
        grey_img = rgb2gray(img)
    else:
        grey_img = img
    return grey_img


def skeletonize(fig):
    """
    Erode pixels down to skeleton of a figure's img object
    :param fig :
    :return: Figure : binarized figure
    """

    skel_fig = binarize(fig)
    skel_fig.img = skeletonize_skimage(skel_fig.img)

    return skel_fig


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
    return Rect(left, right, top, bottom)


def merge_overlap(a, b):
    """ Checks whether panels a and b overlap. If they do, returns new merged panel"""

    if a.overlaps(b) or b.overlaps(a):
        return merge_rect(a, b)
