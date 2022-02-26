# -*- coding: utf-8 -*-
"""
Processing
==========

This module contains low level I/O routines.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

import warnings

from skimage import io as skio


from ..models.segments import Figure

log = logging.getLogger('extract.io_')


def imread(filepath, bin_thresh=0.85):
    """
    This function takes in an image and returns binarised Figure object
    :param string filepath: path to a file
    :param float bin_thresh : threshold used for binarisation
    :return: Figure
    """
    raw_img = skio.imread(filepath, plugin='pil')
    log.info('Reading file - filepath: %s ' % filepath)
    if raw_img.max() == 255:
        raw_img = raw_img/255

    img = skio.imread(filepath, as_gray=True, plugin='pil')
    if img.max() == 255:
        img = img/255
    img = img < bin_thresh
    return Figure(img, raw_img=raw_img)



def imsave(f, img):
    """Save an image to file.
    :param string|file f: Filename or file-like object.
    :param numpy.ndarray img: Image to save. Of shape (M,N) or (M,N,3) or (M,N,4).
    """
    with warnings.catch_warnings(record=True) as ws:
        # Ensure we use PIL so we can guarantee that imsave will accept file-like object as well as filename
        skio.imsave(f, img, plugin='pil', quality=100)


def imdel(f):
    """ Delete an image file
    """

    os.remove(f)

# def extract_images(dirname, debug=False, allow_wildcards=False):
#     """ Extracts the chemical schematic diagrams from a directory of input images
#
#     :param dirname: Location of directory, with figures to be extracted
#     :param debug: Boolean specifying verbose debug mode.
#     :param allow_wildcards: Bool to indicate whether results containing wildcards are permitted
#
#     :return results: List of chemical record objects, enriched with chemical diagram information
#     """
#
#     log.info('Extracting all images at %s ...' % dirname)
#
#     results = []
#
#     if os.path.isdir(dirname):
#         # Extract from all files in directory
#         for file in os.listdir(dirname):
#             results.append(extract_image(os.path.join(dirname, file), debug, allow_wildcards))
#
#     elif os.path.isfile(dirname):
#
#         # Unzipping compressed inputs
#         if dirname.endswith('zip'):
#             # Logic to unzip the file locally
#             log.info('Opening zip file...')
#             zip_ref = zipfile.ZipFile(dirname)
#             extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
#             if not os.path.exists(extracted_path):
#                 os.makedirs(extracted_path)
#             zip_ref.extractall(extracted_path)
#             zip_ref.close()
#
#         elif dirname.endswith('tar.gz'):
#             # Logic to unzip tarball locally
#             log.info('Opening tarball file...')
#             tar_ref = tarfile.open(dirname, 'r:gz')
#             extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
#             if not os.path.exists(extracted_path):
#                 os.makedirs(extracted_path)
#             tar_ref.extractall(extracted_path)
#             tar_ref.close()
#
#         elif dirname.endswith('tar'):
#             # Logic to unzip tarball locally
#             log.info('Opening tarball file...')
#             tar_ref = tarfile.open(dirname, 'r:')
#             extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
#             if not os.path.exists(extracted_path):
#                 os.makedirs(extracted_path)
#             tar_ref.extractall(extracted_path)
#             tar_ref.close()
#         else:
#             # Logic for wrong file type
#             log.error('Input not a directory')
#             raise NotADirectoryError
#
#         imgs = [os.path.join(extracted_path, doc) for doc in os.listdir(extracted_path)]
#         for file in imgs:
#             results.append(extract_image(file, debug, allow_wildcards))
#
#     log.info('Results extracted sucessfully:')
#     log.info(results)
#
#     return results



