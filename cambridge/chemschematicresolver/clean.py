# -*- coding: utf-8 -*-
"""
Image Cleanup Operations
========================

Functions to clean data for improved extraction.

author: Ed Beard
email: ejb207@cam.ac.uk

"""

import copy
import numpy as np
import warnings

from .ocr import read_label, read_diag_text


def find_repeating_unit(labels, diags, fig):
    """ Identifies 'n' labels as repeating unit identifiers.
        Removal only occurs when a label and diagram overlap

    :param labels: List of Label objects
    :param diags: List of Diagram objects
    :param fig: Input Figure
    :returns labels: List of cleaned label objects
    :returns diags: List of diagram objects (flagged as repeating)
    """

    ns = []

    for diag in diags:
        for cand in labels:
            if diag.overlaps(cand):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    repeating_units = [token for sentence in read_label(fig, cand)[0].text for token in sentence.tokens if 'n' is token.text]
                if repeating_units:
                    ns.append(cand)
                    diag.repeating = True

    labels = [label for label in labels if label not in ns]
    return labels, diags


def remove_diagram_numbers(diags, fig):
    """ Removes vertex numbers from diagrams for cleaner OSRA resolution"""

    num_bbox = []
    for diag in diags:

        diag_text = read_diag_text(fig, diag)

        # Simplify into list comprehension when working...
        for token in diag_text:
            if token.text in '123456789':
                print("Numeral successfully extracted %s" % token.text)
                num_bbox.append((diag.left + token.left, diag.left + token.right,
                                 diag.top + token.top, diag.top + token.bottom))

    # Make a cleaned copy of image to be used when resolving diagrams
    diag_fig = copy.deepcopy(fig)

    for bbox in num_bbox:
        diag_fig.img[bbox[2]:bbox[3], bbox[0]:bbox[1]] = np.ones(3)

    return diag_fig


def clean_output(text):
    """ Remove whitespace and newline characters from input text."""

    # text = text.replace(' ', '')
    return text.replace('\n', '')
