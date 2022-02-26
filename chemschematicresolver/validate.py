# -*- coding: utf-8 -*-
"""
Image Processing Validation Metrics
===================================

A toolkit of validation metrics for determining reliability of output.

author: Ed Beard
email: ejb207@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def is_false_positive(label_smile_tuple, allow_wildcards=False):
    """ Identifies failures from absence of labels and incomplete / invalid smiles

    :rtype bool
    :returns : True if result is a false positive
    """

    label_candidates, smile = label_smile_tuple[0], label_smile_tuple[1]
    # Remove results without a label
    if len(label_candidates) == 0:
        return True

    # Remove results containing the wildcard character in the SMILE
    if '*' in smile and not allow_wildcards:
        return True

    # Remove results where no SMILE was returned
    if smile == '':
        return True

    return False


def remove_repeating(diags):
    """ Removes any diagrams containing a repeating element

    :param diags: List of labelled diagrams
    :returns: diags: List of labelled diagtrams, where those with repeating structures are removed...
    """

    diags = [diag for diag in diags if not diag.repeating]
    return diags

