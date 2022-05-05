# -*- coding: utf-8 -*-
"""
ReactionDataExtractor
===================

Automatically extract data from simple chemical reaction schemes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging


__title__ = 'ReactionDataExtractor'
__version__ = '0.0.1'
__author__ = 'Damian Wilary'
__email__ = 'dmw51@cam.ac.uk'
__copyright__ = 'Copyright 2020 Damian Wilary, All rights reserved.'


# log = logging.getLogger(__name__)
# log.addHandler(logging.NullHandler())

from .extract import extract_image, extract_images
