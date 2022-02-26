
# -*- coding: utf-8 -*-
"""
ChemSchematicResolver
===================

Automatically extract data from schematic chemical diagrams
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging


__title__ = 'ChemSchematicResolver'
__version__ = '0.0.1'
__author__ = 'Ed Beard'
__email__ = 'ed.beard94@gmail.com'
__copyright__ = 'Copyright 2019 Ed Beard, All rights reserved.'


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from .extract import extract_image, extract_images, extract_document, extract_documents
