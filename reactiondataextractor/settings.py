# -*- coding: utf-8 -*-
"""
Settings
========

This module contains some settings and universal routines used by other modules.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
import numpy as np
import os

global main_figure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def DISTANCE_FN_CHARS(cc): return 2.25 * np.max([cc.width, cc.height])

main_figure = []
