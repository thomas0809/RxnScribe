# -*- coding: utf-8 -*-
"""
Extract
=======

Main extraction routines.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
from matplotlib import pyplot as plt
import os

from .actions import estimate_single_bond
from .extractors import (ArrowExtractor, ConditionsExtractor, clear_conditions_region, DiagramExtractor, LabelExtractor,
                        RGroupResolver)
from .models.output import ReactionScheme
from .recognise import DiagramRecogniser
from . import settings
from .utils.io_ import imread
from .utils.processing import mark_tiny_ccs

MAIN_DIR = os.getcwd()

import matplotlib

log = logging.getLogger('extract')
file_handler = logging.FileHandler(os.path.join(settings.ROOT_DIR, 'extract.log'))
log.addHandler(file_handler)


def extract_image(filename, debug=False):
    """
    Extracts reaction schemes from a single file specified by ``filename``. ``debug`` enables more detailed logging and
    plotting.

    :param str filename: name of the image file
    :param bool debug: bool enabling debug mode
    :return Scheme: Reaction scheme object
    """
    level = 'DEBUG' if debug else 'INFO'
    ch = logging.StreamHandler()
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    path = os.path.join(MAIN_DIR, filename)
    log.info(f'Extraction started...')

    fig = imread(path)
    settings.main_figure.append(fig)
    fig.single_bond_length = estimate_single_bond(fig)
    mark_tiny_ccs(fig)

    arrow_extractor = ArrowExtractor()
    arrows = arrow_extractor.extract()
    log.info(f'Detected {len(arrows)} arrows')
    diag_extractor = DiagramExtractor()
    structure_panels = diag_extractor.extract()
    log.info(f'Found {len(structure_panels)} panels of chemical diagrams')
    conditions_extractor = ConditionsExtractor(arrows)
    conditions, conditions_structures = conditions_extractor.extract()
#    for step_conditions in conditions:
#        log.info(f'Conditions dictionary found: {step_conditions.conditions_dct}')

    react_prod_structures = [panel for panel in structure_panels if panel not in conditions_structures]
    fig_no_cond = clear_conditions_region(fig)

    label_extractor = LabelExtractor(fig_no_cond, react_prod_structures, conditions_structures)
    diags = label_extractor.extract()
    log.info('Label extraction process finished.')

    resolver = RGroupResolver(diags)
    resolver.analyse_labels()

    recogniser = DiagramRecogniser(diags)
    recogniser.recognise_diagrams()
    log.info('Diagrams have been optically recognised.')
    conditions_extractor.add_diags_to_dicts(diags)

    if debug:
        f = plt.figure()
        ax = f.add_axes([0, 0, 1, 1])
        ax.imshow(fig.img, cmap=plt.cm.binary)
        arrow_extractor.plot_extracted(ax)
        conditions_extractor.plot_extracted(ax)
        diag_extractor.plot_extracted(ax)
        label_extractor.plot_extracted(ax)
        ax.axis('off')
        ax.set_title('Segmented image')
        plt.show()

    scheme = ReactionScheme(conditions, diags, fig)
    log.info('Scheme completed without errors.')

    settings.main_figure = []
    return scheme


def extract_images(indir_path, out_dir_path, debug=False):
    """Performs a series of extraction and outputs the graphs converted into a JSON format

    Extracts reaction schemes from all files in ``dirname`` and saves the output in the JSON format to a ``out_dirname``
    directory
    :param str indir_path: path to the directory containing input files
    :param str out_dir_path: path to the directory, where output will be saved
    """

    for filename in os.listdir(indir_path):
        try:
            if not os.path.exists(out_dir_path):
                os.mkdir(out_dir_path)

            path = os.path.join(indir_path, filename)
            if os.path.isdir(path):
                continue
            else:
                scheme = extract_image(path)

                out_file = '.'.join(filename.split('.')[:-1])+'.json'
                out_path = os.path.join(out_dir_path, out_file)
                with open(out_path, 'w') as out_file:
                    out_file.write(scheme.to_json())

        except Exception as e:
            print(f'Extraction failed for file {filename}')
            if debug:
                print(f'Exception message: {str(e)}')
            continue

    print('all schemes extracted')
