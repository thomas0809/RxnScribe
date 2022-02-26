# -*- coding: utf-8 -*-
"""
Image Processing Actions
========================

A toolkit of image processing actions for segmentation.

author: Ed Beard
email: ejb207@cam.ac.uk, ed.beard94@gmail.com

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import numpy as np
import os
from skimage.util import pad
from skimage.measure import regionprops

import itertools
import copy
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
import osra_rgroup

from .model import Panel, Diagram, Label, Rect, Figure
from .io import imsave, imdel
from .clean import find_repeating_unit, clean_output
from .utils import crop, skeletonize, binarize, binary_close, binary_floodfill, merge_rect, merge_overlap

# Standard path to superatom dictionary file
parent_dir = os.path.dirname(os.path.abspath(__file__))
superatom_file = os.path.join(parent_dir, 'dict', 'superatom.txt')
spelling_file = os.path.join(parent_dir, 'dict', 'spelling.txt')


log = logging.getLogger(__name__)


def segment(fig):
    """ Segments image.

    :param fig: Input Figure
    :return panels: List of segmented Panel objects
    """

    bin_fig = binarize(fig)

    bbox = fig.get_bounding_box()
    skel_pixel_ratio = skeletonize_area_ratio(fig, bbox)

    log.debug(" The skeletonized pixel ratio is %s" % skel_pixel_ratio)

    # Choose kernel size according to skeletonized pixel ratio
    if skel_pixel_ratio > 0.025:
        kernel = 4
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    elif 0.02 < skel_pixel_ratio <= 0.025:
        kernel = 6
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    elif 0.015 < skel_pixel_ratio <= 0.02:
        kernel = 10
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    else:
        kernel = 15
        closed_fig = binary_close(bin_fig, size=kernel)
        log.debug("Segmentation kernel size = %s" % kernel)

    # Using a binary floodfill to identify panel regions
    fill_img = binary_floodfill(closed_fig)
    tag_img = binary_tag(fill_img)
    panels = get_bounding_box(tag_img)

    # Removing relatively tiny pixel islands that are determined to be noise
    area_threshold = fig.get_bounding_box().area / 200
    width_threshold = fig.get_bounding_box().width / 150
    panels = [panel for panel in panels if panel.area > area_threshold or panel.width > width_threshold]
    return panels


def classify_kmeans(panels, fig, skel=True):
    """Takes input image and classifies through k means cluster of the panel area"""

    if len(panels) <= 1:
        raise Exception('Only one panel detected. Cannot cluster')
    return get_labels_and_diagrams_k_means_clustering(panels, fig, skel)


def get_labels_and_diagrams_k_means_clustering(panels, fig, skel=True):
    """ Splits into labels and diagrams using K-means clustering by the skeletonized area ratio or panel height.

    :param panels: List of Panel objects to be clustered
    :param fig: Input Figure
    :param skel: Boolean indication the clustering parameters to use
    :return Lists of Labels and Diagrams after clustering
    """

    cluster_params = []

    for panel in panels:
        if skel:
            cluster_params.append([skeletonize_area_ratio(fig, panel)])
        else:
            cluster_params.append([panel.height])

    all_params = np.array(cluster_params)

    km = KMeans(n_clusters=2)
    clusters = km.fit(all_params)

    group_1, group_2 = [], []

    for i, cluster in enumerate(clusters.labels_):
        if cluster == 0:
            group_1.append(panels[i])
        else:
            group_2.append(panels[i])

    if np.nanmean([panel.area for panel in group_1]) > np.nanmean([panel.area for panel in group_2]):
        diags = group_1
        labels = group_2
    else:
        diags = group_2
        labels = group_1

    # Convert to appropriate types
    labels = [Label(label.left, label.right, label.top, label.bottom, label.tag) for label in labels]
    diags = [Diagram(diag.left, diag.right, diag.top, diag.bottom, diag.tag) for diag in diags]
    return labels, diags


def preprocessing(labels, diags, fig):
    """Pre-processing steps before final K-means classification
    :param labels: List of Label objects
    :param diags: List of Diagram objects
    :param fig: Figure object
    
    :return out_labels: List of Labels after merging and re-tagging
    :return out_diags: List of Diagrams after re-tagging
    """

    # Remove repeating unit indicators
    labels, diags = find_repeating_unit(labels, diags, fig)

    # Remove small pixel islands from diagrams
    diags = remove_diag_pixel_islands(diags, fig)

    # Merge labels together that are sufficiently local
    label_candidates_horizontally_merged = merge_label_horizontally(labels,fig)
    label_candidates_fully_merged = merge_labels_vertically(label_candidates_horizontally_merged)
    labels_converted = convert_panels_to_labels(label_candidates_fully_merged)

    # Re-tagging all diagrams and labels
    retagged_panels = retag_panels(labels_converted + diags)
    out_labels = retagged_panels[:len(labels_converted)]
    out_diags = retagged_panels[len(labels_converted):]

    return out_labels, out_diags


def label_diags(labels, diags, fig_bbox):
    """ Pair all Diagrams to Labels.

    :param labels: List of Label objects
    :param diags: List of Diagram objects
    :param fig_bbox: Co-ordinates of the bounding box of the entire figure

    :returns: List of Diagrams with assigned Labels

    """

    # Sort diagrams from largest to smallest
    diags.sort(key=lambda x: x.area, reverse=True)
    initial_sorting = [assign_label_to_diag(diag, labels, fig_bbox) for diag in diags]

    # Identify failures by the presence of duplicate labels
    failed_diag_label = get_duplicate_labelling(initial_sorting)

    if len(failed_diag_label) == 0:
        return initial_sorting

    # Find average position of label relative to diagram for successful pairings (denoted with compass points: NSEW)
    successful_diag_label = [diag for diag in diags if diag not in failed_diag_label]

    # Where no sucessful pairings found, attempt looking 'South' for all diagrams (most common relative label position)
    if len(successful_diag_label) == 0:
        altered_sorting = [assign_label_to_diag_postprocessing(diag, labels, 'S', fig_bbox) for diag in failed_diag_label]
        if len(get_duplicate_labelling(altered_sorting)) != 0:
            altered_sorting = initial_sorting
            pass
        else:
            return altered_sorting
    else:
        # Get compass positions of labels relative to diagram
        diag_compass = [diag.compass_position(diag.label) for diag in successful_diag_label if diag.label]
        mode_compass = max(diag_compass, key=diag_compass.count)

        # Expand outwards in compass direction for all failures
        altered_sorting = [assign_label_to_diag_postprocessing(diag, labels, mode_compass, fig_bbox) for diag in failed_diag_label]

        # Check for duplicates after relabelling
        failed_diag_label = get_duplicate_labelling(altered_sorting + successful_diag_label)
        successful_diag_label = [diag for diag in successful_diag_label if diag not in failed_diag_label]

        # If no duplicates return all results
        if len(failed_diag_label) == 0:
            return altered_sorting + successful_diag_label

    # Add non duplicates to successes
    successful_diag_label.extend([diag for diag in altered_sorting if diag not in failed_diag_label])

    # Remove duplicate results
    diags_with_labels, diags_without_labels = remove_duplicates(failed_diag_label, fig_bbox)

    return diags_with_labels + successful_diag_label


def assign_label_to_diag(diag, labels, fig_bbox, rate=1):
    """ Iteratively expands the bounding box of diagram until it intersects a Label object

    :param diag: Input Diagram object to expand from
    :param labels: List of Label objects
    :param fig_bbox: Panel object representing the co-ordinates for the entire Figure
    :param rate: Number of pixels to expand by upon each iteration

    :return diag: Diagram with Label object assigned
    """

    probe_rect = Rect(diag.left, diag.right, diag.top, diag.bottom)
    found = False
    max_threshold_width = fig_bbox.width
    max_threshold_height = fig_bbox.height

    while found is False and (probe_rect.width < max_threshold_width or probe_rect.height < max_threshold_height):
        # Increase border value each loop
        probe_rect.right = probe_rect.right + rate
        probe_rect.bottom = probe_rect.bottom + rate
        probe_rect.left = probe_rect.left - rate
        probe_rect.top = probe_rect.top - rate

        for label in labels:
            if probe_rect.overlaps(label):
                found = True
                diag.label = label
    return diag


def assign_label_to_diag_postprocessing(diag, labels, direction, fig_bbox, rate=1):
    """ Iteratively expands the bounding box of diagram in the specified compass direction

    :param diag: Input Diagram object to expand from
    :param labels: List of Label objects
    :param direction: String representing determined compass direction (allowed values: 'E', 'S', 'W', 'N')
    :param fig_bbox: Panel object representing the co-ordinates for the entire Figure
    :param rate: Number of pixels to expand by upon each iteration
    """

    probe_rect = Rect(diag.left, diag.right, diag.top, diag.bottom)
    found = False

    def label_loop():

        for label in labels:
            # Only accepting labels in the average direction
            if diag.compass_position(label) != direction:
                pass
            elif probe_rect.overlaps(label):
                diag.label = label
                return True

        return False

    # Increase border value each loop
    if direction == 'E':
        while found is False and probe_rect.right < fig_bbox.right:
            probe_rect.right = probe_rect.right + rate
            found = label_loop()

    elif direction == 'S':
        while found is False and probe_rect.bottom < fig_bbox.bottom:
            probe_rect.bottom = probe_rect.bottom + rate
            found = label_loop()

    elif direction == 'W':
        while found is False and probe_rect.left > fig_bbox.left:
            probe_rect.left = probe_rect.left - rate
            found = label_loop()

    elif direction == 'N':
        while found is False and probe_rect.top > fig_bbox.top:
            probe_rect.top = probe_rect.top - rate
            found = label_loop()
    else:
        return diag

    return diag


def read_diagram_pyosra(diag, extension='jpg', debug=False, superatom_path=superatom_file, spelling_path=spelling_file):
    """ Converts a diagram to SMILES using pyosra

    :param diag: Diagram to be extracted
    :param extension: String file extension
    :param debug: Bool inicating debug mode

    :return smile: String of extracted chemical SMILE

    """

    # Add some padding to image to help resolve characters on the edge
    padded_img = pad(diag.fig.img, ((5, 5), (5, 5), (0, 0)), mode='constant', constant_values=1)

    # Save a temp image
    temp_img_fname = 'osra_temp.' + extension
    imsave(temp_img_fname, padded_img)

    # Run osra on temp image
    smile = osra_rgroup.read_diagram(temp_img_fname, debug=debug, superatom_file=superatom_path, spelling_file=spelling_path)

    if not smile:
        log.warning('No SMILES string was extracted for diagram %s' % diag.tag)

    if not debug:
        imdel(temp_img_fname)

    smile = clean_output(smile)
    return smile


def remove_diag_pixel_islands(diags, fig):
    """ Removes small pixel islands from the diagram

    :param diags: List of input Diagrams
    :param fig: Figure object

    :return diags: List of Diagrams with small pixel islands removed

    """

    for diag in diags:

        # Make a cleaned copy of image to be used when resolving diagrams
        clean_fig = copy.deepcopy(fig)

        diag_fig = Figure(crop(clean_fig.img, diag.left, diag.right, diag.top, diag.bottom))
        seg_fig = Figure(crop(clean_fig.img, diag.left, diag.right, diag.top, diag.bottom))
        sub_panels = segment(seg_fig)

        panel_areas = [panel.area for panel in sub_panels]
        diag_area = max(panel_areas)

        sub_panels = [panel for panel in sub_panels if panel.area != diag_area]

        sub_bbox = [(panel.left, panel.right, panel.top, panel.bottom) for panel in sub_panels]

        for bbox in sub_bbox:
            diag_fig.img[bbox[2]:bbox[3], bbox[0]:bbox[1]] = np.ones(3)

        diag.fig = diag_fig

    return diags


def pixel_ratio(fig, diag):
    """ Calculates the ratio of 'on' pixels to bounding box area for binary figure

    :param fig : Input binary Figure
    :param diag : Area to calculate pixel ratio

    :return ratio: Float detailing ('on' pixels / bounding box area)
    """

    cropped_img = crop(fig.img, diag.left, diag.right, diag.top, diag.bottom)
    ones = np.count_nonzero(cropped_img)
    all_pixels = np.size(cropped_img)
    ratio = ones / all_pixels
    return ratio


def binary_tag(fig):
    """ Tag connected regions with pixel value of 1

    :param fig: Input Figure
    :returns fig: Connected Figure
    """
    fig.img, no_tagged = ndi.label(fig.img)
    return fig


def get_bounding_box(fig):
    """ Gets the bounding box of each segment

    :param fig: Input Figure
    :returns panels: List of panel objects
    """
    panels = []
    regions = regionprops(fig.img)
    for region in regions:
        y1, x1, y2, x2 = region.bbox
        panels.append(Panel(x1, x2, y1, y2, region.label - 1))# Sets tags to start from 0
    return panels


def retag_panels(panels):
    """ Re-tag panels.

    :param panels: List of Panel objects
    :returns: List of re-tagged Panel objects
    """

    for i, panel in enumerate(panels):
        panel.tag = i
    return panels


def skeletonize_area_ratio(fig, panel):
    """ Calculates the ratio of skeletonized image pixels to total number of pixels

    :param fig: Input figure
    :param panel: Original panel object
    :return: Float : Ratio of skeletonized pixels to total area (see pixel_ratio)
    """

    skel_fig = skeletonize(fig)
    return pixel_ratio(skel_fig, panel)


def order_by_area(panels):
    """ Returns a list of panel objects ordered by area.

    :param panels: Input list of Panels
    :return panels: Output list of sorted Panels
    """

    def get_area(panel):
        return panel.area

    panels.sort(key=get_area)
    return panels


def merge_label_horizontally(merge_candidates, fig):
    """ Iteratively attempt to merge horizontally

    :param merge_candidates: Input list of Panels to be merged
    :return merge_candidates: List of Panels after merging
    """

    done = False

    # Identifies panels within horizontal merging criteria
    while done is False:
        ordered_panels = order_by_area(merge_candidates)
        merge_candidates, done = merge_loop_horizontal(ordered_panels, fig)

    merge_candidates, done = merge_all_overlaps(merge_candidates)
    return merge_candidates


def merge_labels_vertically(merge_candidates):
    """ Iteratively attempt to merge vertically

    :param merge_candidates: Input list of Panels to be merged
    :return merge_candidates: List of Panels after merging
    """

    # Identifies panels within horizontal merging criteria
    ordered_panels = order_by_area(merge_candidates)
    merge_candidates = merge_loop_vertical(ordered_panels)

    merge_candidates, done = merge_all_overlaps(merge_candidates)
    return merge_candidates


def merge_loop_horizontal(panels, fig_input):
    """ Iteratively merges panels by relative proximity to each other along the x axis.
        This is repeated until no panels are merged by the algorithm

    :param panels: List of Panels to be merged.

    :return output_panels: List of merged panels
    :return done: Bool indicating whether a merge occurred
    """

    output_panels = []
    blacklisted_panels = []
    done = True

    for a, b in itertools.combinations(panels, 2):

        # Check panels lie in roughly the same line, that they are of label size and similar height
        if abs(a.center[1] - b.center[1]) < 1.5 * a.height \
                and abs(a.height - b.height) < min(a.height, b.height):

            # Check that the distance between the edges of panels is not too large
            if (0 <= a.left - b.right < (min(a.height, b.height) * 2)) or (0 <= (b.left - a.right) < (min(a.height, b.height) * 2)):

                merged_rect = merge_rect(a, b)
                merged_panel = Panel(merged_rect.left, merged_rect.right, merged_rect.top, merged_rect.bottom, 0)
                output_panels.append(merged_panel)
                blacklisted_panels.extend([a, b])
                done = False

    log.debug('Length of blacklisted : %s' % len(blacklisted_panels))
    log.debug('Length of output panels : %s' % len(output_panels))

    for panel in panels:
        if panel not in blacklisted_panels:
            output_panels.append(panel)

    output_panels = retag_panels(output_panels)

    return output_panels, done


def merge_loop_vertical(panels):
    """ Iteratively merges panels by relative proximity to each other along the y axis.
        This is repeated until no panels are merged by the algorithm

    :param panels: List of Panels to be merged.

    :return output_panels: List of merged panels
    :return done: Bool indicating whether a merge occurred
    """

    output_panels = []
    blacklisted_panels = []

    # Merging labels that are in close proximity vertically
    for a, b in itertools.combinations(panels, 2):

        if (abs(a.left - b.left) < 3 * min(a.height, b.height) or abs(a.center[0] - b.center[0]) < 3 * min(a.height, b.height)) \
                and abs(a.center[1] - b.center[1]) < 3 * min(a.height, b.height) \
                and min(abs(a.top - b.bottom), abs(b.top - a.bottom)) < 2 * min(a.height, b.height):

            merged_rect = merge_rect(a, b)
            merged_panel = Panel(merged_rect.left, merged_rect.right, merged_rect.top, merged_rect.bottom, 0)
            output_panels.append(merged_panel)
            blacklisted_panels.extend([a, b])

    for panel in panels:
        if panel not in blacklisted_panels:
            output_panels.append(panel)

    output_panels = retag_panels(output_panels)

    return output_panels


def get_one_to_merge(all_combos, panels):
    """Merges the first overlapping set of panels found and an returns updated panel list

    :param all_combos: List of Tuple(Panel, Panel) objects of all possible combinations of the input 'panels' variable
    :param panels: List of input Panels

    :return panels: List of updated panels after one overlap is merged
    :return: Bool indicated whether all overlaps have been completed
    """

    for a, b in all_combos:

        overlap_panel = merge_overlap(a, b)
        if overlap_panel is not None:
            merged_panel = Panel(overlap_panel.left, overlap_panel.right, overlap_panel.top, overlap_panel.bottom, 0)
            panels.remove(a)
            panels.remove(b)
            panels.append(merged_panel)
            return panels, False

    return panels, True


def convert_panels_to_labels(panels):
    """ Converts a list of panels to a list of labels

    :param panels: Input list of Panels
    :return : List of Labels
    """

    return [Label(panel.left, panel.right, panel.top, panel.bottom, panel.tag) for panel in panels]


def merge_all_overlaps(panels):
    """ Merges all overlapping rectangles together

    :param panels : Input list of Panels
    :return output_panels: List of merged panels
    :return all_merged: Bool indicating whether all merges are completed
    """

    all_merged = False

    while all_merged is False:
        all_combos = list(itertools.combinations(panels, 2))
        panels, all_merged = get_one_to_merge(all_combos, panels)

    output_panels = retag_panels(panels)
    return output_panels, all_merged


def get_duplicate_labelling(labelled_diags):
    """ Returns diagrams sharing a Label object with other diagrams.

    :param labelled_diags: List of Diagrams with Label objects assigned
    :return failed_diag_label: List of Diagrams that share a Label object with another Diagram
    """

    failed_diag_label = set(diag for diag in labelled_diags if not diag.label)
    filtered_labelled_diags = [diag for diag in labelled_diags if diag not in failed_diag_label]

    # Identifying cases with the same label:
    for a, b in itertools.combinations(filtered_labelled_diags, 2):
        if a.label == b.label:
            failed_diag_label.add(a)
            failed_diag_label.add(b)

    return failed_diag_label


def remove_duplicates(diags, fig_bbox, rate=1):
    """
    Removes the least likely of the duplicate results.
    Likeliness is determined from the distance from the bounding box

    :param diags: All detected diagrams with assigned labels
    :param fig_bbox: Panel object representing the co-ordinates for the entire Figure
    :param rate: Number of pixels to expand by upon each iteration

    :return output_diags : List of Diagrams with Labels
    :return output_labelless_diags : List of Diagrams with Labels removed due to duplicates
    """

    output_diags = []
    output_labelless_diags = []

    # Unique labels
    unique_labels = set(diag.label for diag in diags if diag.label is not None)

    for label in unique_labels:

        diags_with_labels = [diag for diag in diags if diag.label is not None]
        diags_with_this_label = [diag for diag in diags_with_labels if diag.label.tag == label.tag]

        if len(diags_with_this_label) == 1:
            output_diags.append(diags_with_this_label[0])
            continue

        diag_and_displacement = [] # List of diag-distance tuples

        for diag in diags_with_this_label:

            probe_rect = Rect(diag.left, diag.right, diag.top, diag.bottom)
            found = False
            max_threshold_width = fig_bbox.width
            max_threshold_height = fig_bbox.height
            rate_counter = 0

            while found is False and (probe_rect.width < max_threshold_width or probe_rect.height < max_threshold_height):
                # Increase border value each loop
                probe_rect.right = probe_rect.right + rate
                probe_rect.bottom = probe_rect.bottom + rate
                probe_rect.left = probe_rect.left - rate
                probe_rect.top = probe_rect.top - rate

                rate_counter += rate

                if probe_rect.overlaps(label):
                    found = True
                    diag_and_displacement.append((diag, rate_counter))

        master_diag = min(diag_and_displacement, key=lambda x: x[1])[0]
        output_diags.append(master_diag)

        labelless_diags = [diag[0] for diag in diag_and_displacement if diag[0] is not master_diag]

        for diag in labelless_diags:
            diag.label = None

        output_labelless_diags.extend(labelless_diags)

    return output_diags, output_labelless_diags


