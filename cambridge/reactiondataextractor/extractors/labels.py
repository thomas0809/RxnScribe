# -*- coding: utf-8 -*-
"""
Labels
======

This module contains LabelExtractor and methods concerning label assignment, as well as other classes
for RGroup resolution

author: Damian Wilary
email: dmw51@cam.ac.uk

Code snippets for merging loops and RGroup and RGroupResolver taken from chemschematicresolver (MIT licence) by Edward
Beard (ejb207@cam.ac.uk)

"""
import csv
import itertools
import logging
from matplotlib.patches import Rectangle
import os
import re
from urllib.error import URLError

from chemdataextractor.doc.text import Token
import cirpy

from ..models.reaction import Diagram, Label
from ..models.segments import ReactionRoleEnum, FigureRoleEnum, Panel, Rect, Crop
from ..ocr import ASSIGNMENT, SEPARATORS, CONCENTRATION, read_label
from ..utils.processing import dilate_fragments
from .. import settings

log = logging.getLogger('extract.labels')

BLACKLIST_CHARS = ASSIGNMENT + SEPARATORS + CONCENTRATION

# Regular Expressions
NUMERIC_REGEX = re.compile('^\d{1,3}$')
ALPHANUMERIC_REGEX = re.compile('^((d-)?(\d{1,2}[A-Za-z]{1,2}[′″‴‶‷⁗]?)(-d))|(\d{1,3})?$')

# Commonly occuring tokens for R-Groups:
r_group_indicators = ['R', 'X', 'Y', 'Z', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'Y2', 'D', "R'",
                      "R''", "R'''", "R''''"]
r_group_indicators = r_group_indicators + [val.lower() for val in r_group_indicators]

# Standard path to superatom dictionary file
parent_dir = os.path.dirname(os.path.abspath(__file__))
superatom_file = os.path.join(parent_dir, '..', 'dict', 'superatom.txt')
spelling_file = os.path.join(parent_dir, '..', 'dict', 'spelling.txt')


class LabelExtractor:
    """This class is responsible for finding labels and assigning them to appropriate chemical diagrams. As opposed to
        other extractors, this gives diagrams with appropriately assigned labels"""

    def __init__(self, processed_fig, react_prods_structures, conditions_structures, confidence_thresh=0.5):
        self.fig = processed_fig
        self.react_prods_structures = react_prods_structures
        self.conditions_structures = conditions_structures
        self._dilated_fig = self._dilate_fig()
        self.confidence_threshold = confidence_thresh
        self._extracted = None

    def extract(self):
        """Main extraction method"""
        diagrams = [Diagram(panel=structure, label=self.find_label(structure), crop=Crop(self.fig, structure))
                    for structure in self.react_prods_structures]
        diagrams_conditions = [Diagram(panel=structure, label=None, crop=Crop(self.fig, structure))
                               for structure in self.conditions_structures]
        all_diagrams = diagrams + diagrams_conditions
        all_diagrams = self.resolve_labels(all_diagrams)
        self._extracted = all_diagrams
        return self.extracted

    @property
    def extracted(self):
        """Returns extracted objects"""
        return self._extracted

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        params = {'facecolor': (66 / 255, 93 / 255, 166 / 255),
                  'edgecolor': (6 / 255, 33 / 255, 106 / 255),
                  'alpha': 0.4}
        for diag in self._extracted:
            if diag.label:
                panel = diag.label.panel
                rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                      panel.bottom - panel.top, **params)
                ax.add_patch(rect_bbox)

    def find_label(self, structure):
        """Finds a label for each structure.

        First, looks for small connected components around a structure, and expands them by combining with nearby ccs.
        This is done first by looking at a dilated figure and then by further merging with slightly more distant ccs.
        Label candidates are then checked and normalised to give panels independent of the kernel size used in
        the dilated figure.
        :param Panel structure: panel containing a chemical structure
        :return: label associated with the chemical structure
        :rtype: Label"""
        seeds = self._find_seed_ccs(structure)
        clusters_ccs = []
        if seeds:
            for seed in seeds:
                clusters_ccs += self._expand_seed(seed)

            label_clusters = self._merge_into_clusters(clusters_ccs)
            label = self._assess_potential_labels(label_clusters, structure)
            if label:
                label.panel = label.merge_underlying_panels(self.fig)
                return label

    def check_if_plausible_label(self, panel):
        """Checks if ``panel`` is a valid label candidate

        Reads text inside ``panel``. Discards obvious false positives and panels with poor OCR outcomes.
        :param Panel panel: checked panel
        :return: label and its associated text or None if panel is not a valid candidate
        :rtype: Label|None"""
        label = Label(panel)
        text, conf = read_label(settings.main_figure[0], label)

        if not text and conf == 0:
            log.debug('Wrong label assigned - looking for a different panel')
            return None

        elif conf < self.confidence_threshold:
            log.debug('label recognition failed - recognition confidence below threshold')
            label.text = []
            return label

        else:
            label.text = text
            return label

    def resolve_labels(self, diags):
        """Resolves labels assigned to multiple diagrams. Clears labels where assignment is incorrect.

        Pools all labels, then checks them one by one. If any label is assigned to multiple diagrams, reassigns it
        to the closest diagram, clears labels from the other diagrams.
        :param [Diagram,...] diags: iterable of chemical Diagrams
        :return: mutated diagrams - with appropriate ``label``s set to None
        :rtype: list
        """
        all_labels = set([diag.label for diag in diags if diag.label])
        for label in all_labels:
            parent_diags = [diag for diag in diags if diag.label == label]

            if len(parent_diags) > 1:
                closest = min(parent_diags, key=lambda diag: diag.separation(label.panel))
                [setattr(diag, 'label', None) for diag in parent_diags if diag != closest]

        return diags

    def merge_label_horizontally(self, merge_candidates):
        """ Iteratively attempt to merge horizontally.

        :param merge_candidates: Input list of Panels to be merged
        :return merge_candidates: List of Panels after merging
        """

        done = False

        # Identifies panels within horizontal merging criteria
        while done is False:
            ordered_panels = sorted(merge_candidates, key=lambda panel: panel.area)
            merge_candidates, done = self._merge_loop_horizontal(ordered_panels)

        merge_candidates, done = self.merge_all_overlaps(merge_candidates)
        return merge_candidates

    def merge_labels_vertically(self, merge_candidates):
        """ Iteratively attempt to merge vertically.

        :param merge_candidates: Input list of Panels to be merged
        :return merge_candidates: List of Panels after merging
        """

        # Identifies panels within horizontal merging criteria
        ordered_panels = sorted(merge_candidates, key=lambda panel: panel.area)
        merge_candidates = self._merge_loop_vertical(ordered_panels)

        merge_candidates, done = self.merge_all_overlaps(merge_candidates)
        return merge_candidates

    def merge_all_overlaps(self, panels):
        """ Merges all overlapping rectangles together.

        :param panels : Input list of Panels
        :return output_panels: List of merged panels
        :return all_merged: Bool indicating whether all merges are completed
        """

        all_merged = False

        while all_merged is False:
            all_combos = list(itertools.combinations(panels, 2))
            panels, all_merged = self._get_one_to_merge(all_combos, panels)

        output_panels = self._retag_panels(panels)
        return output_panels, all_merged

    def _merge_into_clusters(self, clusters_ccs):
        label_ccs = list(set(clusters_ccs))
        clusters_merged_horizontally = self.merge_label_horizontally(label_ccs)
        clusters_merged = self.merge_labels_vertically(clusters_merged_horizontally)
        return clusters_merged

    def _assess_potential_labels(self, label_clusters, structure):
        # check if valid and filter
        potential_labels = list(filter(lambda label: label is not None,
                                       [self.check_if_plausible_label(cluster) for cluster in label_clusters]))

        # check if the label is not approximately in line with the structure (i.e. a balancing number)
        potential_labels = [label for label in potential_labels if
                            abs(label.panel.center[1] - structure.center[1]) > 0.15 * structure.height]
        clusters_underneath = [cluster for cluster in potential_labels
                               if cluster.center[1] > structure.center[1]]
        if clusters_underneath:
            return min(clusters_underneath, key=lambda cc: cc.separation(structure))
        elif potential_labels:
            return min(potential_labels, key=lambda cc: cc.separation(structure))
        else:
            return None

    def _dilate_fig(self):
        """Dilates the processed figure to reduce running time for subsequent panel merges"""
        ksize = min(list(self.fig.kernel_sizes.values()))
        dilated = dilate_fragments(self.fig, ksize)
        structures = self.react_prods_structures + self.conditions_structures
        for cc in dilated.connected_components:
            if any(cc.contains(structure) for structure in structures):
                cc.role = ReactionRoleEnum.GENERIC_STRUCTURE_DIAGRAM

        return dilated

    def _find_seed_ccs(self, structure):
        """Finds the closest connected components that are a potential label or their parts"""
        non_structures = [cc for cc in self.fig.connected_components
                          if cc.role not in
                          [FigureRoleEnum.STRUCTUREBACKBONE, FigureRoleEnum.STRUCTUREAUXILIARY, FigureRoleEnum.TINY]]
        cutoff = max([structure.width, structure.height]) * 1.25
        close_ccs = sorted(non_structures, key=lambda cc: structure.separation(cc))
        all_structures = self.conditions_structures + self.react_prods_structures
        close_ccs = [cc for cc in close_ccs if not any([s.contains(cc) and s != structure for s in all_structures ])]
        close_ccs = [cc for cc in close_ccs if structure.separation(cc) < cutoff]
        return close_ccs

    def _expand_seed(self, seed):
        """Looks at the dilated panels and chooses those which contain the seeds"""
        char_cluster = [cc for cc in self._dilated_fig.connected_components
                        if cc.contains(seed) and cc.role != ReactionRoleEnum.GENERIC_STRUCTURE_DIAGRAM]
        return char_cluster

    def _merge_loop_horizontal(self, panels,):
        """ Iteratively merges panels by relative proximity to each other along the x axis."""

        output_panels = []
        blacklisted_panels = []
        done = True

        for a, b in itertools.combinations(panels, 2):

            # Check panels lie in roughly the same line, that they are of label size and similar height
            if abs(a.center[1] - b.center[1]) < 1.5 * a.height \
                    and abs(a.height - b.height) < min(a.height, b.height):

                # Check that the distance between the edges of panels is not too large
                if (0 <= a.left - b.right < (min(a.height, b.height) * 1.5)) or (
                        0 <= (b.left - a.right) < (min(a.height, b.height) * 1.5)):
                    merged_rect = self._merge_rect(a, b)
                    merged_panel = Panel(merged_rect.left, merged_rect.right, merged_rect.top, merged_rect.bottom, 0)
                    output_panels.append(merged_panel)
                    blacklisted_panels.extend([a, b])
                    done = False

        log.debug('Length of blacklisted : %s' % len(blacklisted_panels))
        log.debug('Length of output panels : %s' % len(output_panels))

        for panel in panels:
            if panel not in blacklisted_panels:
                output_panels.append(panel)

        output_panels = self._retag_panels(output_panels)

        return output_panels, done

    def _merge_loop_vertical(self, panels):
        """ Iteratively merges panels by relative proximity to each other along the y axis."""

        output_panels = []
        blacklisted_panels = []

        # Merging labels that are in close proximity vertically
        for a, b in itertools.combinations(panels, 2):

            if (abs(a.left - b.left) <  min(a.width, b.width) or abs(a.center[0] - b.center[0]) <  min(a.width,
                                                                                                               b.width)) \
                    and abs(a.center[1] - b.center[1]) < 3 * min(a.height, b.height) \
                    and min(abs(a.top - b.bottom), abs(b.top - a.bottom)) < 1.75 * min(a.height, b.height):
                merged_rect = self._merge_rect(a, b)
                merged_panel = Panel(merged_rect.left, merged_rect.right, merged_rect.top, merged_rect.bottom, 0)
                output_panels.append(merged_panel)
                blacklisted_panels.extend([a, b])

        for panel in panels:
            if panel not in blacklisted_panels:
                output_panels.append(panel)

        output_panels = self._retag_panels(output_panels)

        return output_panels

    def _get_one_to_merge(self, all_combos, panels):
        """Merges the first overlapping set of panels found and an returns updated _panel list"""

        for a, b in all_combos:

            overlap_panel = self._merge_overlap(a, b)
            if overlap_panel is not None:
                merged_panel = Panel(overlap_panel.left, overlap_panel.right, overlap_panel.top, overlap_panel.bottom,
                                     0)
                panels.remove(a)
                panels.remove(b)
                panels.append(merged_panel)
                return panels, False

        return panels, True

    def _merge_overlap(self, a, b):
        """ Checks whether panels a and b overlap. If they do, returns new merged _panel"""

        if a.overlaps(b) or b.overlaps(a):
            return self._merge_rect(a, b)

    def _merge_rect(self, rect1, rect2):
        """ Merges rectangle with another, such that the bounding box enclose both"""

        left = min(rect1.left, rect2.left)
        right = max(rect1.right, rect2.right)
        top = min(rect1.top, rect2.top)
        bottom = max(rect1.bottom, rect2.bottom)
        return Rect(left, right, top, bottom)

    def _retag_panels(self, panels):
        """ Re-tag panels.

        :param panels: List of Panel objects
        :returns: List of re-tagged Panel objects
        """

        for i, panel in enumerate(panels):
            panel.tag = i
        return panels


class RGroupResolver:
    """This class is used for reading diagram labels and recognising R-groups"""

    def __init__(self, diagrams):
        self.diagrams = diagrams

    def analyse_labels(self):
        for diag in self.diagrams:
            if diag.label and diag.label.text:
                self.detect_r_group(diag)


    def detect_r_group(self, diag):
        """ Determines whether a label represents an R-Group structure, and if so gives the variable and value.

        :param diag: Diagram object to search for R-Group indicators
        :return diag: Diagram object with R-Group variable and value candidates assigned.
        """

        sentences = diag.label.text
        first_sentence_tokens = [token.text.replace(' ', '').replace('\n', '') for token in sentences[0].tokens]

        if sentences == []:
            pass
        # # Identifies grid labels from the presence of only variable tokens in the first line
        elif all([True if token in r_group_indicators else False for token in first_sentence_tokens]):

            r_groups = self._resolve_r_group_grid(sentences)
            r_groups_list = self._separate_duplicate_r_groups(r_groups)
            for r_groups in r_groups_list:
                diag.label.add_r_group_variables(RGroupResolver._convert_r_groups_to_tuples(r_groups))

        # Otherwise looks for indicative R-Group characters (=, :)
        else:

            for sentence in sentences:

                all_sentence_text = [token.text for token in sentence.tokens]

                if '=' in all_sentence_text:
                    var_value_pairs = self._detect_r_group_from_sentence(sentence, indicator='=')
                elif ':' in all_sentence_text:
                    var_value_pairs = self._detect_r_group_from_sentence(sentence, indicator=':')
                else:
                    var_value_pairs = []

                # Process R-group values from '='
                r_groups = RGroupResolver._get_label_candidates(sentence, var_value_pairs)
                r_groups = RGroupResolver._standardize_values(r_groups)

                # Resolving positional labels where possible for 'or' cases
                r_groups = RGroupResolver._filter_repeated_labels(r_groups)

                # Separate duplicate variables into separate lists
                r_groups_list = RGroupResolver._separate_duplicate_r_groups(r_groups)

                for r_groups in r_groups_list:
                    diag.label.add_r_group_variables(RGroupResolver._convert_r_groups_to_tuples(r_groups))

        return diag

    @staticmethod
    def _resolve_r_group_grid(sentences):
        """Resolves the special grid case, where data is organised into label-value columns for a specific variable.

            Please note that this only extracts simple tables, where the column indicators are contained in the list of
            r_group_indicators

        :param sentences: A chemdataextractor.doc.text.Sentence objects containing tokens to be probed for R-Groups
        :return var_value_pairs: A list of RGroup objects, containing the variable, value and label candidates
        :rtype: List[chemschematicresolver.model.RGroup]
        """

        var_value_pairs = []  # Used to find variable - value pairs for extraction
        table_identifier, table_rows = sentences[0], sentences[1:]

        variables = table_identifier.tokens
        log.debug('R-Group table format detected. Variable candidates are %s' % variables)

        # Check that the length of all table rows is the same as the table_identifier + 1
        correct_row_lengths = [True for row in table_rows if len(row.tokens) == len(variables) + 1]
        if not all(correct_row_lengths):
            return []

        for row in table_rows:
            tokens = row.tokens
            label_candidates = [tokens[0]]
            values = tokens[1:]
            for i, value in enumerate(values):
                var_value_pairs.append(RGroup(variables[i], value, label_candidates))

        return var_value_pairs

    @staticmethod
    def _standardize_values(r_groups, superatom_path=superatom_file):
        """ Converts values to a format compatible with diagram extraction"""

        # List of tuples pairing multiple definitions to the appropriate SMILES string
        alkyls = [('CH', ['methyl']),
                  ('C2H', ['ethyl']),
                  ('C3H', ['propyl']),
                  ('C4H', ['butyl']),
                  ('C5H', ['pentyl']),
                  ('C6H', ['hexyl']),
                  ('C7H', ['heptyl']),
                  ('C8H', ['octyl']),
                  ('C9H', ['nonyl']),
                  ('C1OH', ['decyl'])]

        for r_group in r_groups:
            # Convert 0's in value field to O
            r_group.value = Token(r_group.value.text.replace('0', 'O'), r_group.value.start, r_group.value.end,
                                  r_group.value.lexicon)

            # Check if r_group value is in the superatom file
            exisiting_abbreviations = [line[0] for line in RGroupResolver._read_superatom(superatom_path)]
            if r_group.value.text not in exisiting_abbreviations:
                sub_smile = RGroupResolver._resolve_structure(r_group.value.text)

                if sub_smile is not None:
                    # Add the smile to the superatom.txt dictionary for resolution in pyosra
                    RGroupResolver._write_to_superatom(sub_smile, superatom_path)
                    r_group.value = Token(sub_smile, r_group.value.start, r_group.value.end, r_group.value.lexicon)

            # Resolve commone alkyls
            # value = r_group.value.text
            # for alkyl in alkyls:
            #     if value.lower() in alkyl[1]:
            #         r_group.value = Token(alkyl[0], r_group.value.start, r_group.value.end, r_group.value.lexicon)

        return r_groups

    @staticmethod
    def _detect_r_group_from_sentence(sentence, indicator='='):
        """ Detects an R-Group from the presence of an input character

         :param sentence: A chemdataextractor.doc.text.Sentence object containing tokens to be probed for R-Groups
         :param indicator: String used to identify R-Groups

         :return var_value_pairs: A list of RGroup objects, containing the variable, value and label candidates
         :rtype: List[chemschematicresolver.model.RGroup]
         """

        var_value_pairs = []  # Used to find variable - value pairs for extraction

        for i, token in enumerate(sentence.tokens):
            if token.text is indicator:
                log.debug('Found R-Group descriptor %s' % token.text)
                if i > 0:
                    log.debug('Variable candidate is %s' % sentence.tokens[i - 1])
                if i < len(sentence.tokens) - 1:
                    log.debug('Value candidate is %s' % sentence.tokens[i + 1])

                if 0 < i < len(sentence.tokens) - 1:
                    variable = sentence.tokens[i - 1]
                    value = sentence.tokens[i + 1]
                    var_value_pairs.append(RGroup(variable, value, []))

            elif token.text == 'or' and var_value_pairs:
                log.debug('"or" keyword detected. Assigning value to previous R-group variable...')

                # Identify the most recent var_value pair
                variable = var_value_pairs[-1].var
                value = sentence.tokens[i + 1]
                var_value_pairs.append(RGroup(variable, value, []))

        return var_value_pairs

    @staticmethod
    def _convert_r_groups_to_tuples( r_groups):
        """ Converts a list of R-Group model objects to R-Group tuples"""

        return [r_group.convert_to_tuple() for r_group in r_groups]

    @staticmethod
    def _get_label_candidates(sentence, r_groups, blacklist_chars=BLACKLIST_CHARS, blacklist_words=['or']):
        """Assign label candidates from a sentence that contains known R-Group variables

        :param sentence: Sentence to probe for label candidates
        :param: r_groups: A list of R-Group objects with variable-value pairs assigned
        :param blacklist_chars: String of disallowed characters
        :param blacklist_words: List of disallowed words

        :return r_groups: List of R-Group objects with assigned label candidates
        """

        # Remove irrelevant characters and blacklisted words
        candidates = [token for token in sentence.tokens if token.text not in blacklist_chars]
        candidates = [token for token in candidates if token.text not in blacklist_words]

        r_group_vars_and_values = []
        for r_group in r_groups:
            r_group_vars_and_values.append(r_group.var)
            r_group_vars_and_values.append(r_group.value)

        candidates = [token for token in candidates if token not in r_group_vars_and_values]

        r_groups = RGroupResolver._assign_label_candidates(r_groups, candidates)

        return r_groups

    @staticmethod
    def _assign_label_candidates(r_groups, candidates):
        """ Gets label candidates for cases where the same variable appears twice in one sentence.
            This is typically indicative of cases where 2 R-Groups are defined on the same line
        """

        # Check - are there repeated variables?
        var_text = [r_group.var.text for r_group in r_groups]
        duplicate_r_groups = [r_group for r_group in r_groups if var_text.count(r_group.var.text) > 1]

        # Check that ALL r_group values have this duplicity (ie has every r_group got a duplicate variable?)
        if len(duplicate_r_groups) == len(r_groups) and len(r_groups) != 0:

            # Now go through r_groups getting positions of tokens
            for i, r_group in enumerate(r_groups):
                if i == 0:
                    end_index = r_group.var.end
                    r_group.label_candidates = [cand for cand in candidates if cand.start < end_index]
                elif i == len(r_groups) - 1:
                    start_index = r_groups[i - 1].value.end
                    end_index = r_group.var.end
                    r_group.label_candidates = [cand for cand in candidates if (
                                start_index < cand.start < end_index) or cand.start > r_group.value.end]
                else:
                    start_index = r_groups[i - 1].value.end
                    end_index = r_group.var.end
                    r_group.label_candidates = [cand for cand in candidates if start_index < cand.start < end_index]

            return r_groups

        else:
            for r_group in r_groups:
                var = r_group.var
                value = r_group.value
                label_cands = [candidate for candidate in candidates if candidate not in [var, value]]
                r_group.label_candidates = label_cands

            return r_groups

    @staticmethod
    def _separate_duplicate_r_groups(r_groups):
        """
         Separate duplicate R-group variables into separate lists

         :param r_groups: List of input R-Group objects to be tested for duplicates
         :return output: List of R-Groups with duplicates separated
        """

        if len(r_groups) is 0:
            return r_groups

        # Getting only the variables with unique text value
        vars = [r_group.var for r_group in r_groups]
        vars_text = [var.text for var in vars]
        unique_vars, unique_vars_text = [], []
        for i, var in enumerate(vars):
            if vars_text[i] not in unique_vars_text:
                unique_vars.append(var)
                unique_vars_text.append(vars_text[i])

        var_quantity_tuples = []
        vars_dict = {}
        output = []

        for var in unique_vars:
            var_quantity_tuples.append((var, vars_text.count(var.text)))
            vars_dict[var.text] = []

        equal_length = all(elem[1] == var_quantity_tuples[0][1] for elem in var_quantity_tuples)

        # If irregular, default behaviour is to just use one of the values
        if not equal_length:
            return [r_groups]

        # Populate dictionary for each unique variable
        for var in unique_vars:
            for r_group in r_groups:
                if var.text == r_group.var.text:
                    vars_dict[var.text].append(r_group)

        for i in range(len(vars_dict[var.text])):
            temp = []
            for var in unique_vars:
                try:
                    temp.append(vars_dict[var.text][i])
                except Exception as e:
                    log.error("An error occurred while attempting to separate duplicate r-groups")
                    log.error(e)
            output.append(temp)

        # Ensure that each complete set contains all label candidates
        for r_groups_output in output:
            total_cands = []
            for r_group in r_groups_output:
                for cand in r_group.label_candidates:
                    total_cands.append(cand)

            for r_group in r_groups_output:
                r_group.label_candidates = total_cands

        return output

    @staticmethod
    def _filter_repeated_labels(r_groups):
        """
         Detects repeated variable values.
         When found, this is determined to be an 'or' case so relative label assignment ensues.

         :param r_groups: Input list of R-Group objects
         :return output_r_groups: R-Group objects corrected for 'or' statements

         """

        or_vars = []
        vars = [r_group.var for r_group in r_groups]
        unique_vars = set(vars)
        for test_var in unique_vars:
            if vars.count(test_var) > 1:
                log.debug('Identified "or" variable')
                or_vars.append(test_var)

        # Get label candidates for r_groups containing this:
        filtered_r_groups = [r_group for r_group in r_groups if r_group.var in or_vars]

        # If no duplicate r_group variables, exit function
        if len(filtered_r_groups) == 0:
            return r_groups

        remaining_r_groups = [r_group for r_group in r_groups if r_group.var not in or_vars]
        label_cands = filtered_r_groups[
            0].label_candidates  # Get the label candidates for these vars (should be the same)

        # Prioritizing alphanumerics for relative label assignment
        alphanumeric_labels = [label for label in label_cands if ALPHANUMERIC_REGEX.match(label.text)]

        output_r_groups = []

        # First check if the normal number of labels is the same
        if len(filtered_r_groups) == len(label_cands):
            for i in range(len(filtered_r_groups)):
                altered_r_group = filtered_r_groups[i]
                altered_r_group.label_candidates = [label_cands[i]]
                output_r_groups.append(altered_r_group)
            output_r_groups = output_r_groups + remaining_r_groups

        # Otherwise, check if alphanumerics match
        elif len(filtered_r_groups) == len(alphanumeric_labels):
            for i in range(len(filtered_r_groups)):
                altered_r_group = filtered_r_groups[i]
                altered_r_group.label_candidates = [alphanumeric_labels[i]]
                output_r_groups.append(altered_r_group)
            output_r_groups = output_r_groups + remaining_r_groups

        # Otherwise return with all labels
        else:
            output_r_groups = r_groups

        return output_r_groups

    @staticmethod
    def _resolve_structure(compound):
        """ Resolves a compound structure using CIRPY """

        try:
            smiles = cirpy.resolve(compound, 'smiles')
            return smiles
        except URLError:
            log.warning('Cannot connect to Chemical Identify Resolver - chemical names may not be resolved.')
            return compound

    @staticmethod
    def _read_superatom(superatom_path):
        """
        Reads the superatom file as a list of tuples
        :param superatom_path: The path to the file containng superatom info
        :return: list of abbreviation-smile tuples for superatoms
        """

        with open(superatom_path, 'r') as inf:
            cleaned_lines = [' '.join(line.split()) for line in inf if not line.startswith('#')]
            cleaned_lines = [line for line in cleaned_lines if len(line) != 0]
            lines = [(line.split(' ')[0], line.split(' ')[1]) for line in cleaned_lines]

        return lines

    @staticmethod
    def _write_to_superatom(sub_smile, superatom_path):
        """
        Adds a smile string to the superatom.txt file, for resolution in pyosra
        :param sub_smile: The smile string to be added to the file
        :param: superatom_path: The path to the file containng superatom info
        """

        lines = RGroupResolver._read_superatom(superatom_path)

        if (sub_smile, sub_smile) not in lines:
            lines.append((sub_smile, sub_smile))
            with open(superatom_path, 'w') as outf:
                csvwriter = csv.writer(outf, delimiter=' ')
                csvwriter.writerows(lines)


class RGroup(object):
    """ Object containing all extracted information for an R-group result"""

    def __init__(self, var, value, label_candidates):
        self.var = var
        self.value = value
        self.label_candidates = label_candidates

    def __repr__(self):
        return '%s(variable=%s, value=%s, label_candidates=%s)' % (
            self.__class__.__name__, self.var, self.value, self.label_candidates
        )

    def __str__(self):
        return '%s(variable=%s, value=%s, label_candidates=%s)' % (
            self.__class__.__name__, self.var, self.value, self.label_candidates
        )

    def convert_to_tuple(self):
        """ Converts the r-group object to a usable a list of variable-value pairs and their list of candidate labels """
        tuple_r_group = (self.var, self.value, self.label_candidates)
        return tuple_r_group

