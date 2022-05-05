# -*- coding: utf-8 -*-
"""
Conditions
=======

This module contains classes and methods for extracting conditions, as well as directly related functions.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from collections import Counter
from itertools import chain
import logging
from matplotlib.patches import Rectangle
import numpy as np
import os
import re

from chemdataextractor.doc import Span
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from ..actions import find_nearby_ccs, extend_line
from ..models import Conditions, SolidArrow, BaseExtractor, Figure, TextLine, Crop, FigureRoleEnum, ReactionRoleEnum, Panel
from ..models.utils import Point, Line, DisabledNegativeIndices
from ..ocr import read_conditions
from ..utils.processing import find_minima_between_peaks, erase_elements
from .. import settings

log = logging.getLogger('extract.conditions')

SPECIES_FILE = os.path.join(settings.ROOT_DIR, 'dict', 'species.txt')


class ConditionsExtractor(BaseExtractor):
    """Main class for extracting reaction conditions from images

    :param arrows: All arrows in a figure
    :type arrows: list[SolidArrow]
    :param fig: main figure
    :type fig: Figure"""

    def __init__(self, arrows, fig=None):
        self.fig = fig if fig is not None else settings.main_figure[0]
        self.arrows = arrows
        self._extracted = None

    def extract(self):
        """Main extraction method"""
        conditions, conditions_structures = [], []
        for arrow in self.arrows:
            step_conditions, step_structures = self.get_conditions(arrow)
            conditions += [step_conditions]
            conditions_structures.extend(step_structures)
        self._extracted = conditions, conditions_structures
        return self.extracted

    @property
    def extracted(self):
        """Returns extracted objects"""
        return self._extracted

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        conditions, conditions_structures = self._extracted
        params = {'facecolor': 'g', 'edgecolor': None, 'alpha': 0.3}
        for panel in conditions_structures:
            rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                  panel.bottom - panel.top, **params)
            ax.add_patch(rect_bbox)

        for step_conditions in conditions:
            for t in step_conditions.text_lines:
                panel = t.panel
                rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                      panel.bottom - panel.top, **params)
                ax.add_patch(rect_bbox)

    def get_conditions(self, arrow):
        """
        Recovers conditions of a single reaction step.

        Marks text lines and chemical structures in the conditions region. Passes text through an OCR engine, and parses
        the output. Forms a Conditions object containing all the collected information.
        :param SolidArrow arrow: Reaction arrow around which the search for conditions is performed
        :return Conditions: Conditions object containing found information.
        """
        textlines, condition_structures = self.find_step_conditions(arrow)
        [setattr(panel, 'role', ReactionRoleEnum.CONDITIONS) for panel in condition_structures]

        if textlines:
            recognised = [read_conditions(self.fig, line, conf_threshold=40) for line in textlines]

            recognised = [sentence for sentence in recognised if sentence]
            parser = ConditionParser(recognised)
            conditions_dct = parser.parse_conditions()
        else:
            conditions_dct = {}
        return Conditions(textlines, conditions_dct, arrow, condition_structures), condition_structures

    def find_step_conditions(self, arrow):
        """
        Finds conditions of a step. Selects a region around an arrow. If the region contains text, scans the text.
        Otherwise it returns None (no conditions found).
        :param Arrow arrow: Arrow around which the conditions are to be looked for
        :return: Collection [Textline,...] containing characters grouped together as text lines
        """

        structure_panels = [cc.parent_panel for cc in self.fig.connected_components if
                            cc.role == FigureRoleEnum.STRUCTUREBACKBONE
                            and cc.parent_panel]
        conditions_panels = [panel for panel in structure_panels if ConditionsExtractor.belongs_to_conditions(panel,
                                                                                                              arrow)]

        text_lines = self.mark_text_lines(arrow, conditions_panels)

        for text_line in text_lines:
            self.collect_characters(text_line)
        text_lines = [text_line for text_line in text_lines if text_line.connected_components]

        return text_lines, conditions_panels

    def mark_text_lines(self, arrow, conditions_panels):
        """
        Isolates conditions around around ``arrow`` in ``fig``.

        Marks text lines first by finding obvious conditions' text characters around an arrow.
        This scan is also performed around `conditions_panels` if any. Using the found ccs, text lines are fitted
        with kernel density estimates.
        :param SolidArrow arrow: arrow around which the region of interest is centered
        :param [Panel,...] conditions_panels: iterable of panels containing connected components representing conditions
        :return: Crop: Figure-like object containing the relevant crop with the arrow removed
        """
        fig = self.fig
        average_height = np.median([cc.height for cc in fig.connected_components])

        areas = [cc.area for cc in fig.connected_components]
        areas.sort()
        def condition1(cc): return cc.role != FigureRoleEnum.STRUCTUREAUXILIARY
        if arrow.is_vertical:
            def condition2(cc): return cc.top > arrow.top and cc.bottom < arrow.bottom
        else:
            def condition2(cc): return cc.left > arrow.left and cc.right < arrow.right

        condition = condition1 and condition2
        middle_pixel = arrow.center_px
        def distance_fn(cc): return 2.2 * cc.height
        core_ccs = find_nearby_ccs(middle_pixel, fig.connected_components, (3 * average_height, distance_fn),
                                   condition=condition)
        if not core_ccs:
            for pixel in arrow.pixels[::10]:
                core_ccs = find_nearby_ccs(pixel, fig.connected_components, (2 * average_height, distance_fn),
                                           condition=condition)
                if len(core_ccs) > 1:
                    break
            else:
                log.warning('No conditions were found in the initial scan. Aborting conditions search...')
                return []

        if conditions_panels:
            for panel in conditions_panels:
                core_ccs += find_nearby_ccs(panel, fig.connected_components, (3 * average_height, distance_fn),
                                            condition=condition)

        conditions_region = Panel.create_megarect(core_ccs)

        cropped_region = Crop(erase_elements(fig, conditions_panels), conditions_region)  # Do not look at structures

        text_lines = [TextLine(None, None, top, bottom, crop=cropped_region, anchor=anchor) for (top, bottom, anchor) in
                      self.identify_text_lines(cropped_region)]

        text_lines = [text_line.in_main_figure for text_line in text_lines]

        return text_lines

    def identify_text_lines(self, crop):
        """Fits text lines of conditions text using kernel density estimation.

        Fits kernel density estimate to bottom boundaries of the relevant panels. Bottom text lines are found as the
        maxima of the estimate subject to a condition that the text lines must be separated by appropriate distance.
        The estimate is then chopped into region based on the deepest minima between peaks and characters assigned to
        these regions. Groups of characters are then used to estimate the top boundary of each text line. Each text line
        is finally associated with an anchor - one of its characters - to situate it in the main image.
        :param Crop crop: cropped region of interest containing the reaction conditions
        :return: iterable of tuples (top boundary, bottom boundary, anchor)
        :rtype: list
        """
        ccs = [cc for cc in crop.connected_components if cc.role != FigureRoleEnum.ARROW]  # filter out arrows

        if len(ccs) == 1:  # Special case
            only_cc = ccs[0]
            anchor = Point(only_cc.center[1], only_cc.center[0])
            return [(only_cc.top, only_cc.bottom, anchor)]
        if len(ccs) > 10:
            ccs = [cc for cc in ccs if
                   cc.area > np.percentile([cc.area for cc in ccs], 0.2)]  # filter out all small ccs (e.g. dots)

        img = crop.img
        bottom_boundaries = [cc.bottom for cc in ccs]
        bottom_boundaries.sort()

        bottom_count = Counter(bottom_boundaries)
        bottom_boundaries = np.array([item for item in bottom_count.elements()]).reshape(-1, 1)

        little_data = len(ccs) < 10
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(0.005, 2.0, 100)},
                            cv=(len(bottom_boundaries) if little_data else 10))  # 10-fold cross-validation
        grid.fit(bottom_boundaries)
        best_bw = grid.best_params_['bandwidth']
        kde = KernelDensity(bandwidth=best_bw, kernel='exponential')
        kde.fit(bottom_boundaries)
        # print(f'params: {kde.get_params()}')
        rows = np.linspace(0, img.shape[0] + 20, img.shape[0] + 21)
        logp_bottom = kde.score_samples(rows.reshape(-1, 1))

        heights = [cc.bottom - cc.top for cc in ccs]
        mean_height = np.mean(heights, dtype=np.uint32)
        bottom_lines, _ = find_peaks(logp_bottom, distance=mean_height * 1.2)
        data = np.array([rows, logp_bottom])
        bottom_lines.sort()

        bucket_limits = find_minima_between_peaks(data, bottom_lines)
        buckets = np.split(rows, bucket_limits)
        bucketed_chars = [[cc for cc in ccs if cc.bottom in bucket] for bucket in buckets]
        top_lines = [np.mean([cc.top for cc in bucket], dtype=int) for bucket in bucketed_chars]
        anchors = [sorted([cc for cc in bucket], key=lambda cc: cc.area)[-1].center for bucket in bucketed_chars]
        anchors = [Point(row=anchor[1], col=anchor[0]) for anchor in anchors]

        return [line for line in zip(top_lines, bottom_lines, anchors)]

    def collect_characters(self, text_line):
        """
        Accurately assigns relevant characters in ``fig`` to ``text_line``

        Uses a proximity search algorithm to carefully assign characters to each text line. Characters are assigned
        based on mutual distance as well as horizontal displacements from the middle of text line and from the
        bottom of the line and panel height.
        :param TextLine text_line: found text line object
        :return: None (mutates connected components assigned to a text line)
        :rtype: None
        """
        relevant_ccs = [cc for cc in self.fig.connected_components if cc.role != FigureRoleEnum.ARROW]
        initial_distance = np.sqrt(np.mean([cc.area for cc in relevant_ccs]))
        distance_fn = settings.DISTANCE_FN_CHARS

        def proximity_coeff(cc): return .75 if cc.area < np.percentile([cc.area for cc in relevant_ccs], 65) else .4
        def condition1(cc): return (
                    abs(text_line.panel.center[1] - cc.center[1]) < proximity_coeff(cc) * text_line.panel.height)

        def condition2(cc): return cc.height < text_line.panel.height * 1.7
        def condition3(cc): return abs(text_line.panel.bottom - cc.bottom) < 0.65 * text_line.panel.height
        def condition(cc): return condition1(cc) and condition2(cc) and condition3(cc)
        # First condition is proximity of panel center to center of text line measured vertically.
        # Second is that height is comparable to text_line.
        # Third is that the base of each letter is close to the bottom text line

        found_ccs = find_nearby_ccs(text_line.anchor, relevant_ccs, (initial_distance, distance_fn),
                                    FigureRoleEnum.CONDITIONSCHAR, condition)
        if found_ccs:
            text_line.connected_components = found_ccs

    def add_diags_to_dicts(self, diags):
        """Adds SMILES representations of diagrams that had been assigned to conditions regions

        :param [Diagram,...] diags: iterable of extracted diagrams
        :return: None (mutates the conditions dictionary)
        :rtype: None"""
        conditions, _ = self.extracted
        for step_conditions in conditions:
            if step_conditions.structure_panels:
                cond_diags = [diag for diag in diags if diag.panel in step_conditions.structure_panels]
                step_conditions.diags = cond_diags
                try:
                    step_conditions.conditions_dct['other species'].extend(
                        [diag.smiles for diag in cond_diags if diag.smiles])
                except KeyError:
                    step_conditions.conditions_dct['other species'] = [diag.smiles for diag in cond_diags if
                                                                       diag.smiles]

    @staticmethod
    def belongs_to_conditions(structure_panel, arrow):
        """
        Checks if a structure is part of the conditions

        Looks if the ``structure_panel`` center lies close to a line parallel to arrow.
        Two points equidistant to the arrow are chosen and the distance from these is compared to two extreme
        points of an arrow. If the centre is closer to either of the two points
        (subject to a maximum threshold distance) than to either of the extremes, the structure is deemed to be
        part of the conditions region.

        :param Panel structure_panel: Panel object marking a structure (superatoms included)
        :param Arrow arrow: Arrow defining the conditions region
        :return: bool True if within the conditions region else close
        """

        pixels = arrow.pixels
        react_endpoint = pixels[0]
        prod_endpoint = pixels[-1]
        midpoint = pixels[len(pixels) // 2]
        parallel_line_dummy = Line([midpoint])

        slope = arrow.line.slope
        parallel_line_dummy.slope = -1 / slope if abs(slope) > 0.05 else np.inf
        parallel_1, parallel_2 = extend_line(parallel_line_dummy,
                                             extension=react_endpoint.separation(prod_endpoint) // 2)

        closest = min([parallel_1, parallel_2, react_endpoint, prod_endpoint],
                      key=lambda point: structure_panel.separation(point))

        if closest in [parallel_1, parallel_2] and structure_panel.separation(arrow.panel) < 1.0 * np.sqrt(
                structure_panel.area):
            return True
        else:
            return False


class ConditionParser:
    """
    This class is used to parse conditions text. It is composed of several methods to facilitate parsing recognised text
    using formal grammars.

    The following strings define formal grammars to detect catalysts (cat) and coreactants (co) based on their units.
    Species which fulfill neither criterion can be parsed as `other_chemicals`. `default_values` is also defined to help 
    parse both integers and floating-point values.

    :param sentences: Sentence object retrieved from an OCR engine.
    :type sentences: chemdataextractor.Sentence
    """
    default_values = r'((?:\d\.)?\d{1,3})'
    cat_units = r'(mol\s?%|M|wt\s?%)'
    # co_units = r'(eq\.?(?:uiv(?:alents?)?\.?)?|m?L)'
    co_units = r'(equivalents?|equiv\.?|eq\.?|m?L)'

    def __init__(self, sentences):

        self.sentences = sentences  # sentences are ChemDataExtractor Sentence objects

    def parse_conditions(self):
        parse_fns = [ConditionParser._parse_coreactants, ConditionParser._parse_catalysis,
                     ConditionParser._parse_other_species, ConditionParser._parse_other_conditions]
        conditions_dct = {'catalysts': None, 'coreactants': None, 'other species': None, 'temperature': None,
                          'pressure': None, 'time': None, 'yield': None}

        coreactants_lst = []
        catalysis_lst = []
        other_species_lst = []
        for sentence in self.sentences:
            parsed = [parse(sentence) for parse in parse_fns]

            coreactants_lst.extend(parsed[0])
            catalysis_lst.extend(parsed[1])
            other_species_lst.extend(ConditionParser._filter_species(parsed))
            conditions_dct.update(parsed[3])

        conditions_dct['coreactants'] = coreactants_lst
        conditions_dct['catalysts'] = catalysis_lst
        conditions_dct['other species'] = other_species_lst
        return conditions_dct

    @staticmethod
    def _identify_species(sentence):

        with open(SPECIES_FILE, 'r') as file:
            species_list = file.read().strip().split('\n')

        # letters between which some lowercase letters and digits are allowed, optional brackets
        formulae_brackets = r'((?:[A-Z]*\d?[a-z]\d?)\((?:[A-Z]*\d?[a-z]?\d?)*\)?\d?[A-Z]*[a-z]*\d?)*'
        formulae_bracketless = r'(?<!°)\b(?<!\)|\()((?:[A-Z]+\d?[a-z]?\d?)+)(?!\(|\))\b'
        letter_upper_identifiers = r'((?<!°)\b[A-Z]{1,4}\b)(?!\)|\.)'  # Up to four capital letters? Just a single one?
        letter_lower_identifiers = r'(\b[a-z]\b)(?!\)|\.)'  # Accept single lowercase letter subject to restrictions

        number_identifiers = r'(?:^| )(?<!\w)([1-9])(?!\w)(?!\))(?:$|[, ])(?![A-Za-z])'
        # number_identifiers matches the following:
        # 1, 2, 3, three numbers as chemical identifiers
        # CH3OH, 5, 6 (5 equiv) 5 and 6 in the middle only
        # 5 5 equiv  first 5 only
        # A 5 equiv -no matches
        entity_mentions_brackets = re.finditer(formulae_brackets, sentence.text)
        entity_mentions_bracketless = re.finditer(formulae_bracketless, sentence.text)
        entity_mentions_letters_upper = re.finditer(letter_upper_identifiers, sentence.text)
        entity_mentions_letters_lower = re.finditer(letter_lower_identifiers, sentence.text)

        entity_mentions_numbers = re.finditer(number_identifiers, sentence.text)

        spans = [Span(e.group(1), e.start(), e.end()) for e in
                 chain(entity_mentions_brackets, entity_mentions_bracketless,
                       entity_mentions_numbers, entity_mentions_letters_upper,
                       entity_mentions_letters_lower) if e.group(1)]
        slashed_names = []
        for token in sentence.tokens:
            if '/' in token.text:
                slashed_names.append(token)

        all_mentions = ConditionParser._resolve_spans(spans+slashed_names)
        # Add species from the list, treat them as seeds - allow more complex names
        # eg. based on 'pentanol' on the list, allow '1-pentanol'
        species_from_list = [token for token in sentence.tokens
                             if any(species in token.text.lower() for species in species_list if species)]  # except ''
        all_mentions += species_from_list
        return list(set(all_mentions))

    @staticmethod
    def _parse_coreactants(sentence):
        co_values = ConditionParser.default_values
        co_str = co_values + r'\s?' + ConditionParser.co_units

        return ConditionParser._find_closest_cem(sentence, co_str)

    @staticmethod
    def _parse_catalysis(sentence):
        cat_values = ConditionParser.default_values
        cat_str = cat_values + r'\s?' + ConditionParser.cat_units

        return ConditionParser._find_closest_cem(sentence, cat_str)

    @staticmethod
    def _parse_other_species(sentence):
        cems = ConditionParser._identify_species(sentence)
        return [cem.text for cem in cems]

    @staticmethod
    def _parse_other_conditions(sentence):
        other_dct = {}
        parsed = [ConditionParser._parse_temperature(sentence), ConditionParser._parse_time(sentence),
                  ConditionParser._parse_pressure(sentence), ConditionParser._parse_yield(sentence)]

        temperature, time, pressure, yield_ = parsed
        if temperature:
            other_dct['temperature'] = temperature  # Create the key only if temperature was parsed
        if time:
            other_dct['time'] = time
        if pressure:
            other_dct['pressure'] = pressure
        if yield_:
            other_dct['yield'] = yield_

        return other_dct

    @staticmethod
    def _find_closest_cem(sentence, parse_str):
        """Assign closest chemical species to found units (e.g. 'mol%' or 'eq')"""
        phrase = sentence.text
        matches = []
        cwt = ChemWordTokenizer()
        bracketed_units_pat = re.compile(r'\(\s*'+parse_str+r'\s*\)')
        bracketed_units = re.findall(bracketed_units_pat, sentence.text)
        if bracketed_units:   # remove brackets
            phrase = re.sub(bracketed_units_pat, ' '.join(bracketed_units[0]), phrase)
        for match in re.finditer(parse_str, phrase):
            match_tokens = cwt.tokenize(match.group(0))
            phrase_tokens = cwt.tokenize(phrase)
            match_start_idx = [idx for idx, token in enumerate(phrase_tokens) if match_tokens[0] in token][0]
            match_end_idx = [idx for idx, token in enumerate(phrase_tokens) if match_tokens[-1] in token][0]
            # To simplify syntax above, introduce a new tokeniser that splits full stops more consistently
            # Accept two tokens, strip commas and full stops, especially if one of the tokens
            species = DisabledNegativeIndices(phrase_tokens)[match_start_idx-2:match_start_idx]
            species = ' '.join(token for token in species).strip('()., ')
            if not species:
                try:
                    species = DisabledNegativeIndices(phrase_tokens)[match_end_idx+1:match_start_idx+4]
                    # filter special signs and digits
                    species = map(lambda s: s.strip('., '), species)
                    species = filter(lambda token: token.isalpha(), species)
                    species = ' '.join(token for token in species)
                except IndexError:
                    log.debug('Closest CEM not found for a catalyst/coreactant key phrase')
                    species = ''

            if species:
                matches.append({'Species': species, 'Value': float(match.group(1)), 'Units': match.group(2)})

        return matches

    @staticmethod
    def _filter_species(parsed):
        """ If a chemical species has been assigned as both catalyst or coreactant, and `other species`, remove if from
        the latter. Also remove special cases"""
        coreactants, catalysts, other_species, _ = parsed
        combined = [d['Species'] for d in coreactants] + [d['Species'] for d in catalysts]
        # if not coreactants or catalysts found, return unchanged
        if not combined:
            return other_species

        else:
            unaccounted = []
            combined = ' '.join(combined)
            for species in other_species:
                found = re.search(re.escape(species), combined)  # include individual tokens for multi-token names
                if not found and species != 'M':
                    unaccounted.append(species)
            return list(set(unaccounted))

    @staticmethod
    def _resolve_spans(spans):
        span_copy = spans.copy()
        # spans is ~10-15 elements long at most
        for span1 in spans:
            for span2 in spans:
                if span1.text != span2.text:
                    if span1.text in span2.text:
                        try:
                            span_copy.remove(span1)
                        except ValueError:
                            pass
                    elif span2.text in span1.text:
                        try:
                            span_copy.remove(span2)
                        except ValueError:
                            pass

        return span_copy

    @staticmethod
    def _parse_time(sentence):  # add conditions to add the parsed data
        t_values = ConditionParser.default_values
        t_units = r'(h(?:ours?)?|m(?:in)?|s(?:econds)?|days?)'
        time_str = re.compile(r'(?<!\w)' + t_values + r'\s?' + t_units + r'(?=$|\s?,)')
        time = re.search(time_str, sentence.text)
        if time:
            return {'Value': float(time.group(1)), 'Units': time.group(2)}

    @staticmethod
    def _parse_temperature(sentence):
        # The following formals grammars for temperature and pressure are quite complex, but allow to parse additional
        # generic descriptors like 'heat' or 'UHV' in `.group(1)'
        t_units = r'\s?(?:o|O|0|°)C|K'   # match 0C, oC and similar, as well as K

        t_value1 = r'-?\d{1,4}' + r'\s?(?=' + t_units + ')'  # capture numbers only if followed by units
        t_value2 = r'r\.?\s?t\.?'
        t_value3 = r'heat|reflux|room\s?temp'

        # Add greek delta?
        t_or = re.compile('(' + '|'.join((t_value1, t_value2, t_value3)) + ')' + '(' + t_units + ')' + '?', re.I)
        temperature = re.search(t_or, sentence.text)
        return ConditionParser._form_dict_entry(temperature)

    @staticmethod
    def _form_dict_entry(match):
        if match:
            units = match.group(2) if match.group(2) else 'N/A'
            try:
                return {'Value': float(match.group(1)), 'Units': units}
            except ValueError:
                return {'Value': match.group(1), 'Units': units}   # if value is rt or heat, gram scale etc

    @staticmethod
    def _parse_pressure(sentence):
        p_units = r'(?:m|h|k|M)?Pa|m?bar|atm'   # match bar, mbar, mPa, hPa, MPa and atm

        p_values1 = r'\d{1,4}' + r'\s?(?=' + p_units + ')'  # match numbers only if followed by units
        p_values2 = r'(?:U?HV)|vacuum'

        p_or = re.compile('(' + '|'.join((p_values1, p_values2)) + ')' + '(' + p_units + ')' + '?')
        pressure = re.search(p_or, sentence.text)
        if pressure:
            units = pressure.group(2) if pressure.group(2) else 'N/A'
            return {'Value': float(pressure.group(1)), 'Units': units}

    @staticmethod
    def _parse_yield(sentence):
        y_units = r'%'   # match 0C, oC and similar, as well as K

        y_value1 = r'\d{1,2}' + r'\s?(?=' + y_units + ')'  # capture numbers only if followed by units
        y_value2 = r'gram scale'

        # Add greek delta?
        y_or = re.compile('(' + '|'.join((y_value1, y_value2)) + ')' + '(' + y_units + ')' + '?')
        y = re.search(y_or, sentence.text)
        return ConditionParser._form_dict_entry(y)


def clear_conditions_region(fig):
    """Removes connected components belonging to conditions and denoises the figure afterwards

    :param Figure fig: Analysed figure
    :return: new Figure object with conditions regions erased"""

    fig_no_cond = erase_elements(fig, [cc for cc in fig.connected_components
                                       if cc.role == FigureRoleEnum.ARROW or cc.role == FigureRoleEnum.CONDITIONSCHAR])

    area_threshold = fig.get_bounding_box().area / 30000
    # width_threshold = fig.get_bounding_box().width / 200
    noise = [panel for panel in fig_no_cond.connected_components if panel.area < area_threshold]

    return erase_elements(fig_no_cond, noise)
