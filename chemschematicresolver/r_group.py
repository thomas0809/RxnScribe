# -*- coding: utf-8 -*-
"""
R-Group
=======

Scripts for identifying R-Group structure diagrams

author: Ed Beard
email: ejb207@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import osra_rgroup
import cirpy
import itertools
import os

from . import io
from . import actions
from .model import RGroup
from .ocr import ASSIGNMENT, SEPERATORS, CONCENTRATION

import re
from skimage.util import pad
from urllib.error import URLError

from chemdataextractor.doc.text import Token

log = logging.getLogger(__name__)

BLACKLIST_CHARS = ASSIGNMENT + SEPERATORS + CONCENTRATION

# Regular Expressions
NUMERIC_REGEX = re.compile('^\d{1,3}$')
ALPHANUMERIC_REGEX = re.compile('^((d-)?(\d{1,2}[A-Za-z]{1,2}[′″‴‶‷⁗]?)(-d))|(\d{1,3})?$')

# Commonly occuring tokens for R-Groups:
r_group_indicators = ['R', 'X', 'Y', 'Z', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'Y2', 'D', "R'", "R''", "R'''", "R''''"]
r_group_indicators = r_group_indicators + [val.lower() for val in r_group_indicators]

# Standard path to superatom dictionary file
parent_dir = os.path.dirname(os.path.abspath(__file__))
superatom_file = os.path.join(parent_dir, 'dict', 'superatom.txt')
spelling_file = os.path.join(parent_dir, 'dict', 'spelling.txt')


def detect_r_group(diag):
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

        r_groups = resolve_r_group_grid(sentences)
        r_groups_list = separate_duplicate_r_groups(r_groups)
        for r_groups in r_groups_list:
            diag.label.add_r_group_variables(convert_r_groups_to_tuples(r_groups))

    # Otherwise looks for indicative R-Group characters (=, :)
    else:

        for sentence in sentences:

            all_sentence_text = [token.text for token in sentence.tokens]

            if '=' in all_sentence_text:
                var_value_pairs = detect_r_group_from_sentence(sentence, indicator='=')
            elif ':' in all_sentence_text:
                var_value_pairs = detect_r_group_from_sentence(sentence, indicator=':')
            else:
                var_value_pairs = []

            # Process R-group values from '='
            r_groups = get_label_candidates(sentence, var_value_pairs)
            r_groups = standardize_values(r_groups)

            # Resolving positional labels where possible for 'or' cases
            r_groups = filter_repeated_labels(r_groups)

            # Separate duplicate variables into separate lists
            r_groups_list = separate_duplicate_r_groups(r_groups)

            for r_groups in r_groups_list:
                diag.label.add_r_group_variables(convert_r_groups_to_tuples(r_groups))

    return diag


def detect_r_group_from_sentence(sentence, indicator='='):
    """ Detects an R-Group from the presence of an input character

     :param sentence: A chemdataextractor.doc.text.Sentence object containing tokens to be probed for R-Groups
     :param indicator: String used to identify R-Groups

     :return var_value_pairs: A list of RGroup objects, containing the variable, value and label candidates
     :rtype: List[chemschematicresolver.model.RGroup]
     """

    var_value_pairs = []  # Used to find variable - value pairs for extraction

    for i, token in enumerate(sentence.tokens):
        if token.text is indicator:
            log.info('Found R-Group descriptor %s' % token.text)
            if i > 0:
                log.info('Variable candidate is %s' % sentence.tokens[i - 1])
            if i < len(sentence.tokens) - 1:
                log.info('Value candidate is %s' % sentence.tokens[i + 1])

            if 0 < i < len(sentence.tokens) - 1:
                variable = sentence.tokens[i - 1]
                value = sentence.tokens[i + 1]
                var_value_pairs.append(RGroup(variable, value, []))

        elif token.text == 'or' and var_value_pairs:
            log.info('"or" keyword detected. Assigning value to previous R-group variable...')

            # Identify the most recent var_value pair
            variable = var_value_pairs[-1].var
            value = sentence.tokens[i + 1]
            var_value_pairs.append(RGroup(variable, value, []))

    return var_value_pairs


def resolve_r_group_grid(sentences):
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
    log.info('R-Group table format detected. Variable candidates are %s' % variables)

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


def get_label_candidates(sentence, r_groups, blacklist_chars=BLACKLIST_CHARS, blacklist_words=['or']):
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

    r_groups = assign_label_candidates(r_groups, candidates)

    return r_groups


def assign_label_candidates(r_groups, candidates):
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
                r_group.label_candidates = [cand for cand in candidates if (start_index< cand.start < end_index) or cand.start > r_group.value.end]
            else:
                start_index = r_groups[i - 1].value.end
                end_index = r_group.var.end
                r_group.label_candidates = [cand for cand in candidates if start_index< cand.start < end_index]

        return r_groups

    else:
        for r_group in r_groups:
            var = r_group.var
            value = r_group.value
            label_cands = [candidate for candidate in candidates if candidate not in [var, value]]
            r_group.label_candidates = label_cands

        return r_groups


def filter_repeated_labels(r_groups):
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
    label_cands = filtered_r_groups[0].label_candidates # Get the label candidates for these vars (should be the same)

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


def get_rgroup_smiles(diag, extension='jpg', debug=False, superatom_path=superatom_file, spelling_path=spelling_file):
    """ Extract SMILES from a chemical diagram (powered by pyosra)

    :param diag: Input Diagram
    :param extension: String indicating format of input file
    :param debug: Bool to indicate debugging

    :return labels_and_smiles: List of Tuple(List of label candidates, SMILES) objects
    """

    # Add some padding to image to help resolve characters on the edge
    padded_img = pad(diag.fig.img, ((5, 5), (5, 5), (0, 0)), mode='constant', constant_values=1)

    # Save a temp image
    img_name = 'r_group_temp.' + extension
    io.imsave(img_name, padded_img)

    osra_input = []
    label_cands = []

    # Format the extracted rgroup
    for tokens in diag.label.r_group:
        token_dict = {}
        for token in tokens:
            token_dict[token[0].text] = token[1].text

        # Assigning var-var cases to true value if found (eg. R1=R2=H)
        for a, b in itertools.combinations(token_dict.keys(), 2):
            if token_dict[a] == b:
                token_dict[a] = token_dict[b]

        osra_input.append(token_dict)
        label_cands.append(tokens[0][2])

    # Run osra on temp image
    smiles = osra_rgroup.read_rgroup(osra_input, input_file=img_name, verbose=False, debug=debug, superatom_file=superatom_path, spelling_file=spelling_path)

    if not smiles:
        log.warning('No SMILES strings were extracted for diagram %s' % diag.tag)

    if not debug:
        io.imdel(img_name)

    smiles = [actions.clean_output(smile) for smile in smiles]

    labels_and_smiles = []
    for i, smile in enumerate(smiles):
        labels_and_smiles.append((label_cands[i], smile))

    return labels_and_smiles


def clean_chars(value, cleanchars):
    """ Remove chars for cleaning
    :param value: String to be cleaned
    :param cleanchars: Characters to remove from value

    :return value: Cleaned string
    """

    for char in cleanchars:
        value = value.replace(char, '')

    return value


def resolve_structure(compound):
    """ Resolves a compound structure using CIRPY """

    try:
        smiles = cirpy.resolve(compound, 'smiles')
        return smiles
    except URLError:
        log.warning('Cannot connect to Chemical Identify Resolver - chemical names may not be resolved.')
        return compound


def convert_r_groups_to_tuples(r_groups):
    """ Converts a list of R-Group model objects to R-Group tuples"""

    return [r_group.convert_to_tuple() for r_group in r_groups]


def standardize_values(r_groups, superatom_path=superatom_file):
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
        r_group.value = Token(r_group.value.text.replace('0', 'O'), r_group.value.start, r_group.value.end, r_group.value.lexicon)

        # Check if r_group value is in the superatom file
        exisiting_abbreviations = [line[0] for line in io.read_superatom(superatom_path)]
        if r_group.value.text not in exisiting_abbreviations:
            sub_smile = resolve_structure(r_group.value.text)

            if sub_smile is not None:
                # Add the smile to the superatom.txt dictionary for resolution in pyosra
                io.write_to_superatom(sub_smile, superatom_path)
                r_group.value = Token(sub_smile, r_group.value.start, r_group.value.end, r_group.value.lexicon)

        # Resolve commone alkyls
        # value = r_group.value.text
        # for alkyl in alkyls:
        #     if value.lower() in alkyl[1]:
        #         r_group.value = Token(alkyl[0], r_group.value.start, r_group.value.end, r_group.value.lexicon)

    return r_groups


def separate_duplicate_r_groups(r_groups):
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
