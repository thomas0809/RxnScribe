# -*- coding: utf-8 -*-
"""
Classes for parsing relevant info

========
author: Ed Beard
email: ejb207@cam.ac.uk

"""

from chemdataextractor.parse.cem import BaseParser, lenient_chemical_label
from chemdataextractor.nlp.tokenize import WordTokenizer
from chemdataextractor.model import Compound


class LabelParser(BaseParser):

    root = lenient_chemical_label

    def interpret(self, result, start, end):
        for label in result.xpath('./text()'):
            yield Compound(labels=[label])


class ChemSchematicResolverTokeniser(WordTokenizer):
    """ Bespoke version of ChemDiagramExtractor's word tokenizer that doesn't split on prime characters"""

    #: Split before and after these sequences, wherever they occur, unless entire token is one of these sequences
    SPLIT = [
        ' ',  # Specific whitespace characters
        '----',
        '––––',  # \u2013 en dash
        '————',  # \u2014 em dash
        '<--->',
        '---',
        '–––',  # \u2013 en dash
        '———',  # \u2014 em dash
        '<-->',
        '-->',
        '...',
        '--',
        '––',  # \u2013 en dash
        '——',  # \u2014 em dash
        '``',
       # "''",
        '->',
        '<',
        '>',
        '–',  # \u2013 en dash
        '—',  # \u2014 em dash
        '―',  # \u2015 horizontal bar
        '~',  # \u007e Tilde
        '⁓',  # \u2053 Swung dash
        '∼',  # \u223c Tilde operator
        '°',  # \u00b0 Degrees
        ';',
        '@',
        '#',
        '$',
        '£',  # \u00a3
        '€',  # \u20ac
        '%',
        '&',
        '?',
        '!',
        '™',  # \u2122
        '®',  # \u00ae
        '…',  # \u2026
        '⋯',  # \u22ef Mid-line ellipsis
        '†',  # \u2020 Dagger
        '‡',  # \u2021 Double dagger
        '§',  # \u00a7 Section sign
        '¶'  # \u00b6 Pilcrow sign
        '≠',  # \u2260
        '≡',  # \u2261
        '≢',  # \u2262
        '≣',  # \u2263
        '≤',  # \u2264
        '≥',  # \u2265
        '≦',  # \u2266
        '≧',  # \u2267
        '≨',  # \u2268
        '≩',  # \u2269
        '≪',  # \u226a
        '≫',  # \u226b
        '≈',  # \u2248
        '=',
        '÷',  # \u00f7
        '×',  # \u00d7
        '→',  # \u2192
        '⇄',  # \u21c4
        # '"',  # \u0022 Quote mark
        # '“',  # \u201c
        # '”',  # \u201d
        '„',  # \u201e
        #'‟',  # \u201f
        # '‘',  # \u2018 Left single quote
        # '’',  # \u2019 Right single quote  - Regularly used as an apostrophe, so don't always split
        '‚',  # \u201a Single low quote
        # '‛',  # \u201b Single reversed quote
        # '`',  # \u0060
        # '´',  # \u00b4
        # Brackets
        '(',
        '[',
        '{',
        '}',
        ']',
        ')',
        '+',  # \u002b Plus
        '±',  # \u00b1 Plus/Minus
    ]

    SPLIT_START_WORD = []
    SPLIT_END_WORD = []

    # def __init__(self):
    #     super().__init__(self)

    def _subspan(self, s, span, nextspan):
        """Recursively subdivide spans based on a series of rules."""
        text = s[span[0]:span[1]]
        lowertext = text.lower()

        # Skip if only a single character or a split sequence
        if span[1] - span[
            0] < 2 or text in self.SPLIT or text in self.SPLIT_END_WORD or text in self.SPLIT_START_WORD or lowertext in self.NO_SPLIT:
            return [span]

        # Skip if it looks like URL
        if text.startswith('http://') or text.startswith('ftp://') or text.startswith('www.'):
            return [span]

        # Split full stop at end of final token (allow certain characters to follow) unless ellipsis
        if self.split_last_stop and nextspan is None and text not in self.NO_SPLIT_STOP and not text[-3:] == '...':
            if text[-1] == '.':
                return self._split_span(span, -1)

        # Split off certain sequences at the end of a word
        for spl in self.SPLIT_END_WORD:
            if text.endswith(spl) and len(text) > len(spl) and text[-len(spl) - 1].isalpha():
                return self._split_span(span, -len(spl), 0)

        # Split off certain sequences at the start of a word
        for spl in self.SPLIT_START_WORD:
            if text.startswith(spl) and len(text) > len(spl) and text[-len(spl) - 1].isalpha():
                return self._split_span(span, len(spl), 0)

        # Split around certain sequences
        for spl in self.SPLIT:
            ind = text.find(spl)
            if ind > -1:
                return self._split_span(span, ind, len(spl))

        # Split around certain sequences unless followed by a digit
        for spl in self.SPLIT_NO_DIGIT:
            ind = text.rfind(spl)
            if ind > -1 and (len(text) <= ind + len(spl) or not text[ind + len(spl)].isdigit()):
                return self._split_span(span, ind, len(spl))

        # Characters to split around, but with exceptions
        for i, char in enumerate(text):
            if char == '-':
                before = lowertext[:i]
                after = lowertext[i + 1:]
                # By default we split on hyphens
                split = True
                if before in self.NO_SPLIT_PREFIX or after in self.NO_SPLIT_SUFFIX:
                    split = False  # Don't split if prefix or suffix in list
                elif not before.strip(self.NO_SPLIT_CHARS) or not after.strip(self.NO_SPLIT_CHARS):
                    split = False  # Don't split if prefix or suffix entirely consist of certain characters
                if split:
                    return self._split_span(span, i, 1)

        # Split contraction words
        for contraction in self.CONTRACTIONS:
            if lowertext == contraction[0]:
                return self._split_span(span, contraction[1])
        return [span]

