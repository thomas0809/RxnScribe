# -*- coding: utf-8 -*-
from .methods import Methods
from chemdataextractor.doc import Paragraph
import re


class RoyalSocietyChemistryTemplate(Methods):
    """Template for PDFs from Royal Society of Chemistry"""

    def __init__(self, pdf):
        """
        :param pdf: PDF extracted in ordered textblocks with features added
        :param extraction_pattern: Unique regex pattern for section title extraction of Royal Society of Chemistry
        :param metadata: Metadata extracted by default get_metadata() method, containing abstract, keywords, doi, figure caption and title
        """
        Methods.__init__(self)
        self.pdf = pdf
        self.extraction_pattern = '^[A-Z][a-z]+( [A-Za-z]+)*\n*[A-Za-z]+( [A-Za-z]+)*$'
        self.metadata = self.get_metadata(pdf)

    def test(self):
        """Check if template is returned by extraction function"""
        print('PDF returned successfully')

    def author(self):
        """Temporarily taken down"""
        pass

    def reference(self):
        """
        Reference extraction for PDFs from Royal Society of Chemistry.
        Such extraction is conducted in 5 steps.

        1. ref_text building: Arrange whole reference text into a list where each element starts with a sequence number and filter unwanted noise
        2. ref_text sorting: Naturally sort the list from previous step
        3. ref_text concatenating: Join the the previous list into a single string
        4. location pairs building: Use regex pattern to locate the span of each reference
        5. indexing ref_text with location pairs: Use span from step-4 as index to to slice reference text from step-3, and store results

        :param reference: A dictionary used to store final extracted results where key is the sequence number and value is the corresponding entry
        :param ref_text: A list to store plain reference text, each element starts with a sequence number
        :param location: A list contains two sub lists, and the span of each reference entry is stored accordingly
        :param dic: PDF extracted and stored in JSON, where key is the section title and value is the corresponding text
        :param footnote: footnote of the page, used to filter out noise
        """

        reference = {}
        ref_text = []
        location = [[], []]
        dic = self.section()
        footnote = self.journal('year') + ', ' + self.journal('volume') + ', ' + self.journal('page')

        # ref_text building
        for key, value in dic.items():
            if re.search('refer', str(key), re.IGNORECASE):  # check if it is reference text block
                for seq, ref in enumerate(value):
                    if 'The Royal Society of Chemistry' in ref or footnote in ref:
                        continue  # filter unwanted text/noise
                    else:
                        if re.search('^\d+ ', ref):
                            ref_text.append(ref)
                        else:
                            if len(ref) != 0 and seq != 0:  # concatenate current reference text to previous one and delete itself
                                ref_text.append(ref_text[-1] + ' ' + ref)
                                ref_text.remove(ref_text[-2])

        # ref_text sorting
        def tryint(s):
            try:
                return int(s)
            except ValueError:
                return s

        def alphanum_key(s):
            return [tryint(x) for x in re.split('([0-9]+)', s)]

        def natural_sort(l):
            l.sort(key=alphanum_key)

        natural_sort(ref_text)

        # ref_text concatenating
        ref_text = ' '.join(ref_text)

        # location pairs building
        for match in re.finditer('\d+ [A-Z]', ref_text):
            if match.group():
                location[0].append(match.span()[1] - 1)
                location[1].append(match.span()[0])

        # indexing ref_text with location pairs
        for ref in range(len(location[0])):
            try:
                reference[str(ref)] = ''.join(ref_text[location[0][ref]:location[1][ref + 1]].replace('\n', ''))
            except IndexError:
                reference[str(ref)] = ''.join(ref_text[location[1][-1]:].replace('\n', ''))
        return reference

    def section(self):
        """Extract section title and corresponding text"""
        return self.get_section(self.pdf, self.extraction_pattern, pub='rsc')

    def plaintext(self):
        return self.get_puretext(self.pdf)

    def journal(self, info_type=None, journal_status=False):
        """
        Extract journal information, info_type including jounal name, year, volume and page

        :param info_type: user-defined argument used to select jounal name, year, volume or page
        """
        journal = {'name': '',
                   'year': '',
                   'volume': '',
                   'page': ''
                   }

        for key, value in self.pdf.items():
            if value['universal_sequence'] == 1:  # check if current textblock is the first one on this page
                journal['name'] = value['text']

            pattern = re.search('Cite this', value['text'], re.IGNORECASE)
            if pattern and journal_status == False:
                jounal_text = re.findall('\d+', value['text'][pattern.span()[1]:].strip('\n'))
                journal['year'] = jounal_text[0]
                journal['volume'] = jounal_text[1]
                journal['page'] = jounal_text[2]
                journal_status = True

        if info_type == None:
            return journal
        else:
            return journal[info_type]

    def doi(self):
        return self.metadata['doi']

    def title(self):
        return self.metadata['title']

    def keywords(self):
        return self.metadata['keywords']

    def abstract(self, chem=False):
        """
        Every textblock that span across the page is firstly selected and then unwanted textblocks are filtered
        based on average font sizes

        :param results: A list used to store results
        :param font_size: A lsit used to store font size of each text block
        :param target_size: font size of textblocks that are part of 'abstract'
        """
        results = []
        font_size = []

        # Get textblocks that span across the page
        for key, value in self.pdf.items():
            if key[0] == 0:  # First page
                if value['horizontal'] > value['page_x']:  # span horizontally acroos the middle of the page
                    font_size.append(round(value['font']['font_size_ave'], 3))

        # Get font size of 'abstract' text blocks
        target_size = self.most_frequent(font_size)

        # Filter unwanted textblocks
        for key, value in self.pdf.items():
            if key[0] == 0:  # First page
                rounded_size = round(value['font']['font_size_ave'], 3)
                if rounded_size >= target_size * 0.98 and rounded_size <= target_size * 1.02:
                    results.append(value['text'].replace('\n', ''))

        abstract = results

        if not chem:
            return abstract
        else:
            return Paragraph(abstract)

    def caption(self, nicely=False):
        if nicely == True:
            for seq, caption in self.metadata['figure'].items():
                print(seq)
                print(caption)
                print('\n')
        return self.metadata['figure']
