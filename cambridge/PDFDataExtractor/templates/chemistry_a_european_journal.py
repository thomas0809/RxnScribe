# -*- coding: utf-8 -*-
from .methods import Methods
from chemdataextractor.doc import Paragraph
import re


class ChemistryEuropeanJournalTemplate(Methods):
    """Template for PDFs from Chemistry A European Journal"""

    def __init__(self, pdf):
        """
        :param pdf: PDF extracted in ordered textblocks with features added
        :param extraction_pattern: Unique regex pattern for section title extraction of Chemistry A European Journal
        :param metadata: Metadata extracted by default get_metadata() method, containing abstract, keywords, doi, figure caption and title
        :param footnotes: A list contains two sub lists, used by footnotes_detect to store journal name and publisher
        """

        Methods.__init__(self)
        self.pdf = pdf
        self.metadata = self.get_metadata(pdf)
        self.extraction_pattern = '^[A-Z][a-z]+( [A-Za-z]+)*\n*[A-Za-z]+( [A-Za-z]+)*$|^[A-Z][a-z]+$|^(?i)Reference(s)*(\s)*.+|^Acknowledg(e)*ment(s)*|Keywords|Conflict of Interest|Data Availability Statement|Supporting Information|Author Contributions'
        self.footnotes = [[], []]

    def author(self):
        """Temporarily taken down"""
        pass

    def reference(self):
        """
        Reference extraction for PDFs from Chemistry A European Journal
        Such extraction is conducted in 5 steps.

        1. ref_text building: Arrange whole reference text into a list where each element starts with a sequence number and filter unwanted noise.
        Because Chemistry A European Journal doesn't have actual 'Reference' title, so 'Keywords' title is used as anchor point
        2. ref_text sorting: Naturally sort the list from previous step
        3. ref_text concatenating: Join the the previous list into a single string
        4. location pairs building: Use regex pattern to locate the span of each reference
        5. indexing ref_text with location pairs: Use span from step-4 as index to to slice reference text from step-3, and store results

        :param reference: A dictionary used to store final extracted results where key is the sequence number and value is the corresponding entry
        :param ref_text: A list to store plain reference text, each element starts with a sequence number
        :param location: A list contains two sub lists, and the span of each reference entry is stored accordingly
        :param pattern: footnotes on pages where references are.
        """

        reference = {}
        ref_text = []
        location = [[], []]
        pattern = self.footnotes_detect()
        ref_status = False

        # ref_text building
        for key, value in self.pdf.items():
            text = value['text']

            if 'keyword' in text.lower():
                ref_status = True
                continue

            if ref_status == True:
                if text not in pattern:
                    # text = text.replace('\xa0', ' ')
                    if re.search('^\[\d', text):
                        ref_text.append(text)
                    else:
                        if len(ref_text) != 0:
                            ref_text.append(ref_text[-1] + ' ' + text)
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

        # Location pairs building
        for match in re.finditer('\[\d+\]', ref_text):
            if match.group():
                location[0].append(match.span()[1])
                location[1].append(match.span()[0])

        # Indexing ref_text with location pairs
        for ref in range(len(location[0])):
            try:
                reference[str(ref)] = ''.join(ref_text[location[0][ref]:location[1][ref + 1]].replace('\n', ''))
            except IndexError:
                reference[str(ref)] = ''.join(ref_text[location[1][-1]:].replace('\n', ''))
        return reference

    def keywords(self):
        """
        :param result: A list to store results
        :param identifier: coordinates of 'Keywords' textblock, used to find actual keywords
        """
        result = []
        identifier = 0

        for key, value in self.pdf.items():
            if 'keyword' in value['text'].lower():
                identifier = value['position_y']

        for key, value in self.pdf.items():
            if identifier[1] <= value['position_y'][1] <= identifier[1]:
                result.append(value['text'].replace('\n', ' '))

        return result

    def footnotes_detect(self):
        """
        Get footnotes from pages where Keywords and References are.

        :param pages: A list to store page numbers
        :param footnotes: A list contains two sub lists, used by footnotes_detect to store journal name and publisher
        """
        keyword_status = False
        pages = []

        for key, value in self.pdf.items():
            if 'Keyword' in value['text']:
                keyword_status = True

            # Get publisher
            if keyword_status == True:
                text = value['text']
                if 'wiley-vch ' in text.lower():
                    self.footnotes[0].append(text)

                # Get page number
                page_number = re.search('^\d+$', text)
                if page_number:
                    pages.append(page_number.group())

                # Get journal name
                publisher = re.search('^Chem\. Eur\. J.+\d$', text)
                if publisher:
                    self.footnotes[1].append(publisher.group())

        self.footnotes[0] = self.most_frequent(self.footnotes[0])
        self.footnotes[1] = self.most_frequent(self.footnotes[1])

        return self.footnotes + pages

    def section(self):
        """Extract section title and corresponding text"""
        return self.get_section(self.pdf, self.extraction_pattern, pub='wiley-vch gmbh')

    def test(self):
        print('PDF returned successfully')

    def plaintext(self):
        return self.get_puretext(self.pdf)

    def journal(self, info_type=None):
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
            if key[0] == 0:
                text = re.search('^Chem\. Eur\. J\. \d+.+\d$', value['text'])
                if text:
                    text = text.group().split(',')
                    journal['name'] = 'Chemistry—A European Journal'
                    journal['year'] = re.search('\d+', text[0]).group()
                    journal['volume'] = text[1]
                    journal['page'] = text[2]

        if info_type == None:
            return journal
        else:
            return journal[info_type]

    def doi(self):
        return self.metadata['doi']

    def title(self):
        """
        :param identifier: used to select the textblock with the largest font size, kept updated until
        largest font size is obtained.
        """
        identifier = 0

        for key, value in self.pdf.items():
            if key[0] == 0:  # Check first page
                if value['font']['font_size_max'] > 13 and value['number_of_word'] > 3:
                    if value['obj_mid'] >= (value['page_x'] * 0.8) and value['obj_mid'] <= (
                            value['page_x'] * 1.2):  # check if textblock is centred
                        if len(value['font']['max_out_of_mixed']) > 1:  # Chars with max font size
                            if value['font']['font_size_max'] > identifier:
                                identifier = value['font']['font_size_max']
                                self.output_result['title'] = value['font']['max_out_of_mixed']

        return self.output_result['title']

    def abstract(self, chem=False):
        """
        Abstract in Chemistry A European Journal is seperated into two columns, essentially, two textblocks.
        First step is to find the text block that contains word 'abstract', and then add and assign its text and
        coordinates to results and identifier. Second step is to add the next textblock by comparing their relative
        positions.

        :param result: A list to store results
        :param identifier: coordinate of the fisrt text block
        """
        results = []
        identifier = 0

        for key, value in self.pdf.items():
            if key[0] == 0:  # First page
                if 'abstract' in value['text'].lower():
                    # result.append(value['text'].replace('\n',''))
                    identifier = value['position_y'][1]

                if identifier != 0:
                    if identifier * 0.85 < value['position_y'][1] < identifier * 1.25:
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
