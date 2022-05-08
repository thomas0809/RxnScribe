# -*- coding: utf-8 -*-
from .methods import Methods
from chemdataextractor.doc import Paragraph
import re

class AmericanChemicalSocietyTemplate(Methods):
    """Template for PDFs from American Chemical Society"""

    def __init__(self, pdf):
        """
        :param pdf: PDF extracted in ordered textblocks with features added
        :param extraction_pattern: Unique regex pattern for section title extraction of American Chemical Society
        :param metadata: Metadata extracted by default get_metadata() method, containing abstract, keywords, doi, figure caption and title
        """

        Methods.__init__(self)
        self.pdf = pdf
        self.extraction_pattern = r'^[^\w]\s[A-Z]+(\s[A-Z]+)*$'
        #'^\d\. [A-Z]+'
        #|^[^\w]\s[A-Z]+(\s[A-Z]+)*$|^(?i)Reference(s)*(\s)*.+|^Acknowledg(e)*ment(s)*'
        self.metadata = self.get_metadata(pdf)

    def author(self):
        """Temporarily taken down"""
        pass

    def reference(self):
        """Seperate reference part from the whole PDF"""
        return self.get_reference(self.section())

    def section(self):
        """Extract section title and corresponding text"""
        return self.get_section(self.pdf, self.extraction_pattern, pub='acs')

    def test(self):
        print('PDF returned successfully')

    def plaintext(self):
        return self.get_puretext(self.pdf)

    def journal(self, info_type=None):
        journal = {'name': '',
                   'year': '',
                   'volume': '',
                   'page': ''
                   }
        journal_status = False

        for key, value in self.pdf.items():
            pattern = re.search('(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)', value['text'])
            if pattern and journal_status == False:
                text = value['text'][pattern.span()[1]:].strip('\n')
                journal_status = True

        text_num = re.findall('\d+', text)

        journal['name'] = text
        journal['year'] = text_num[0]
        journal['volume'] = text_num[1]
        journal['page'] = text_num[2] + '-' + text_num[3]

        if info_type == None:
            return journal
        else:
            return journal[info_type]

    def doi(self):
        return self.metadata['doi']

    def title(self):
        return self.metadata['title']

    def keywords(self):
        text = self.metadata['abstract'].lower()
        if 'keywords' in text:
            target = re.search('keywords', text)
            return text[target.span()[1]:]
        else:
            return self.metadata['keywords']

    def abstract(self, chem=False):
        abstract = self.metadata['abstract']

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