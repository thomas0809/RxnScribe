# -*- coding: utf-8 -*-
"""
PDF document reader.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextLine, LTTextBox, LTChar, LTAnno, LTTextLineHorizontal
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from collections import Counter, OrderedDict
from templates import *


class Reader:
    """Reader that reads in PDF files"""

    def __init__(self, showPublisher=None):
        """
        :param publishers: Publisher identifiers used to detect publisher
        :param bool publisher_state: Weather publisher is found. Default False
        :param showPublisher: A flag to control if the name of the publisher is showed
        :param universal_sequence: Sequence number assigned to each text block to show its location on Document level
        """
        self.publishers = ['elsevier', 'rsc.', 'acs', 'chem. eur. j', 'wiley-vch ', 'angewandte', 'springer']
        self.publisher_state = False
        self.showPublisher = showPublisher
        self.universal_sequence = 0

    def detect(self, fstring, fname=None):
        """Detect if file passed into the pipeline is a pdf file"""
        if fname and not fname.endswith('.pdf'):
            return False
        return True

    def font_name_size(self, current_lt_obj):
        """
        Font analysis of each text block

        :param current_lt_obj: # The pdfminer.layout.object
        :param fonts_size: A list that stores the font sizes of each character
        :param fonts_name: A list that stores the font names of each character
        :param text: A list that stores each character
        :param final: A list that stores characters that have the largest font sizes in a text block
        """

        fonts_size, fonts_name, text, final = [], [], [], []

        # for page_layout in extract_pages("test.pdf"):
        #     for element in page_layout:
        #         if isinstance(element, LTTextContainer):
        #             for text_line in element:
        #                 for character in text_line:
        #                     if isinstance(character, LTChar):
        #                         print(character.fontname)
        #                         print(character.size)

        # for o in current_lt_obj._objs:
        #     if o.get_text().strip():  # Check if the current text block is empty
        #         for char in o._objs:  # For each characters in the text block
        #             if isinstance(char, LTChar):
        #                 if char.size != 0:
        #                     fonts_size.append(char.size)
        #                     fonts_name.append(char.fontname)
        #                     text.append(char.get_text())
        #
        #             elif isinstance(char, LTAnno):
        #                 fonts_size.append(-100)  # Just to occupy a space
        #                 text.append(char.get_text())

        for o in current_lt_obj:
            if isinstance(o, LTTextLineHorizontal):
                if o.get_text().strip():  # Check if the current text block is empty
                    for char in o._objs:  # For each characters in the text block
                        if isinstance(char, LTChar):
                            if char.size != 0:
                                fonts_size.append(char.size)
                                fonts_name.append(char.fontname)
                                text.append(char.get_text())

                        elif isinstance(char, LTAnno):
                            fonts_size.append(-100)  # Just to occupy a space
                            text.append(char.get_text())
            elif isinstance(o, LTChar):
                pass
        # Replace -100 to max font size
        fonts_size = [max(fonts_size) if x == -100 else x for x in fonts_size]

        # Add characters to the list
        for i in [index for index, value in enumerate(fonts_size) if value == max(fonts_size)]:
            final.append(text[i])

        if fonts_size:
            return {'font_size_max': max(fonts_size),
                    'font_size_min': min(fonts_size),
                    'font_size_ave': sum(fonts_size) / float(len(fonts_size)),
                    'font_name_most': Counter(fonts_name).most_common(),
                    'max_out_of_mixed': ''.join(final).replace('\n', ' ').strip()
                    }

    def detect_publisher(self, input_text):
        """Detect the publisher of the input file"""
        for publisher in self.publishers:
            if publisher in input_text.lower():
                if self.showPublisher == True:
                    print('Publisher: ***', publisher, '***')
                self.publisher_state = True
                self.publishers = self.publishers[self.publishers.index(publisher)]
                return
            else:
                continue

    def span_across(self, textblock):
        return

    def single_page_layout(self, layout, page_seq, page_x, page_y):
        """
        Process an LTPage layout and return a list of elements

        :param layout: device.get_result() returned by PDFMiner
        :param page_seq: Current page number
        :param page_x: Middle point of current page on the x axis
        :param page_y: Middle point of current page on the y axis
        :param dic_page: A dictionary that stores the results, keys are page number and textblock number; Values are features generated at this step
        """

        dic_page = {}

        for lt_obj_seq, lt_obj in enumerate(layout):
            if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):


                # print(lt_obj.get_text())


                if not self.publisher_state:
                    if page_seq == 0:
                        self.detect_publisher(lt_obj.get_text())  # Pass current text to function
                    else:
                        pass

                x0, y0 = lt_obj.bbox[0], lt_obj.bbox[1]  # Coordinates of bottom left point
                x1, y1 = lt_obj.bbox[2], lt_obj.bbox[3]  # Coordinates of top right point
                horizontal = abs((x1 - x0))
                vertical = abs((y1 - y0))
                area = horizontal * vertical
                self.universal_sequence += 1

                dic_page[int(page_seq), int(lt_obj_seq)] = {  # i -> page_number, j-> obj_number on each page
                    'position_x': (x0, y0),  # Coordinates of bottom left point
                    'position_y': (x1, y1),  # Coordinates of top right point
                    'text': str(lt_obj.get_text().rstrip('\n')),  # Plain text of the current object
                    'horizontal': horizontal,  # Horizontal length
                    'vertical': vertical,  # Vertical length
                    'area': area,  # Area of current object
                    'sequence': (page_seq, lt_obj_seq),  # Page number and current object number
                    'universal_sequence': self.universal_sequence,# Universal sequence number across the whole PDF dcocument
                    'font': self.font_name_size(lt_obj),  # Font analysis of text block
                    'number_of_char': len(lt_obj.get_text().rstrip('\n')),  # Number of characters in the text block
                    'number_of_word': len(lt_obj.get_text().replace('\n', '').split(' ')),# Number of words in the text block
                    'obj_mid': ((x0 + x1) / 2),  # Centre point of text block
                    'page_x': page_x,  # Middle point of current page on the x axis
                    'page_y': page_y,  # Middle point of current page on the y axis
                    'publisher': self.publishers
                }
            else:
                pass

        return dic_page

    def PDFsetup(self):
        dic = {}
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        return device, interpreter, dic

    def dic_sorting(self, dic_unsorted):
        return OrderedDict(sorted(dic_unsorted.items()))

    def extraction(self, publisher_name, pdf, f):
        """Select specific template for input pdf"""
        # print(publisher_name)

        if publisher_name == 'elsevier':
            print('*** Elsevier detected ***')
            return ElsevierTemplate(pdf, f)

        elif publisher_name == 'acs':
            print('*** American Chemistry Society detected ***')
            return AmericanChemicalSocietyTemplate(pdf)

        elif publisher_name == 'rsc.':
            print('*** Royal Society of Chemistry detected ***')
            return RoyalSocietyChemistryTemplate(pdf)

        elif publisher_name == 'wiley-vch ':
            print('*** Advanced Materials Family detected ***')
            return AdvancedMaterialsFamilyTemplate(pdf)

        elif publisher_name == 'chem. eur. j':
            print('***Chemistry A European Journal detected ***')
            return ChemistryEuropeanJournalTemplate(pdf)

        elif publisher_name == 'angewandte':
            print('*** Angewandte detected ***')
            return AngewandteTemplate(pdf)

        # elif publisher_name == 'springer':
        #     return Springer(pdf)

        else:
            print('Publisher Not Supported at the moment')

    def read_file(self, file_name):
        """
        Read in a file and process each page using 'single_page_layout()'
        """

        print('Reading: ', file_name)

        try:
            f = open(file_name, 'rb')
            device, interpreter, dic = self.PDFsetup()

            # Loop through every page of a PDF file
            for page_seq, page in enumerate(
                    PDFPage.get_pages(f, pagenos=set(), maxpages=0, password='', caching=True, check_extractable=True)):
                interpreter.process_page(page)
                layout = device.get_result()
                page_x = (float(page.mediabox[2]) / 2)  # Middle point of current page on the x axis
                page_y = (float(page.mediabox[3]) / 2)  # Middle point of current page on the y axis

                dic.update(self.single_page_layout(layout, page_seq, page_x, page_y))  # Result for a single page

            pdf = self.dic_sorting(dic)  # Whatever the publisher is, this one stays the same

            pdf_extracted = self.extraction(self.publishers, pdf, f)
            return pdf_extracted

        except Exception as e:
            pass
