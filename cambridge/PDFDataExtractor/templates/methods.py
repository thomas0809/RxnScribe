import re


class Methods(object):
    """Generic methods for extraction """

    def __init__(self):
        """
        :param output_result: A dictionary to store metadata
        :param section_result: A dictionary to store section title and its corresponding text
        :param text: A list used by get_section() to store corresponding text under each section title
        """

        self.output_result = {'abstract': '',
                              'title': '',
                              'keywords': ''
                              }
        self.section_result = {}
        self.text = []

    def header_removal(self, dic_page):
        """Find page headers, numbers, figure captions """

        headers = []
        compare = []

        for key, value in dic_page.items():
            if int(key[1]) < 3:  # First 3 elements of one page
                if value['text'] in compare:
                    headers.append(value['text'])
                else:
                    pass
                    compare.append(value['text'])

        # re.search pattern transformation
        removals = ['^\d+$|^Fig.+|^Table.+']  # Page number, figure and table captions
        rep = {' ': '\s',
               '.': '\.',
               '(': '\(',
               ')': '\)'}  # Page title
        rep = dict((re.escape(k), v) for k, v in rep.items())  # Replace multiple substrings
        pattern = re.compile("|".join(rep.keys()))

        if headers:
            for header in list(set(headers)):
                removals.append(pattern.sub(lambda m: rep[re.escape(m.group(0))],
                                            header))  # This is actually the string of headers found

        return str('|^'.join(removals))  # Search pattern for page number, figure/table caption, and page headers

    def check_number(self, input_string):
        """Check if any number in the string"""
        return bool(re.search(r'\d', input_string))

    def check_bold(self, input_textblock):
        """Check if input string is in Bold, elsevier only atm"""
        text = input_textblock['font']['font_name_most'][0][0][-1]
        if text == 'B':
            return True
        elif text == 'd':
            return True

    def most_frequent(self, List):
        """Return the most frequent element from a list"""
        return max(set(List), key=List.count)

    def get_reference(self, dic):
        """
        Extract references from extracted sections in a three-stage manner:
        1. build text from pdf dictionary
        2. build location pairs, which indicates the span of each reference entry
        3. index reference text with location pairs

        Two styles of refernece extracted:
        1. Reference marked with a sequence number at the beginning
        2. Reference with no sequence number marking

        :param reference: A dictionary used to store final extracted results where key is the sequence number and value is the corresponding entry
        :param ref_text: A list to store plain reference text, each element starts with a sequence number
        :param location: A list contains two sub lists, and the span of each reference entry is stored accordingly
        :param ref_sequence: A list that stores the ending sequence of each reference entry
        :param ref_loc_pair: A list that stores the span of each reference entry

        """

        reference = {}
        ref_text = []
        location = [[], []]
        ref_sequence = []
        ref_loc_pair = []
        numbered_style = False

        for key, value in dic.items():
            if re.search('refer', str(key), re.IGNORECASE):  # Check whether it is reference section

                # Build text for both styles
                for seq, ref in enumerate(value):  # Check each reference
                    ref_text.append(ref)  # Add unextracted refs to ref_text list

                    if re.search('^\[\d+\]\s|^\(\d+\)\s', ref):  # Ccheck whether each ref is marked with brackets
                        bracket_style = re.search('^\[\d+\]\s|^\(\d+\)\s', ref).group()[0]
                        numbered_style = True

                    elif re.search(r'\d\.$|\)\.$', ref):  # Determine the ending sequence for each ref, non numbered only
                        ref_sequence.append(seq)

                # Reference with sequence number
                if numbered_style == True:
                    ref_text = ''.join(str(ref_text))
                    if bracket_style == '[':  # Check bracket style
                        pattern = '\[\d+\]\s'
                    else:
                        pattern = '\(\d+\)\s'

                    # Build location pairs
                    for match in re.finditer(pattern, str(ref_text)):
                        if match.group():
                            location[0].append(match.span()[1])
                            location[1].append(match.span()[0])

                    # Indexing text with location pairs
                    for ref in range(len(location[0])):
                        try:
                            reference[str(ref)] = ref_text[location[0][ref]:location[1][ref + 1]].replace('\n',
                                                                                                          '').split(',')
                        except IndexError:
                            reference[str(ref)] = ref_text[location[1][-1]:].replace('\n', '').split(',')
                    return reference

                # Reference without sequence number
                else:
                    # Build location pairs
                    for location in range(len(ref_sequence) - 1):  # Construct location pairs for each ref
                        ref_loc_pair.append((ref_sequence[location] - 1, ref_sequence[location + 1] - 1))

                    # Indexing text with location pairs
                    for ref in range(len(ref_sequence)):
                        try:
                            reference[str(ref)] = ''.join(ref_text[ref_loc_pair[ref][0]:ref_loc_pair[ref][1]])
                        except IndexError:
                            try:
                                reference[str(ref)] = ref_text[ref_loc_pair[ref - 1][1]:]
                            except IndexError:
                                pass
                    return reference

    def get_section(self, dic, extraction_pattern, pub):
        """
        Segment document main body into different sections

        :param dic: Pre-processed pdf file
        :param extraction_pattern: A regex pattern that is unique to each publisher, used to find section titles
        :param pub: name of the publisher
        :param locations: A list that stores the universal_sequence of text blocks
        :param titles: A list that stores the strings of extracted section titles
        :param sizes: A list that stores font sizes of extracted section titles
        :param location_new: Updated from 'locations' based on 'sizes'
        :param titles_new: Updated from 'titles_new' based on 'sizes'
        :param target: Noisy information including Headers, captions and page number
        :param droped_text: Headers, captions and page number are collected in this list
        """

        locations, text, titles, sizes, location_new, titles_new = [], [], [], [], [], []
        target = self.header_removal(dic)  # Headers, captions and page number get removed here from the whole text
        droped_text = []

        # Build text
        for key, value in dic.items():
            if re.search(target, value['text']):  # Dump page numbers and headers for each page
                self.text.append('')
                droped_text.append(value['universal_sequence'])
            else:
                self.text.append(value['text'].replace('\n', ' '))

        # Build indexing
        title_search = re.compile(extraction_pattern)
        for key, value in dic.items():

            if value['universal_sequence'] in droped_text:
                pass
            elif value['horizontal'] > value['page_x']:
                pass
            else:
                match = title_search.match(value['text'].replace('\n', ' ').strip())
                if match:
                    if pub == 'els':
                        if self.check_bold(value) == True:
                            locations.append(value['universal_sequence'])
                            titles.append(match.group())
                            sizes.append(value['font']['font_size_max'])

                    elif pub == 'acs':
                        locations.append(value['universal_sequence'])
                        titles.append(match.group())
                        sizes.append(value['font']['font_size_max'])

                    elif pub == 'angewandte':
                        if value['font']['font_name_most'][0][0] == 'AdvSCASFBDI':
                            pass
                        else:
                            locations.append(value['universal_sequence'])
                            titles.append(value['font']['max_out_of_mixed'])
                            sizes.append(value['font']['font_size_max'])

                    else:
                        locations.append(value['universal_sequence'])
                        titles.append(value['font']['max_out_of_mixed'])
                        sizes.append(value['font']['font_size_max'])

        # Update location and title
        for s in [size for size, value in enumerate(sizes) if
                  value >= (max(sizes) * 0.98) and value <= (max(sizes) * 1.02)]:
            location_new.append(locations[s])
            titles_new.append(titles[s])

        # print(titles_new)
        pair = [(location_new[location] - 1, location_new[location + 1] - 1) for location in
                range(len(location_new) - 1)]  # construction of location pairs
        try:
            pair.append((pair[-1][1], None))  # Add reference location to the slicing index list
        except IndexError:
            pass

        # Slice the text based on indexing
        for k in range(len(titles_new)):
            try:
                self.section_result[titles_new[k]] = self.text[int(pair[k][0]) + 1:pair[k][1]]
            except IndexError:
                self.section_result[titles_new[k]] = self.text[pair[k - 1][1]:]
        return self.section_result

    def get_puretext(self, pdf):
        pdf_body = ''
        for key, value in pdf.items():
            pdf_body += value['text'] + '\n\n'
        return pdf_body

    def get_metadata(self, pdf):
        """
        Default metadata extraction methods, including:
        abstract, keywords, doi, figure captions, title

        :param pdf: a dictionary representing the whole paper / data model
        :param bool figure_status: Whether figure caption is found. Default False
        :param bool table_status: Whether table is found. Default False
        :param bool abstract_status: Whether abstract is found. Default False
        :param bool title__status: Whether title is found. Default False
        :param abstract_location: Coordinates of the text block that contains string 'abstract'
        :param identifier: Strings of the text block that contains string 'abstract'
        """

        figure_status = False
        table_status = False
        abstract_status = False
        title__status = False
        abstract_location = 0
        identifier = ''

        for key, value in pdf.items():
            if key[0] == 0:  # Check whether it is first page

                # Abstract
                if abstract_status == False:
                    if re.match(r"A B S T R A C T|ABSTRACT", value['text'],
                                re.IGNORECASE):  # This regex could be weak, the number of space should be considered here
                        abstract_location = value['position_x'][0]  # Assign its x0 to abstract_location for later use
                        identifier = value['text']
                        if value[
                            'number_of_word'] > 50:  # It is likely that abstract is found if this block is longer than 50 characters
                            self.output_result['abstract'] = value['text'].replace('\n', ' ')
                            abstract_status = True

                    elif value['position_x'][0] >= (abstract_location * 0.9) and value['position_x'][0] <= (
                            abstract_location * 1.1):
                        if value['text'] != identifier:
                            self.output_result['abstract'] = value['text'].replace('\n', ' ')
                            abstract_status = True

                # Title
                if title__status == False:
                    if value['font']:
                        if value['font']['font_size_max'] > 13 and value['number_of_word'] > 3:
                            if value['obj_mid'] >= (value['page_x'] * 0.8) and value['obj_mid'] <= (
                                    value['page_x'] * 1.2):  # centred?
                                if len(value['font']['max_out_of_mixed']) > 1:  # Chars with max font size
                                    self.output_result['title'] = value['font']['max_out_of_mixed']
                                    title__status == True

                            elif value['position_x'][1] > value['page_y']:  # Ppper part of the page
                                if len(value['font']['max_out_of_mixed']) > 1:
                                    title__status == False
                                    self.output_result['title'] = value['font']['max_out_of_mixed']

                # Keywords
                if re.search('keyword', value['text'], re.IGNORECASE):
                    self.output_result['keywords'] = value['text'].replace('\n', ' ').strip()

                # doi
                pattern = re.search('(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)', value['text'])
                if pattern:
                    result = pattern.group()
                    self.output_result['doi'] = result
                else:
                    pass

            # Figure caption
            if figure_status == False:
                self.output_result['figure'] = {}
            if re.search('^fig', value['text'], re.IGNORECASE):
                order = re.search('\d+', value['text']).group()  # Sequence and text of current figure
                self.output_result['figure']['figure ' + str(order)] = value['text'].replace('\n', ' ').strip()
                figure_status = True

            # Table caption
            if table_status == False:
                self.output_result['table'] = {}

            if re.search('^table', value['text'], re.IGNORECASE):
                caption = re.search('\d+', value['text'])
                if caption:
                    order = caption.group()  # Sequence and text of current figure
                self.output_result['table']['table ' + str(order)] = value['text'].replace('\n', ' ').strip()
                figure_status = True

        return self.output_result
