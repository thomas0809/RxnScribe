from pdfdataextractor import Reader
from difflib import SequenceMatcher
import glob
import os
import string
import re
import xml.etree.ElementTree as ET


def get_result_xml(path):
    r = {'title': '',
         'abstract': '',
         'doi': '',
         'keywords': '',
         'section': '',
         'author': '',
         'figure': {},
         'journal': {'name': '', 'volume': '', 'year': '', 'page': ''},
         'reference': {}
         }


    author = []
    keywords = []
    section = []

    mytree = ET.parse(path)
    root = mytree.getroot()

    for i in root.iter():
        if i.tag == '{http://purl.org/dc/elements/1.1/}identifier':
            r['doi'] = i.text

        elif i.tag == '{http://www.elsevier.com/xml/common/dtd}title':
            r['title'] = i.text

        elif i.tag == '{http://purl.org/dc/elements/1.1/}creator':
            author.append(i.text)

        elif i.tag == '{http://purl.org/dc/elements/1.1/}description':
            r['abstract'] = i.text

        elif i.tag == '{http://purl.org/dc/terms/}subject':
            keywords.append(i.text)

        elif i.tag == '{http://www.elsevier.com/xml/common/dtd}figure':
            label = ''
            for j in i.iter('{http://www.elsevier.com/xml/common/dtd}label'):
                if re.search('\d+', j.text):
                    label = re.search('\d+', j.text).group()
                else:
                    continue

            for k in i.iter('{http://www.elsevier.com/xml/common/dtd}simple-para'):
                r['figure']['figure ' + label] = k.text

        elif i.tag == '{http://www.elsevier.com/xml/svapi/article/dtd}coredata':

            for publisher in i.iter('{http://prismstandard.org/namespaces/basic/2.0/}publicationName'):
                r['journal']['name'] = publisher.text

            for volume in i.iter('{http://prismstandard.org/namespaces/basic/2.0/}volume'):
                r['journal']['volume'] = volume.text

            for date in i.iter('{http://prismstandard.org/namespaces/basic/2.0/}coverDisplayDate'):
                r['journal']['year'] = re.sub('[^0-9]', '', date.text)

            for page in i.iter('{http://prismstandard.org/namespaces/basic/2.0/}pageRange'):
                r['journal']['page'] = page.text

        elif i.tag == '{http://www.elsevier.com/xml/common/dtd}bib-reference':

            for seq, element in enumerate(root.iter('{http://www.elsevier.com/xml/common/dtd}bib-reference')):
                buffer = []
                for j in element.iter():
                    buffer = j.tag

                    ref = []
                    for text in element.iter():
                        if text.text:
                            ref.append(text.text)

                    ref = ' '.join(ref)
                    ref = " ".join(ref.split())
                    r['reference'][seq] = ref

                if '{http://www.elsevier.com/xml/common/dtd}source-text' in buffer:
                    for text in element.iter():
                        if text.tag == '{http://www.elsevier.com/xml/common/dtd}source-text':
                            if not text.text:
                                pass
                            else:
                                r['reference'][seq] = text.text

            break

        elif i.tag == '{http://www.elsevier.com/xml/xocs/dtd}item-toc-entry':

            buffer = ['', '']
            for j in i.iter():
                if j.tag == '{http://www.elsevier.com/xml/xocs/dtd}item-toc-label':
                    buffer[0] = j.text

                elif j.tag == '{http://www.elsevier.com/xml/xocs/dtd}item-toc-section-title':
                    buffer[1] = j.text

            if '.' in buffer[0]:
                pass
            else:
                if None in buffer:
                    continue
                else:
                    section.append(''.join(buffer))

    author_corrected = author[0].split(',')
    author_corrected.reverse()

    r['section'] = section
    r['author'] = author_corrected  # swap places here
    r['keywords'] = keywords

    return r


def get_result_pdf(path):
    reader = Reader()
    pdf = reader.read_file(path)
    r = {'title': '',
         'abstract': '',
         'doi': '',
         'keywords': '',
         'section': '',
         'journal': {'name': '', 'volume': '', 'year': '', 'page': ''},
         'figure': '',
         'author': '',
         'reference': '',
         }

    try:
        r['title'] = pdf.title()
        r['abstract'] = pdf.abstract()
        r['doi'] = pdf.doi()
        r['keywords'] = pdf.keywords()
        r['section'] = pdf.section()
        r['journal'] = pdf.journal()
        r['figure'] = pdf.caption()
        r['author'] = pdf.author()
        r['reference'] = pdf.reference()

    except Exception as e:
        pass

    return r


def test_write(line):
    import csv
    output = line

    with open('/Users/miao/Desktop/PDFDataExtractor/SI/elsevier.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(output)


def get_results():
    categories = ['battery', 'catalysis', 'cells', 'city', 'dssc', 'light', 'nano', 'neel', 'solar', 'super alloy']

    for category in categories:
        path_xml = r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/xml/{}/'.format(category)
        path_pdf = r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/{}/'.format(category)

        # pdf and xml share the same file name, using one of them is okay
        for pdf in glob.glob(path_pdf + '/*.pdf'):
            name = os.path.splitext(pdf)[0]
            name_short = name.split('/')[-1]

            xml = get_result_xml(path_xml + name_short + '.xml')
            pdf = get_result_pdf(path_pdf + name_short + '.pdf')

            data = compare(xml, pdf)
            data.insert(0, name_short)
            test_write(data)
            print(data)
            print('-------')


def journal_calculate(a_list):
    total = 0.0
    for info in a_list:
        if info:
            total += float(info)
        elif info == None:
            total += 0.0
    if total == 0.0:
        return 0
    else:
        return total/4


def compare(xml, pdf):
    data = []

    # get title similarity
    title = similarity(xml['title'], pdf['title'])
    data.append(title)

    # get doi similarity
    doi = similarity(xml['doi'], pdf['doi'])
    data.append(doi)

    # get abstract similarity
    abstract = similarity(xml['abstract'], pdf['abstract'])
    data.append(abstract)

    # get keywords similarity
    keywords = similarity(xml['keywords'], pdf['keywords'])
    data.append(keywords)

    # get author similarity
    if pdf['author']:
        author = similarity(xml['author'], pdf['author'])
        data.append(author)
    else:
        author = similarity(xml['author'], '')
        data.append(author)

    # get section similarity
    if pdf['section']:
        section = similarity(xml['section'], [i for i in pdf['section'].keys()])
        data.append(section)
    else:
        section = similarity(xml['section'], [''])
        data.append(section)

    # get journal similarity
    journal_buffer = []
    for info_type in ['name', 'volume', 'year', 'page']:
        journal = similarity(xml['journal'][info_type], pdf['journal'][info_type])
        journal_buffer.append(journal)
    data.append(journal_calculate(journal_buffer))

    #get figure results
    for figure_r in figure_results(xml, pdf):
        data.append(figure_r)

    # get reference results
    for reference_r in reference_results(xml, pdf):
        data.append(reference_r)

    return data

def figure_results(xml, pdf):
    tp = 0
    fp = 0
    fn = 0

    if bool(xml['figure']) == False:
        if bool(pdf['figure']) == False:  # F + F
            return  0, 0, 0
        else:  # F + T
            xx = pdf['figure']
            count = 0
            for seq, (key, value) in enumerate(xx.items()):
                count += 1
            fp = count

    elif bool(xml['figure']) == True:  # T + T
        if bool(pdf['figure']) == True:
            pdf_fig = []
            xml_fig = []

            a = pdf['figure']
            b = xml['figure']

            for seq, (key, value) in enumerate(a.items()):
                pdf_fig.append(value)

            for seq, (key, value) in enumerate(b.items()):
                xml_fig.append(value)

            for pdf_fig_entry, xml_fig_entry in zip(pdf_fig, xml_fig):
                r = similarity(pdf_fig_entry, xml_fig_entry)
                # print(similarity(pdf_fig_entry, xml_fig_entry))
                if r >= 0.7:
                    tp += 1
                elif r < 0.7:
                    fp += 1

        else:  # T + F
            b = xml['figure']
            count = 0

            for seq, (key, value) in enumerate(b.items()):
                count += 1

            fn = count

    return tp, fp, fn

def reference_results(xml, pdf):
    tp = 0
    fp = 0
    fn = 0

    if bool(xml['reference']) == False:
        if bool(pdf['reference']) == False:  # F + F
            return  0, 0, 0
        else:  # F + T
            xx = pdf['reference']
            count = 0
            for seq, (key, value) in enumerate(xx.items()):
                count += 1
            fp = count

    elif bool(xml['reference']) == True:  # T + T
        if bool(pdf['reference']) == True:
            pdf_ref = []
            xml_ref = []

            c = pdf['reference']
            d = xml['reference']

            for seq, (key, value) in enumerate(c.items()):
                pdf_ref.append(value)

            for seq, (key, value) in enumerate(d.items()):
                xml_ref.append(value)

            for pdf_ref_entry, xml_ref_entry in zip(pdf_ref, xml_ref):
                r = similarity(pdf_ref_entry, xml_ref_entry)
                # print(similarity(pdf_ref_entry, xml_ref_entry))
                if r >= 0.6:
                    tp += 1
                elif r < 0.6:
                    fp += 1

        else:  # T + F
            b = xml['reference']
            count = 0

            for seq, (key, value) in enumerate(b.items()):
                count += 1

            fn = count

    return tp, fp, fn


def similarity(string_1, string_2):

    if string_2:
        if isinstance(string_1, list):
            string_1 = ''.join(string_1)

        if isinstance(string_2, list):
            string_2 = ''.join(string_2)

        string_1 = normalise_string(string_1)
        string_2 = normalise_string(string_2)

        # print(string_1)
        # print(string_2)

        r = SequenceMatcher(None, string_1, string_2).ratio()

        return r

    else:
        return


def normalise_string(text):
    if text:
        text_lower_case = text.lower()  # lower the case #todo: do this here, prevent error
        text_no_spaces = text_lower_case.replace(" ", "")  # remove white spaces
        text_normalised = text_no_spaces.translate(str.maketrans('', '', string.punctuation))  # remove punctuations
        return text_normalised
    else:
        return ''


get_results()