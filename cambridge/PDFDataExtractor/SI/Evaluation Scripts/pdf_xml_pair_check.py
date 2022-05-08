"""
this is used make sure that articles exist in both pdf and xml
"""

import shutil
import glob
import os


def xml_list(path):
    """
    this function reads in a json file for each category
    and return a list which contains all file names
    """

    xml_names = []
    for xml in glob.glob(path + '/*.xml'):
        name = os.path.splitext(xml)[0]
        name_short = name.split('/')[-1]
        xml_names.append(name_short)

    return xml_names


def pdf_list(path):
    """
    this function reads in a json file for each category
    and return a list which contains all file names
    """

    pdf_names = []
    for pdf in glob.glob(path + '/*.pdf'):
        name = os.path.splitext(pdf)[0]
        name_short = name.split('/')[-1]
        pdf_names.append(name_short)

    return pdf_names


def pdf_xml_pair_check(path_xml, path_pdf):
    """
    this function checks the difference between pdf sets and xml sets
    and returns files to be removed in sets
    """
    xml = xml_list(path_xml)
    pdf = pdf_list(path_pdf)
    difference = set(pdf) - set(xml)
    print(len(difference))

    return difference


def get_excessive_pdf():
    """
    this function is a generator to be used by 'remove_excessive_pdf()'
    """
    categories = ['battery', 'catalysis', 'cells', 'city', 'dssc', 'light', 'nano', 'neel', 'solar', 'super alloy']

    for category in categories:
        path_xml = r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/xml/{}'.format(category)
        path_pdf = r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/{}'.format(category)
        dump_list = pdf_xml_pair_check(path_xml, path_pdf)
        yield (category, dump_list)


def remove_excessive_pdf():
    """
    this function equivalently checks the difference between pdf and xml sets
    and removed any unmatched pdf files
    """
    categories = get_excessive_pdf()  # a generator which contains the category name and corresponding file names

    for articles in categories:
        category = articles[0] # articles[0] is the category name, for example, 'battery' or 'catalysis'

        for article in articles[1]: # articles[1] is a list which contains all the file names that need to be removed
            pdf_file_path = r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/{}/{}'.format(category, article) + '.pdf'
            shutil.move(pdf_file_path, r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/test')

remove_excessive_pdf()
