"""
this is used to down journals in XML format using previously downloaded doi
"""

import requests
import glob
import re
import time
import json
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument


def get_doi_from_pdf():
    """
    get dois from downloaded articles. This function checks every pdf file and finds doi by looking
    at the embedded metadata. It creates a json file for each category.
    """

    categories = ['battery', 'catalysis', 'cells', 'city', 'dssc', 'light', 'nano', 'neel', 'solar', 'super alloy']
    doi_pattern = '(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)'

    for category in categories:
        dois = {}
        total = 0
        folder = r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/{}/*.pdf'.format(category)

        for paper in glob.glob(folder):
            pdf = open(paper, 'rb')
            text = pdf.name
            file_name = re.findall(r'[^\/]+(?=\.)', text)[0]
            parser = PDFParser(pdf)
            doc = PDFDocument(parser)

            try:
                doi = re.search(doi_pattern, str(doc.info[0]['doi']))
                dois[doi.group()] = file_name
                total += 1
                print(doi.group())

            # doi may be hidden in other key-value pairs
            except KeyError:
                for key, value in doc.info[0].items():
                    doi = re.search(doi_pattern, str(value))
                    if doi:
                        dois[doi.group()] = file_name
                        total += 1
                        print(doi.group())
                    else:
                        pass

        with open(r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/xml/{}.json'.format(category), "w") as f:
            json.dump(dois, f)

    return


def download_xml():
    """
    This function reads in a json file which contains dois and file name, and download articles in xml correspondingly.
    """
    api_key = ''
    url = 'https://api.elsevier.com/content/article/doi/{}?apiKey={}&httpAccept=text%2Fxml'
    path = r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/xml/{}.json'
    # categories = ['battery', 'catalysis', 'cells', 'city', 'dssc', 'light', 'nano', 'neel', 'solar', 'super alloy']
    categories = ['cells', 'city', 'dssc', 'light', 'nano', 'neel', 'solar', 'super alloy']
    count = 0

    for category in categories:
        # load dois for each category
        f = open(path.format(category), "r")
        data = json.loads(f.read())
        f.close()

        # download xml for each category
        for doi, file_name in data.items():
            print('Downloading: ', file_name)
            request_url = url.format(doi, api_key)

            try:
                with open(r'/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/xml/{}/{}.xml'.format(category,
                                                                                                            file_name),
                          'w') as xml:
                    r = requests.get(request_url).text
                    xml.write(r)
                    count += 1

                    if count % 10:
                        time.sleep(5)  # need to sleep for 5 seconds, otherwise the server returns error

            except Exception as e:
                print(e)
                continue


# get_doi_from_pdf()  # run this function first to get json files for each category
download_xml()
