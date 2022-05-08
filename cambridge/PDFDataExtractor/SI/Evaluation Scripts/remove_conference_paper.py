"""
this is used remove any conference paper that does not follow Elsevier format
"""

from pdfdataextractor import Reader
import glob
import shutil


def read_single(file, category):
    reader = Reader()
    pdf = reader.read_file(file)

    # remove any none Elsevier paper(some publishers use search service provided by Elsevier)
    if pdf is None:
        path = "/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/test/{}".format(category)
        shutil.move(file, path)

    # remove any conference paper
    else:
        try:
            for text_block in pdf.output_page(0):# make sure it is first page
                text = text_block['text']

                if 'Procedia'in text:
                    print(text)
                    dump_folder = "/Users/miao/Desktop/PDFDataExtractor/SI/articles/Elesvier/test/{}".format(category)
                    shutil.move(file, dump_folder)
                    break

        except AttributeError:
            pass


def read_multiple(path, category):
    for seq, file in enumerate(path):
        print('Reading article: ' + str(seq))
        read_single(file, category)
        print('---------------------------------------', '\n')


def remove_conference_paper():
    categories = ['battery', 'catalysis', 'cells', 'city', 'dssc', 'light', 'nano', 'neel', 'solar', 'super alloy']

    for category in categories:
        path = r'/Users/miao/Downloads/{}/*.pdf'.format(category)
        read_multiple(glob.glob(path), category)


remove_conference_paper()

