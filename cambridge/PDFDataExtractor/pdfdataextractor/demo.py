from pdfdataextractor import Reader
import glob


def read_single(file):
    reader = Reader()
    pdf = reader.read_file(file)

    try:
        # print(pdf.plaintext())
        # print(pdf.test())

        a = pdf.abstract(chem=True)
        print(a.records.serialize())

        # print(pdf.title())
        # print(pdf.doi())
        # print(pdf.caption())
        # print(pdf.author())
        # print(pdf.journal())
        # print(pdf.keywords())

        # print(pdf.section())
        # for i,j in pdf.reference().items():
        #     print(i)
        #     print(j)


    except:
        pass


def read_multiple(path):
    for seq, pdf in enumerate(path):
        read_single(pdf)
        print('-------------------', '\n')

# read_multiple(glob.glob(r'/Users/miao/Desktop/PDFDataExtractor/SI/Others/acs/*.pdf'))
# read_single(r'/Users/miao/Documents/untitled folder/articles/Elesvier/battery/-Green--effects-of-hybrid-actors-through-carbon-tr_2020_Global-Transitions-P.pdf')
# read_single(r'/Volumes/Backup/PDE_papers/articles/Elesvier/dssc/The-effect-of-molecular-structure-on-the-properties-of-quinox_2020_Dyes-and-.pdf')
