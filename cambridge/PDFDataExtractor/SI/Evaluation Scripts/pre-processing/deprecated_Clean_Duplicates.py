'''
This is used to clean duplicate from downloaded piis

'''


import glob
import re

path = '/Users/miao/Desktop/PDFDataExtractor/SI/Evaluation Scripts/pii/'



def clean(Input_file, year):
    lines_seen = set()#Seen lines stored here
    with open(path + year + "_cleaned.json", "w") as f:
        for line in open(Input_file, "r"):
            if line not in lines_seen:  # check if line is not duplicate
                f.write(line)
                lines_seen.add(line)

for file in glob.glob(path + '*.json'):
    year = re.search('\d+', file).group()
    clean(file, year)

