# extract diagrams from ACS data

acs_data = "../tml_data"
# list of filenames in tml_data
file_list = "../tml_data.files"

import os
from tqdm.auto import tqdm
output_dir = "/data/scratch/jiang_guo/chemie/diagram-parse/tml_data/diagrams"
os.makedirs(output_dir, exist_ok=True)

with open(file_list) as f:
    pdf_files = f.read().split("\n")
    pdf_files = [x for x in pdf_files if x.endswith(".pdf")]

print(f"{len(pdf_files)} PDF documents found.")
def extract(filename):
    dirname = os.path.join(output_dir, filename[:-4])
    os.makedirs(dirname, exist_ok=True)
    prefix = dirname + "/rxn"
    try:
        filepath = os.path.join(acs_data, filename)
        os.system(f"pdffigures {filepath} -c {prefix} -o {prefix}")
    except:
        print(f"{filepath} extraction error.")
        os.system(f"rm -rf {dirname}")

from multiprocessing import Pool
p = Pool(20)
p.map(extract, pdf_files)

