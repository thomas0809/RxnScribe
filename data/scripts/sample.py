# Sample diagram images from raw data for annotation
# Images are sampled based on a pre-defined amount for each journal type
#   and a size constraint (xxx, xxx)

import os, sys
from random import shuffle
from tqdm.auto import tqdm
from pathlib import Path
from collections import defaultdict

NUM_SAMPLES = {
    'jacs': 1000,
    'joc': 1000,
    'ol': 500,
    'op': 500
}

PREFIXES = {
    'jacs': ['jacs', 'ja'],
    'joc': ['acs.joc', 'jo'],
    'ol': ['ol', 'acs.orglett'],
    'op': ['op', 'acs.oprd']
}

data = defaultdict(list)

# read diagram files into data
with open("raw/doi.list") as f:
    doi_list = f.read().splitlines()

image_dir = "raw/diagrams"
for doi in doi_list:
    if any([doi.startswith(k) for k in PREFIXES['jacs']]):
        data['jacs'].append(doi)
    elif any([doi.startswith(k) for k in PREFIXES['joc']]):
        data['joc'].append(doi)
    elif any([doi.startswith(k) for k in PREFIXES['ol']]):
        data['ol'].append(doi)
    elif any([doi.startswith(k) for k in PREFIXES['op']]):
        data['op'].append(doi)
    else:
        print(f"Unknown journal type for: {doi}")

# shuffle DOIs for each journal
for k in data.keys():
    shuffle(data[k])

# sampling
target_dir = "./annotate/images"
os.makedirs(target_dir, exist_ok=True)

def sample(dois, n):
    selected = []
    for doi in tqdm(dois[:n], total=n):
        dir_path = os.path.join("raw/diagrams", doi)
        p = Path(dir_path)
        for i, img in enumerate(p.glob('*-c*.png')):
            img_parent, img_name = os.path.split(img)
            target_path = os.path.join(target_dir, doi + f"-{img_name[4:]}")
            os.system(f'cp {img} {target_path}')
            selected.append([str(img), target_path])
        # if len(selected) >= n:
        #     break
    return selected

sampled_data = {}
for k, v in data.items():
    print(f"Sampling {k}")
    sampled_data[k] = sample(v, NUM_SAMPLES[k])

import json
with open("sampled_data.json", "w") as f:
    json.dump(sampled_data, f, indent=2)

