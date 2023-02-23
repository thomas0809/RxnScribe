import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch

from rxnscribe import RxnScribe

def main():

    model_path = 'output/pix2seq_reaction_nov_cv_ep600_3e-4/0/checkpoints/best.ckpt'
    image_file = 'assets/acs.joc.5b01703-Scheme-c1.png'
    device = torch.device('cuda')
    model = RxnScribe(model_path, device)

    print(model.predict_image_file(image_file, molscribe=True, ocr=True))


if __name__ == "__main__":
    main()
