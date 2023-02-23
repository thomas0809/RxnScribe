import argparse
import torch

from rxnscribe import RxnScribe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--image_path', type=str, default=None, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    model = RxnScribe(args.model_path, device)

    print(model.predict_image_file(args.image_path, molscribe=True, ocr=True))


if __name__ == "__main__":
    main()
