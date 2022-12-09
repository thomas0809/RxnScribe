import os
import json
import argparse
from reaction.evaluate import ReactionEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--num_splits', type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    gold_images = []
    pred_images = []
    for split in range(5):
        data_path = os.path.join(args.data_path, f'test{split}.json')
        pred_path = os.path.join(args.pred_path, f'{split}/prediction_test{split}.json')
        with open(data_path) as f:
            data = json.load(f)
        with open(pred_path) as f:
            predictions = json.load(f)
        gold_images += data['images']
        pred_images += predictions['reaction']

    evaluator = ReactionEvaluator()
    scores, summarize, group_stats = evaluator.evaluate_summarize(gold_images, pred_images)
    print(json.dumps(scores, indent=4))
    for key1 in ['overall', 'single', 'multiple']:
        for key2 in ['precision', 'recall', 'f1']:
            print("%.3f" % scores[key1][key2], end=' ')
    print()
    scores, summarize, group_stats = evaluator.evaluate_summarize(gold_images, pred_images,
                                                                  mol_only=True, merge_condition=True)
    print(json.dumps(scores, indent=4))
    for key1 in ['overall', 'single', 'multiple']:
        for key2 in ['precision', 'recall', 'f1']:
            print("%.3f" % scores[key1][key2], end=' ')
    print()
