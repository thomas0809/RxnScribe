import os
import json
import argparse
from rxnscribe.evaluate import ReactionEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--num_splits', type=int, default=5)
    args = parser.parse_args()
    return args


def print_scores(scores):
    for key, val in scores.items():
        print(f'{key:<10} precision: {val["precision"]:.3f}  recall: {val["recall"]:.3f}  f1: {val["f1"]:.3f}')
    print()


if __name__ == "__main__":
    args = get_args()
    gold_images = []
    pred_images = []
    for split in range(args.num_splits):
        data_path = os.path.join(args.data_path, f'test{split}.json')
        pred_path = os.path.join(args.pred_path, f'{split}/prediction_test{split}.json')
        with open(data_path) as f:
            data = json.load(f)
        with open(pred_path) as f:
            predictions = json.load(f)
        max_len = max(len(data['images']), len(predictions['reaction']))
        gold_images += data['images'][:max_len]
        pred_images += predictions['reaction'][:max_len]

    evaluator = ReactionEvaluator()
    print('Exact match')
    scores, summarize, size_stats = evaluator.evaluate_summarize(gold_images, pred_images)
    print_scores(scores)
    scores, group_stats = evaluator.evaluate_by_group(gold_images, pred_images)
    print_scores(scores)
    print('Soft match')
    scores, summarize, size_stats = evaluator.evaluate_summarize(gold_images, pred_images,
                                                                 mol_only=True, merge_condition=True)
    print_scores(scores)
    scores, group_stats = evaluator.evaluate_by_group(gold_images, pred_images, mol_only=True, merge_condition=True)
    print_scores(scores)
