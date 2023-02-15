import os
import json
import shutil
import argparse
import multiprocessing
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--num_splits', type=int, default=5)
    return parser.parse_args()


def process_image(image_file, tmp_file):
    import reactiondataextractor as rde
    # print(image_file)
    output = rde.extract_image(image_file)
    # print('DONE')
    image = Image.open(image_file)
    height, width = image.height, image.width
    results = []
    for r in output.reaction_steps:
        res = {'reactants': [], 'conditions': [], 'products': []}
        for m in r.reactants:
            p = m.panel
            x1, y1, x2, y2 = p.left / width, p.top / height, p.right / width, p.bottom / height
            res['reactants'].append({
                'bbox': [x1, y1, x2, y2],
                'category': '[Mol]',
                'category_id': 1,
            })
        for m in r.products:
            p = m.panel
            x1, y1, x2, y2 = p.left / width, p.top / height, p.right / width, p.bottom / height
            res['products'].append({
                'bbox': [x1, y1, x2, y2],
                'category': '[Mol]',
                'category_id': 1,
            })
        for p in r.conditions.text_lines:
            x1, y1, x2, y2 = p.left / width, p.top / height, p.right / width, p.bottom / height
            res['conditions'].append({
                'bbox': [x1, y1, x2, y2],
                'category': '[Txt]',
                'category_id': 2,
            })
        results.append(res)
    with open(os.path.join(tmp_file), 'w') as of:
        json.dump(results, of)
    return


if __name__ == "__main__":
    args = get_args()
    for split in range(args.num_splits):
        data_path = os.path.join(args.data_path, f'test{split}.json')
        pred_path = os.path.join(args.pred_path, f'{split}/prediction_test{split}.json')
        os.makedirs(os.path.join(args.pred_path, f'{split}'), exist_ok=True)
        tmp_path = os.path.join(args.pred_path, f'{split}/tmp')
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(tmp_path, exist_ok=False)
        with open(data_path) as f:
            data = json.load(f)
        # prediction = [process_image(os.path.join(args.image_path, image['file_name'])) for image in data['images'][:10]]
        # image_files = [os.path.join(args.image_path, image['file_name']) for image in data['images']]
        image_files = [image['file_name'] for image in data['images']]
        prediction = []
        num_failed = 0
        num_total = len(image_files)
        failed_images = []
        batch_size = 40
        for start_idx in range(0, num_total, batch_size):
            end_idx = min(start_idx + batch_size, num_total)
            processes = []
            for idx in range(start_idx, end_idx):
                name = image_files[idx]
                p = multiprocessing.Process(
                    target=process_image,
                    args=(os.path.join(args.image_path, name), os.path.join(tmp_path, name))
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join(120)
            # with multiprocessing.Pool(processes=batch_size+10) as pool:
            #     results = [pool.apply_async(process_image, (image_file,))
            #                for image_file in image_files[start_idx:end_idx]]
            #     for i, res in enumerate(results):
            #         try:
            #             prediction.append(res.get(timeout=120))
            #         except Exception as e:
            #             print(e)
            #             num_failed += 1
            #             prediction.append([])
        for idx in range(num_total):
            name = image_files[idx]
            tmp_file = os.path.join(tmp_path, name)
            if os.path.exists(tmp_file):
                with open(tmp_file) as f:
                    prediction.append(json.load(f))
            else:
                num_failed += 1
                prediction.append([])
                failed_images.append(name)
        print('Failed:', num_failed)
        print('Total:', num_total)
        print(failed_images)
        with open(pred_path, 'w') as f:
            json.dump({'reaction': prediction}, f)
