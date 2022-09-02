import json
import copy
import random

with open('annotations.json') as f:
    annotations = json.load(f)

random.seed(42)
images = annotations['images']
random.shuffle(images)

n = len(images)
n_train = int(n * 0.7)
n_test = int(n * 0.2)

for i in range(5):
    end = (i+1)*n_test if i != 4 else n
    test_images = images[i*n_test:end]
    train_images = images[end:] + images[:i*n_test]
    train_images, dev_images = train_images[:n_train], train_images[n_train:]
    with open(f'splits/train{i}.json', 'w') as f:
        data = copy.deepcopy(annotations)
        data['images'] = train_images
        json.dump(data, f)
    with open(f'splits/dev{i}.json', 'w') as f:
        data = copy.deepcopy(annotations)
        data['images'] = dev_images
        json.dump(data, f)
    with open(f'splits/test{i}.json', 'w') as f:
        data = copy.deepcopy(annotations)
        data['images'] = test_images
        json.dump(data, f)
