import os
import multiprocessing
from datetime import datetime
import reactiondataextractor as rde

image_path = 'rde_evaluation_set/'
output_path = 'output/'
image_files = sorted([f for f in os.listdir(image_path) if f[0] != '.'])
output_file = os.path.join(output_path, datetime.now().strftime("%Y_%m_%d_%H_%M_%S.txt"))

os.makedirs(output_path, exist_ok=True)
# rde.extract_images(image_path, output_path)


def extract_image(image_file, name):
    result = rde.extract_image(image_file)
    print(result)
    with open(os.path.join(output_file), 'a') as f:
        f.write(name + '\n')
    return


for name in image_files:
    p = multiprocessing.Process(target=extract_image, args=(os.path.join(image_path, name), name))
    p.start()
    p.join()

print('Total:', len(image_files))
with open(output_file) as f:
    success_log = [line.strip() for line in f.readlines()]
    print('Success:', len(success_log))

failed = [f for f in image_files if f not in success_log]
print('Failed:', len(failed))
print(failed)
