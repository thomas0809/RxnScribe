import os
import reactiondataextractor as rde

image_path = 'rde_evaluation_set/'
output_path = 'output/'
image_files = sorted([f for f in os.listdir(image_path) if f[0] != '.'])

os.makedirs(output_path, exist_ok=True)
rde.extract_images(image_path, output_path)
