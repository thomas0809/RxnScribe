import sys
import json

input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(input_filename) as f:
    data = json.load(f)

img_to_bbox = {}
for anno in data['annotations']:
    image_id = anno['image_id']
    if image_id not in img_to_bbox:
        img_to_bbox[image_id] = []
    img_to_bbox[image_id].append({
        'bbox': anno['bbox'],
        'category_id': anno['category_id'],
        'id': anno['id']
    })

for image in data['images']:
    image['bboxes'] = img_to_bbox[image['id']]

# data.pop('annotations')

with open(output_filename, 'w') as f:
    json.dump(data, f)
