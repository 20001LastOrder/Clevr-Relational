from .utils import preprocess_rle, rle_masks_to_boxes
import os
import json
from tqdm import tqdm
import numpy as np

def get_complete_categories():
    colors = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
    materials = ['rubber', 'metal']
    shapes = ['cube', 'cylinder', 'sphere']
    size = ['small', 'large']

    category_ids = []
    categories = []
    cat_id = 1
    for c in colors:
        for m in materials:
            for s in shapes:
                categories.append(' '.join([c, m, s]))
                category_ids.append(cat_id)
                cat_id += 1
    category_map = {cat : i for i, cat in enumerate(categories)}
    
def load_clevr(dataset_folder, annotation_file, obj_to_category):
    folder = dataset_folder
    with open(os.path.join(folder, annotation_file)) as f:
        scenes = json.load(f)
    img_dir = os.path.join(folder, 'images')
    annotated_scenes = []
    for scene in tqdm(scenes['scenes'], 'loading dataset'):
        annotated_scene = {
            'file_name': os.path.join(img_dir, scene['image_filename']),
            'height': 320,
            'width': 480,
            'image_id': scene['image_index']
        }

        objs = []
        for anno in scene['objects']:
            obj = {}
            rle = preprocess_rle(anno['mask'])
            bbox, _ = rle_masks_to_boxes([rle])
            obj['bbox'] = bbox[0]
            obj['segmentation'] = rle
            obj['bbox_mode'] = 0
            obj['category_id'] = obj_to_category(anno)
            objs.append(obj)
        annotated_scene['annotations'] = objs
        annotated_scenes.append(annotated_scene)
    return annotated_scenes