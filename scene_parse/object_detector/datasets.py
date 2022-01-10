from .utils import preprocess_rle, rle_masks_to_boxes, process_object_mask
import os
import json
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, Callable, List


class ObjectDetectorDataset(ABC):
    category_map: Dict[str, int]

    # map object to its corresponding category with its attributes
    obj_to_cat: Callable[[Dict], int]
    img_height: int
    img_width: int

    def __init__(self, category_map: Dict[str, int], obj_to_cat: Callable[[Dict], int], image_folder,
                 annotation_fp, img_height: int = 320, img_width: int = 480):
        self.category_map = category_map
        self.obj_to_cat = obj_to_cat
        self.img_width = img_width
        self.img_height = img_height
        self.image_folder = image_folder
        self.annotation_fp = annotation_fp

    def dataset_loader(self) -> Callable[[], List[Dict]]:
        dataset = self.load_dataset()
        return lambda: dataset

    def get_categories(self):
        categories = [''] * len(self.category_map)
        for cat, idx in self.category_map.items():
            categories[idx] = cat
        return categories

    def load_dataset(self) -> List[Dict]:
        with open(self.annotation_fp) as f:
            scenes = json.load(f)

        annotated_scenes = []
        for scene in tqdm(scenes['scenes'], 'loading dataset'):
            annotated_scene = {
                'file_name': os.path.join(self.image_folder, scene['image_filename']),
                'height': self.img_height,
                'width': self.img_width,
                'image_id': scene['image_index']
            }

            objs = []
            for anno in scene['objects']:
                obj = process_object_mask(anno, self.obj_to_cat(anno))
                objs.append(obj)
            annotated_scene['annotations'] = objs
            annotated_scenes.append(annotated_scene)
        return annotated_scenes


class ClevrSingleClassDataset(ObjectDetectorDataset):

    def get_category_map(self) -> Dict[str, int]:
        return {'object': 0}

    def obj_to_category(self, category_map: Dict[str, int]) -> Callable[[Dict], int]:
        return lambda obj: 0

    def load_dataset(self, image_folder: str, annotation_file: str, obj_to_category: Callable[[Dict], int]) \
            -> List[Dict]:
        with open(annotation_file) as f:
            scenes = json.load(f)

        annotated_scenes = []
        for scene in tqdm(scenes['scenes'], 'loading dataset'):
            annotated_scene = {
                'file_name': os.path.join(image_folder, scene['image_filename']),
                'height': 320,
                'width': 480,
                'image_id': scene['image_index']
            }

            objs = []
            for anno in scene['objects']:
                obj = process_object_mask(anno, obj_to_category(anno))
                objs.append(obj)
            annotated_scene['annotations'] = objs
            annotated_scenes.append(annotated_scene)
        return annotated_scenes


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
    category_map = {cat: i for i, cat in enumerate(categories)}


class CarlaDataset(ObjectDetectorDataset):
    def get_category_map(self) -> Dict[str, int]:
        return {'car': 0}

    def obj_to_category(self, category_map: Dict[str, int]) -> Callable[[Dict], int]:
        return lambda obj: 0