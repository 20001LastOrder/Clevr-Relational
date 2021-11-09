import abc

from .utils import preprocess_rle, rle_masks_to_boxes
import os
import json
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, Callable, List


class ObjectDetectorDataset(ABC):
    @abstractmethod
    def get_category_map(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def obj_to_category(self, category_map: Dict[str, int]) -> Callable[[Dict], int]:
        pass

    @abstractmethod
    def load_dataset(self, image_folder: str, annotation_file: str, obj_to_category: Callable[[Dict], int]) \
            -> List[Dict]:
        pass

    def dataset_loader(self, image_folder: str, annotation_file: str) -> Callable[[], List[Dict]]:
        obj_to_category = self.obj_to_category(self.get_category_map())
        dataset = self.load_dataset(image_folder, annotation_file, obj_to_category)
        return lambda: dataset


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

    def get_categories(self):
        category_map = self.get_category_map()
        categories = [''] * len(category_map)
        for cat, idx in category_map.items():
            categories[idx] = cat
        return categories


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


def load_clevr(dataset_folder, annotation_file, obj_to_category):
    folder = dataset_folder
    with open(os.path.join(folder, annotation_file)) as f:
        scenes = json.load(f)
    img_dir = os.path.join(folder, 'images'),
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
