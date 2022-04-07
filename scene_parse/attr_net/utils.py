import os
import json
import numpy as np
from typing import Dict, List
import pickle


def invert_dict(d):
    return {v: k for k, v in d.items()}


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_feat_vec(obj: Dict, attr_map: Dict[str, List], directions: Dict[str, List]):
    features = {}

    if obj is not None:
        for attr in attr_map.keys():
            features[attr] = attr_map[attr].index(obj[attr])
        features['x'] = np.dot(obj['location'], directions['right'])
        features['y'] = np.dot(obj['location'], directions['front'])
        features['z'] = np.dot(obj['location'], directions['above'])
    else:
        features = {attr: -1 for attr in attr_map.keys()}
    return features


def get_attrs_clevr(feat_vec):
    shapes = ['sphere', 'cube', 'cylinder']
    sizes = ['large', 'small']
    materials = ['metal', 'rubber']
    colors = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
    obj = {
        'shape': shapes[np.argmax(feat_vec[0:3])],
        'size': sizes[np.argmax(feat_vec[3:5])],
        'material': materials[np.argmax(feat_vec[5:7])],
        'color': colors[np.argmax(feat_vec[7:15])],
        'position': feat_vec[15:18].tolist(),
    }
    return obj


def load_clevr_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_dict = json.load(f)['scenes']
    scenes = []
    for s in scenes_dict:
        objs = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            if '3d_coords' in o:
                item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                    np.dot(o['3d_coords'], s['directions']['front']),
                                    o['3d_coords'][2]]
            else:
                item['position'] = o['position']
            item['color'] = o['color']
            item['material'] = o['material']
            item['shape'] = o['shape']
            item['size'] = o['size']
            item['mask'] = o['mask']
            objs.append(item)
        scenes.append({
            'objects': objs,
        })
    return scenes


def iou(m1, m2):
    intersect = m1 * m2
    union = 1 - (1 - m1) * (1 - m2)
    return intersect.sum() / union.sum()


def iomin(m1, m2):
    if m1.sum() == 0 or m2.sum() == 0:
        return 1.0
    intersect = m1 * m2
    return intersect.sum() / min(m1.sum(), m2.sum())