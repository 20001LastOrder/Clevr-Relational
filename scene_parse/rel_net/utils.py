import os
import json
import numpy as np


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


def get_feat_vec_clevr(obj):
    attr_to_idx = {
        'shape': {
            'sphere': 0,
            'cube': 1,
            'cylinder': 2,
        },
        'size': {
            'large': 0,
            'small': 1,
        },
        'material': {
            'metal': 0,
            'rubber': 1,
        },
        'color': {
            'blue': 0,
            'brown': 1,
            'cyan': 2,
            'gray': 3,
            'green': 4,
            'purple': 5,
            'red': 6,
            'yellow': 7
        }
    }
    features = {}

    if obj is not None:
        for attr in ['color', 'material', 'shape', 'size']:
            features[attr] = attr_to_idx[attr][obj[attr]]
        features['position'] = obj['position']
    else:
        features['shape'] = attr_to_idx['shapes']['none']
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
