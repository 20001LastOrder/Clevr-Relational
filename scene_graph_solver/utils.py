from typing import Dict
import numpy as np

def process_coord_scene(scene: Dict, dataset: str):
    if dataset == 'blockworld':
        return process_coord_scene_blocksworld(scene)
    else:
        raise Exception('Dataset is not supported!')


def process_coord_scene_blocksworld(scene: Dict):
    n = len(scene['objects'])
    relationships = {
        'above': [[] for _ in range(n)],
        'below': [[] for _ in range(n)],
        'left': [[] for _ in range(n)],
        'right': [[] for _ in range(n)]
    }

    for i, o1 in enumerate(scene['objects']):
        for j, o2 in enumerate(scene['objects']):
            if o1 == o2:
                continue
            relation = ''
            if o1['x'] - o2['x'] > 0.1:
                relation = 'left'
            if o1['x'] - o2['x'] < -0.1:
                relation = 'right'
            if o1['z'] > o2['z'] and np.abs(o1['x'] - o2['x']) < 0.1:
                relation = 'below'
            if o1['z'] < o2['z'] and np.abs(o1['x'] - o2['x']) < 0.1:
                relation = 'above'
            # assume the relationships are perfectly predicted
            relationships[relation][i].append([j, 1])
            for key in relationships:
                if key != relation:
                    relationships[key][i].append([j, 0])

    scene['relationships'] = relationships
    for o in scene['objects']:
        del o['x']
        del o['z']

    return scene

def process_coord_scene_clevr(scene: Dict):
    n = len(scene['objects'])
    relationships = {
        'front': [[] for _ in range(n)],
        'behind': [[] for _ in range(n)],
        'left': [[] for _ in range(n)],
        'right': [[] for _ in range(n)]
    }

    relation_map = {
        'left': 'right',
        'right': 'left',
        'front': 'behind',
        'behind': 'front'
    }

    for i, o1 in enumerate(scene['objects']):
        for j, o2 in enumerate(scene['objects']):
            if o1 == o2:
                continue
            if o1['x'] > o2['x']:
                relation = 'left'
            if o1['x'] < o2['x']:
                relation = 'right'

            relationships[relation][i].append([j, 1])
            relationships[relation_map[relation]][i].append([j, 0])

            if o1['y'] > o2['y']:
                relation = 'behind'
            if o1['y'] < o2['y']:
                relation = 'front'
            # assume the relationships are perfectly predicted
            relationships[relation][i].append([j, 1])
            relationships[relation_map[relation]][i].append([j, 0])

    scene['relationships'] = relationships
    for o in scene['objects']:
        del o['x']
        del o['y']

    return scene
