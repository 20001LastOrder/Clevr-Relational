import json
import networkx as nx
from typing import Dict, List, Union, Optional
import numpy as np
from collections import defaultdict

def read_json(file_path: str) -> Union[Dict, List]:
    with open(file_path, 'r') as f:
        return json.load(f)


def convert_to_nx(scene, directions, colors=None) -> nx.Graph:
    graph = nx.MultiDiGraph()
    nodes = [(i, data) for i, data in enumerate(scene['objects'])]

    graph.add_nodes_from(nodes)
    for i, direction in enumerate(directions):
        data = scene['relationships'][direction]
        for source, targets in enumerate(data):
            for target in targets:
                if colors is not None:
                    graph.add_edge(source, target, direction, color=colors[i])
                else:
                    graph.add_edge(source, target, direction, direction=direction)
    return graph


def convert_to_nx_di(scene, directions) -> nx.Graph:
    graph = nx.DiGraph()
    nodes = [(i, data) for i, data in enumerate(scene['objects'])]
    edges = defaultdict(set)
    graph.add_nodes_from(nodes)
    for i, direction in enumerate(directions):
        data = scene['relationships'][direction]
        for source, targets in enumerate(data):
            for target in targets:
                edges[(source, target)].add(direction)

    for (source, target), directions in edges.items():
        graph.add_edge(source, target, directions=directions)
    return graph


def get_attributes(obj: Dict[str, List[float]], attr_map: Dict[str, List[str]], attr_names: List[str]) -> List[str]:
    attributes = []
    for attr_name in attr_names:
        idx = np.argmax(obj[attr_name])
        attributes.append(attr_map[attr_name][idx])
    return attributes


def find_matching_object(obj_desc: str, objects: List[Dict[str, List[float]]], attr_map: Dict[str, List[str]],
                         attributes: List[str], ) -> Optional[Dict[str, List[float]]]:
    """
    find the object in the scene with the same attributes value defined in as string in the order same as the one in
    `attributes`
    """
    target_objs = []
    for obj in objects:
        obj_str = ' '.join(get_attributes(obj, attr_map, attributes))
        if obj_str == obj_desc:
            target_objs.append(obj)
    if len(target_objs) > 1:
        raise Exception('Have more than one matching objects')
    return target_objs[0] if len(target_objs) > 0 else None


def is_more_significant(target: Dict[str, List[float]], objects: List[Dict[str, List[float]]], attr_name: str,
                        attr_idx: int) -> bool:
    """
    Check if the probability in `attr_idx` of `attr_name` is more significant (higher) than other objects whose
    prediction was not `attr_idx`
    """
    target_proba = target[attr_name][attr_idx]
    more_significant = True
    for obj in objects:
        # do not check the target object itself and other objects whose attr_idx was predicted to be the highest
        if obj != target and np.argmax(obj[attr_name]) != attr_idx:
            more_significant = more_significant and (target_proba > obj[attr_name][attr_idx])

    return more_significant


def process_gt_scenes(scene: Dict, schema: Dict) -> Dict:
    """
    Process the ground truth scenes to make sure they only contains desired attributes and relationships
    :param scene: ground truth scene
    :param schema: schema contains the list of attributes and relationships
    :return:
    """
    objects = []
    for obj in scene['objects']:
        objects.append({attr_name: obj[attr_name] for attr_name in schema['attributes'].keys()})

    relationships = {rel_name: scene['relationships'][rel_name] for rel_name in schema['relations']}

    return {
        'objects': objects,
        'relationships': relationships
    }
