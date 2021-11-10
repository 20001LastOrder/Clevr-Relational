import json
import networkx as nx
from typing import Dict, List, Union, Optional
import numpy as np


def read_json(file_path: str) -> Union[Dict, List]:
    with open(file_path, 'r') as f:
        return json.load(f)


def convert_to_nx(scene, directions, colors) -> nx.Graph:
    graph = nx.MultiDiGraph()
    nodes = [(i, data) for i, data in enumerate(scene['objects'])]

    graph.add_nodes_from(nodes)
    for color, direction in zip(colors, directions):
        data = scene['relationships'][direction]
        for source, targets in enumerate(data):
            for target in targets:
                graph.add_edge(source, target, direction, color=color)
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
