import json
import networkx as nx
from typing import Dict, List, Union, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm


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


def dict_match(dict_1: Dict, dict_2: Dict) -> bool:
    return dict_1 == dict_2


def dict_cost(dict_1: Dict, dict_2: Dict) -> int:
    cost = 0
    if dict_1.keys() != dict_2.keys():
        return max(len(dict_1.keys()), len(dict_2.keys()))
    for attr_name in dict_1.keys():
        if dict_1[attr_name] != dict_2[attr_name]:
            cost += 1
    return cost


def minimum_graph_edit_match(predicted_graph: nx.Graph, gt_graph: nx.Graph, timeout: int = 10, node_ins_cost: int = 4,
                             edge_ins_cost: int = 2) -> List:
    if nx.is_isomorphic(predicted_graph, gt_graph, node_match=dict_match, edge_match=dict_match):
        return []
    else:
        return list(
            nx.optimize_edit_paths(predicted_graph, gt_graph, node_subst_cost=dict_cost, edge_subst_cost=dict_cost,
                                   node_ins_cost=lambda a: node_ins_cost, edge_ins_cost=lambda e: edge_ins_cost,
                                   timeout=timeout))


def error_classification(predicted_graph: nx.Graph, gt_graph: nx.Graph, graph_matches: List[List]) -> Dict:
    if len(graph_matches) == 0:
        return {}

    error_map = {
        'missing_objects': [],
        'more_objects': [],
        "missing_edges": [],
        "more_edges": [],
        'attribute_errors': [],
        'relationship_errors': []
    }
    path = graph_matches[-1]
    node_matches = path[0]
    edge_matches = path[1]
    for src, target in node_matches:
        if src is None:
            error_map['missing_objects'].append(gt_graph.nodes[target])
            continue
        if target is None:
            error_map['more_objects'].append(predicted_graph.nodes[src])
            continue

        node1_dict = predicted_graph.nodes[src]
        node2_dict = gt_graph.nodes[target]
        if node1_dict != node2_dict:
            error_map['attribute_errors'].append((node1_dict, node2_dict))

    for src, target in edge_matches:
        if src is None:
            error_map['missing_edges'].append(gt_graph.edges[target])
            continue
        if target is None:
            error_map['more_edges'].append(predicted_graph.edges[src])
            continue

        edge1_dict = predicted_graph.edges[src]
        edge2_dict = gt_graph.edges[target]
        if edge1_dict != edge2_dict:
            error_map['relationship_errors'].append((edge1_dict, edge2_dict))
    return error_map


def error_classification_for_scenes(predicted_scenes: List[Dict], gt_scenes: List[Dict], relationships: List[str],
                                    progress: bool = True, timeout: int = 10, node_ins_cost: int = 4,
                                    edge_ins_cost: int = 2) -> List[Dict]:
    error_maps = []
    if progress:
        collection = tqdm(list(zip(predicted_scenes, gt_scenes)))
    else:
        collection = zip(predicted_scenes, gt_scenes)

    for i, (scene, gt_scene) in enumerate(collection):
        graph, gt_graph = convert_to_nx_di(scene, relationships), convert_to_nx_di(gt_scene, relationships)
        graph_matches = minimum_graph_edit_match(graph, gt_graph, timeout, node_ins_cost, edge_ins_cost)
        if len(graph_matches) == 0:
            # the predicted graph is isomorphic as the ground truth graph, no error should be recorded
            continue

        error_map = error_classification(graph, gt_graph, graph_matches)
        error_map['id'] = i
        error_maps.append(error_map)

    return error_maps
