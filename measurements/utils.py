import json
import networkx as nx
from typing import Dict, List, Union, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pycocotools.mask as mask_util


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
        new_obj = {attr_name: obj[attr_name] for attr_name in schema['attributes'].keys()}
        new_obj['mask'] = obj['mask']
        objects.append(new_obj)

    relationships = {rel_name: scene['relationships'][rel_name] for rel_name in schema['relations']}

    return {
        'objects': objects,
        'relationships': relationships
    }


def process_coord_scene(scene: Dict, schema: Dict) -> Dict:
    objects = []
    for obj in scene['objects']:
        new_obj = {attr_name: obj[attr_name] for attr_name in schema['attributes'].keys()}
        new_obj['mask'] = obj['mask']
        objects.append(new_obj)

    relationships = {rel_name: [[] for _ in objects] for rel_name in schema['relations']}

    for i, o1 in enumerate(scene['objects']):
        for j, o2 in enumerate(scene['objects']):
            if o1 == o2:
                continue
            if o1['x'] - o2['x'] > 0.1:
                relationships['left'][i].append(j)
            if o1['x'] - o2['x'] < -0.1:
                relationships['right'][i].append(j)
            if o1['z'] > o2['z'] and np.abs(o1['x'] - o2['x']) < 0.1:
                relationships['below'][i].append(j)
            if o1['z'] < o2['z'] and np.abs(o1['x'] - o2['x']) < 0.1:
                relationships['above'][i].append(j)

    return {
        'objects': objects,
        'relationships': relationships
    }


def dict_match(dict_1: Dict, dict_2: Dict) -> bool:
    return dict_1 == dict_2


def IoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union


def node_cost(node1: Dict, node2: Dict):
    mask1 = mask_util.decode(node1['mask'])
    mask2 = mask_util.decode(node2['mask'])

    node1 = {key: value for key, value in node1.items() if key != 'mask'}
    node2 = {key: value for key, value in node2.items() if key != 'mask'}

    if IoU(mask1, mask2) < 0.5:
        return 1000
    else:
        return dict_cost(node1, node2)


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
        # currently not possible with the object masks
        return []
    else:
        return list(
            nx.optimize_edit_paths(predicted_graph, gt_graph, node_subst_cost=node_cost, edge_subst_cost=dict_cost,
                                   node_ins_cost=lambda a: node_ins_cost, node_del_cost=lambda a: node_ins_cost,
                                   timeout=timeout))


def remove_node_attribute(graph: nx.Graph, attr_name):
    for node_id in graph.nodes:
        del graph.nodes[node_id][attr_name]
    return graph


def error_classification(predicted_graph: nx.Graph, gt_graph: nx.Graph, graph_matches: List[List]) -> Dict:
    if len(graph_matches) == 0:
        print(f'Graph matching for graphs failed for {predicted_graph} and {gt_graph}!!!')
        return {}

    predicted_graph = remove_node_attribute(predicted_graph, 'mask')
    gt_graph = remove_node_attribute(gt_graph, 'mask')
    num_attributes = sum([len(attributes) for _, attributes in gt_graph.nodes(data=True)])
    num_edges = sum([len(edge['directions']) for _, _, edge in gt_graph.edges(data=True)])

    error_map = {
        'missing_objects': [],
        'more_objects': [],
        "missing_edges": [],
        "more_edges": [],
        'attribute_errors': [],
        'relationship_errors': [],
        'SGGen': num_edges,
        'SGGen+': num_edges + num_attributes,
        'SA': 1,
        'FSA': 1
    }
    path = graph_matches[-1]
    node_matches = path[0]
    edge_matches = path[1]
    non_matching_nodes = set()

    for src, target in node_matches:
        if src is None:
            error_map['missing_objects'].append(gt_graph.nodes[target])
            error_map['FSA'] = 0
            error_map['SA'] = 0

            # remove all attr_score
            error_map['SGGen+'] -= len(gt_graph.nodes[target])

            continue
        if target is None:
            error_map['SA'] = 0
            error_map['more_objects'].append(predicted_graph.nodes[src])
            continue

        node1_dict = {key: value for key, value in predicted_graph.nodes[src].items()}
        node2_dict = {key: value for key, value in gt_graph.nodes[target].items()}
        if node1_dict != node2_dict:
            error_map['FSA'] = 0
            error_map['SA'] = 0
            error_map['attribute_errors'].append((node1_dict, node2_dict))

            # non_matching attributes
            error_map['SGGen+'] -= sum([1 for key in node2_dict if node1_dict[key] != node2_dict[key]])
            non_matching_nodes.add(target)

    for src, target in edge_matches:
        if src is None:
            error_map['missing_edges'].append(gt_graph.edges[target])
            error_map['FSA'] = 0
            error_map['SA'] = 0
            error_map['SGGen'] -= len(gt_graph.edges[target])
            error_map['SGGen+'] -= len(gt_graph.edges[target])
            continue
        if target is None:
            error_map['SA'] = 0
            error_map['more_edges'].append(predicted_graph.edges[src])
            continue

        edge1_dict = predicted_graph.edges[src]
        edge2_dict = gt_graph.edges[target]
        diff = len(edge2_dict['directions'].difference(edge1_dict['directions']))
        if diff > 0:
            error_map['FSA'] = 0
            error_map['SA'] = 0
            error_map['relationship_errors'].append((edge1_dict, edge2_dict))
            error_map['SGGen+'] -= diff

        error_map['SGGen'] -= len(gt_graph.edges[target]['directions']) \
            if target[0] in non_matching_nodes or target[1] in non_matching_nodes else diff

    error_map['SGGen'] /= num_edges
    error_map['SGGen+'] /= (num_edges + num_attributes)

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
        if graph_matches[-1][2] == 0:
            error_maps.append({'SGGen': 1, 'SGGen+': 1, 'SA': 1, 'FSA': 1})
            continue

        error_map = error_classification(graph, gt_graph, graph_matches)
        error_map['id'] = i
        error_maps.append(error_map)

    return error_maps


def dag_constraint(adj: np.matrix):
    """
    measure the number of violations of  the dag constraint
    :param adj: the adjacency matrix
    :return: number of violations (number of cycles from a node to itself at all path length)
    """

    cumulator = adj
    result = adj
    n = adj.shape[-1]

    for i in range(1, n):
        cumulator = adj @ cumulator
        result = result + cumulator
    return np.multiply(result, np.eye(n)).sum()


def anti_symmetry_constraint(adj: np.matrix):
    """
    measure the number of anti_symmetry constraint violation
    :param adj: the adjacency matrix
    :return:
    """
    anti_symmetry = ~np.logical_xor(adj, adj.transpose((0, 2, 1)))
    filter_mat = 1 - np.eye(adj.shape[-1])

    return np.multiply(anti_symmetry, filter_mat).sum() / 2  # each violation is counted twice


def construct_adj(relationships, num_nodes):
    adj = np.zeros((len(relationships), num_nodes, num_nodes))

    for i, (rel, pairs) in enumerate(relationships.items()):
        for source, targets in enumerate(pairs):
            for target in targets:
                adj[i, source, target] = 1

    return adj
