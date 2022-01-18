from typing import Dict, List
import numpy as np


def get_concrete_scene_graph(graph: Dict, schema: Dict, model: str = 'simple') -> Dict:
    """
    Transform a probabilistic scene graph into a concrete scene graph
    :param graph:
    :param model: 'simple' or 'consistent'
    :return:
    """
    if model == 'simple':
        return get_simple_concrete_scene_graph(graph, schema)


def get_simple_concrete_attributes(graph: Dict, schema: Dict) -> List[Dict]:
    objects = []
    attributes = schema['attributes']
    # handle objects attributes
    for obj in graph['objects']:
        new_obj = {}
        for attr_name, probs in obj.items():
            new_obj[attr_name] = attributes[attr_name][np.argmax(probs)]
        objects.append(new_obj)
    return objects


def get_simple_concrete_scene_graph(graph: Dict, schema: Dict) -> Dict:
    objects = get_simple_concrete_attributes(graph, schema)

    # handle relationship
    relationships = {}
    for rel_name, rels in graph['relationships'].items():
        relationships[rel_name] = [[target for target, prob in rel if prob > 0.5] for rel in rels]

    return {
        'objects': objects,
        'relationships': relationships
    }

def get_consistent_concrete_scene_graph(graph: Dict, schema: Dict) -> Dict:
    pass