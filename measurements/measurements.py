from collections import defaultdict
import numpy as np
from utils import read_json, error_classification_for_scenes, process_gt_scenes
import os
from scene_graph_transformer import get_concrete_scene_graph, get_simple_concrete_attributes
from tqdm import tqdm


class ProblemDomain:
    def __init__(self, constraints, metrics, gt_folder, predicted_scenes_folder, schema_folder, use_mask=True,
                 process_gt=True):
        self.constraints = constraints
        self.metrics = metrics
        self.use_mask = use_mask
        self.gt_scenes = read_json(os.path.join(gt_folder, 'scenes.json'))['scenes']
        self.schema = read_json(schema_folder)

        if process_gt:
            self.gt_scenes = [process_gt_scenes(scene, self.schema, self.use_mask) for scene in self.gt_scenes]

        self.predicted_scenes_folder = predicted_scenes_folder

        self.predicted_scenes = None
        self.predicted_scenes_fixed = None
        self.coord_scenes = None
        self.coord_scenes_fixed = None

    def load_predicted_scenes(self, coord_scene_transformer,
                              scenes_fname='rel_scenes.json',
                              fixed_scenes_fname='rel_scenes_fix_c1234.json',
                              coord_scenes_fname='scenes_coord_prob.json',
                              coord_fixed_fname='rel_scenes_fix_coord.json', concrete_scene=False):
        if scenes_fname is not None:
            print(os.path.join(self.predicted_scenes_folder, scenes_fname))
            self.predicted_scenes = read_json(os.path.join(self.predicted_scenes_folder, scenes_fname))['scenes']
            if not concrete_scene:
                self.predicted_scenes = [get_concrete_scene_graph(scene, self.schema) for scene in self.predicted_scenes]

        if fixed_scenes_fname is not None:
            self.predicted_scenes_fixed = read_json(os.path.join(self.predicted_scenes_folder, fixed_scenes_fname))['scenes']

        if coord_scenes_fname is not None:
            self.coord_scenes = read_json(os.path.join(self.predicted_scenes_folder, coord_scenes_fname))['scenes']
            self.coord_scenes = [coord_scene_transformer(scene, self.schema) for scene in self.coord_scenes]
            for scene in self.coord_scenes:
                scene['objects'] = get_simple_concrete_attributes(scene, self.schema)
        if coord_fixed_fname is not None:
            self.coord_scenes_fixed = read_json(os.path.join(self.predicted_scenes_folder, coord_fixed_fname))['scenes']

    def cal_matrics_statistics(self, sample_size=200):
        if self.predicted_scenes is not None:
            self.scene_stats = self.cal_single_error_statistics(self.predicted_scenes, sample_size)
        if self.predicted_scenes_fixed is not None:
            self.scene_fixed_stats = self.cal_single_error_statistics(self.predicted_scenes_fixed, sample_size)
        if self.coord_scenes is not None:
            self.coord_scene_stats = self.cal_single_error_statistics(self.coord_scenes, sample_size)
        if self.coord_scenes_fixed is not None:
            self.coord_scene_fixed_stats = self.cal_single_error_statistics(self.coord_scenes_fixed, sample_size)

    def cal_single_error_statistics(self, scenes, sample_size):
        return get_error_statistics(scenes, self.gt_scenes, self.schema, self.constraints, self.metrics, sample_size,
                                    use_mask=self.use_mask)


def get_error_statistics(predicted_scenes, gt_scenes, schema, constraints, metrics, sample_size=200, use_mask=True):
    error_maps = error_classification_for_scenes(predicted_scenes, gt_scenes, schema['relations'], use_mask=use_mask)
    return get_metrics(error_maps, predicted_scenes, constraints, metrics, sample_size)


def get_metrics(error_maps, predicted_scenes, constraints, metrics, sample_size=200):
    result = defaultdict(list)

    for i in range(0, len(error_maps), sample_size):
        scene_partition = predicted_scenes[i: i + sample_size]
        for key in metrics:
            result[key].append(np.mean([m[key] for m in error_maps[i: i + sample_size]]))

        for scene in scene_partition:
            for idx, obj in enumerate(scene['objects']):
                obj['id'] = idx

        errors = [sum([constraint.evaluate(scene) for constraint in constraints]) for scene in
                  scene_partition]
        result['Con'].append(len([e for e in errors if e == 0]) / len(errors))

        for constraint in constraints:
            result[constraint.name].append(sum([constraint.evaluate(scene) for scene in scene_partition]))
    return result


def to_adj(relationships, num_nodes):
    tuples = [['' for _ in range(num_nodes)] for _ in range(num_nodes)]
    for rel, sources in relationships.items():
        for source, targets in enumerate(sources):
            for target in targets:
                tuples[source][target] = rel
    return tuples


def to_tuple_list(relationships):
    tuples = []
    for rel, sources in relationships.items():
        for source, targets in enumerate(sources):
            for target in targets:
                tuples.append((source, target, rel))
    return tuples

def get_metrics_node_ordered(gt_scenes, predicted_scenes, constraints):
    # assume the objects are deterministic and in order
    results = []
    for gt_scene, predicted_scene in tqdm(list(zip(gt_scenes, predicted_scenes))):
        metric_result = {
            'SGGen': 0,
            'SGGen+': 0
        }

        num_objects = len(gt_scene['objects'])
        gt_tuples = to_tuple_list(gt_scene['relationships'])
        predicted_adj = to_adj(predicted_scene['relationships'], num_objects)

        for source, target, rel in gt_tuples:
            if predicted_adj[source][target] == rel:
                metric_result['SGGen+'] += 1
                if gt_scene['objects'][source]['label'] == predicted_scene['objects'][source]['label'] and \
                        gt_scene['objects'][target]['label'] == predicted_scene['objects'][target]['label']:
                    metric_result['SGGen'] += 1

        for gt_obj, predicted_obj in zip(gt_scene['objects'], predicted_scene['objects']):
            if gt_obj['label'] == predicted_obj['label']:
                metric_result['SGGen+'] += 1

        metric_result['SGGen'] = metric_result['SGGen'] / len(gt_tuples)
        metric_result['SGGen+'] = metric_result['SGGen+'] / (len(gt_tuples) + len(gt_scene['objects']))
        metric_result['SA'] = int(metric_result['SGGen'] == 1)
        metric_result['Con'] = sum([constraint.evaluate(predicted_scene) for constraint in constraints]) == 0
        results.append(metric_result)

    metrics = results[0].keys()
    return {
        metric: np.mean([result[metric] for result in results]) for metric in metrics
    }, results




