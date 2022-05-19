from collections import defaultdict
import numpy as np
from utils import read_json, error_classification_for_scenes, process_gt_scenes
import os
from scene_graph_transformer import get_concrete_scene_graph, get_simple_concrete_attributes


class ProblemDomain:
    def __init__(self, constraints, metrics, gt_folder, predicted_scenes_folder, schema_folder):
        self.constraints = constraints
        self.metrics = metrics
        self.gt_scenes = read_json(os.path.join(gt_folder, 'scenes.json'))['scenes']
        self.schema = read_json(schema_folder)

        self.gt_scenes = [process_gt_scenes(scene, self.schema) for scene in self.gt_scenes]
        self.predicted_scenes_folder = predicted_scenes_folder

        self.predicted_scenes = None
        self.predicted_scenes_fixed = None
        self.coord_scenes = None
        self.coord_scenes_fixed = None

    def load_predicted_scenes(self, coord_scene_transformer,
                              scenes_fname='rel_scenes.json',
                              fixed_scenes_fname='rel_scenes_fix_c1234.json',
                              coord_scenes_fname='scenes_coord_prob.json',
                              coord_fixed_fname='rel_scenes_fix_coord.json'):
        if scenes_fname is not None:
            print(os.path.join(self.predicted_scenes_folder, scenes_fname))
            self.predicted_scenes = read_json(os.path.join(self.predicted_scenes_folder, scenes_fname))['scenes']
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
        return get_error_statistics(scenes, self.gt_scenes, self.schema, self.constraints, self.metrics, sample_size)


def get_error_statistics(predicted_scenes, gt_scenes, schema, constraints, metrics, sample_size=200):
    error_maps = error_classification_for_scenes(predicted_scenes, gt_scenes, schema['relations'])
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
