import argparse
import json
from typing import List

from tqdm import tqdm
from blocks import State, load_colors, Unstackable
from constraints import get_constraint_map, SceneConstraint


def initialize_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--properties-json', default='data/properties.json',
                        help="JSON file defining objects, materials, sizes, colors, and constraints.")
    parser.add_argument('--allow-duplicates', action="store_true",
                        help="Allow duplicate objects")

    # Settings for objects
    parser.add_argument('--min-objects', default=3, type=int,
                        help="The minimum number of objects to place in each scene")
    parser.add_argument('--max-objects', default=10, type=int,
                        help="The maximum number of objects to place in each scene")
    parser.add_argument('--table-size', default=5, type=int,
                        help="The approximate table size relative to the large object size * 1.5.")
    parser.add_argument('--object-jitter', default=0.0, type=int,
                        help="The magnitude of random jitter to add to the x,y position of each block.")

    # Output settings
    parser.add_argument('--num-scenes', default=100, type=int,
                        help="The number of images to render")
    parser.add_argument('--output-file', default='output/scenes_not_rendered.json',
                        help="The directory where output will be stored. It will be " +
                             "created if it does not exist.")
    return parser


def evaluate_constraints(state: State, constraints: List[SceneConstraint]):
    """
    Evaluate the constraint for the scene, true means all constraints are satisfied
    """
    for constraint in constraints:
        if not constraint.evaluate(state):
            return False
    return True


def main(args):
    properties = load_colors(args)
    constraint_map = get_constraint_map()
    constraints = [constraint_map[cname] for cname in properties['constraints']]

    num_scenes_generated = 0
    scenes = []

    # Regular generation
    # progress = tqdm(args.num_scenes)
    # while len(scenes) < args.num_scenes:
    #     try:
    #         state = State(args)
    #         num_scenes_generated += 1
    #
    #         if evaluate_constraints(state, constraints):
    #             objects = state.for_rendering()
    #             scenes.append({
    #                 'objects': objects
    #             })
    #             progress.update(1)
    #
    #         progress.set_description(f'Generate scenes ({num_scenes_generated} generated)', True)
    #     except Unstackable:
    #         continue

    # Generation with more than 10 objects
    progress = tqdm(args.num_scenes)
    objects_sizes = list(range(11, 21))
    while len(scenes) < 10:
        try:
            num_objects = objects_sizes[len(scenes)]
            args.min_objects = num_objects
            args.max_objects = num_objects
            state = State(args)
            num_scenes_generated += 1

            if evaluate_constraints(state, constraints):
                objects = state.for_rendering()
                scenes.append({
                    'objects': objects
                })
                progress.update(1)

            progress.set_description(f'Generate scenes ({num_scenes_generated} generated)', True)
        except Unstackable:
            continue

    with open(args.output_file, 'w') as f:
        json.dump({'scenes': scenes, 'constraints': properties['constraints']}, f)


if __name__ == '__main__':
    parser = initialize_parser()
    main(parser.parse_args())