from LPProblem import BlocksworldLPProblem
from tqdm import tqdm
from LPProblem import ClevrLPProblem
import json
from utils import process_coord_scene_clevr, process_coord_scene_blocksworld
import argparse
import os


def main(args):
    if args.dataset_name == 'clevr':
        process_coord_scene = process_coord_scene_clevr
    elif args.dataset_name == 'block':
        process_coord_scene = process_coord_scene_blocksworld
    else:
        raise Exception('Dataset not supported!')

    with open(os.path.join(args.folder, args.src_file), 'r') as f:
        scenes = json.load(f)['scenes']
    with open(os.path.join(args.schema_fp), 'r') as f:
        attr_map = json.load(f)

    if args.is_coord:
        scenes = [process_coord_scene(scene) for scene in scenes]

    if args.dataset_name == 'clevr':
        problem_solver = ClevrLPProblem(attr_map, color_between=True, material_order=True, large_cube=True, object_behind=True)
    elif args.dataset_name == 'block':
        problem_solver = BlocksworldLPProblem(attr_map, include_small_obj=True, include_large_cube=True, include_bottom_stack=True, include_yellow=True)
    else:
        raise Exception('Dataset not supported!')

    predicted_scenes = []
    for scene in tqdm(scenes):
        predicted_scenes.append(problem_solver.solve_for_scene(scene))

    with open(os.path.join(args.folder, args.output_file), 'w') as f:
        json.dump({'scenes': predicted_scenes}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--src_file', required=True)
    parser.add_argument('--schema_fp', required=True)
    parser.add_argument('--is_coord', action='store_true', default=False)
    parser.add_argument('--output_file', required=True)

    arguments = parser.parse_args()
    main(arguments)
