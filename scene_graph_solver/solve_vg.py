from LPProblem import VGLPProblem
from clevr_block_gen.constraints import get_vg_constraint_map
import torch
import json
from tqdm import tqdm
from multiprocessing import Pool

with open('./results/vg/schema.json', 'r') as f:
    schema = json.load(f)


def get_problem(scene):
    problem_solver = VGLPProblem(attr_map=schema, opposite=True, person=True, loop=True)
    return problem_solver.solve_for_scene(scene)


if __name__ == '__main__':
    constraints = get_vg_constraint_map()
    with open('./results/vg/probablistic_scenes.pytorch', 'rb') as f:
        scenes = torch.load(f)['scenes']

    with Pool(10) as p:
        predicted_scenes = list(tqdm(p.imap(get_problem, scenes), total=len(scenes)))

    # problem_solver = VGLPProblem(attr_map=schema, opposite=True, person=True, transitivity=True)
    # predicted_scenes = []
    # for scene, problem in tqdm(list(zip(scenes, problems))):
    #     problem_solver.solve_problem(scene, *problem)

    with open('./results/vg/scene_fixed.json', 'w') as f:
        json.dump({'scenes': predicted_scenes}, f)
