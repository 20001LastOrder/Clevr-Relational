# %%
import pulp
import json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

REL_MAP = {
    'left': 'right',
    'front': 'behind'
}


def main(args):
    with open(args.input, 'r') as f:
        scenes = json.load(f)

    for scene in tqdm(scenes['scenes'], 'Processing scenes...'):
        relationships = {}
        for rel in REL_MAP:
            ps = {}
            for source, targets in enumerate(scene['relationships'][rel]):
                for target, p in targets:
                    ps[f'{source},{target}'] = p
            relations = cal_most_prob_scene(ps, len(scene['objects']))
            ops_relations = reverse_relation(relations)

            relationships[rel] = relations
            relationships[REL_MAP[rel]] = ops_relations
        scene['relationships'] = relationships

    with open(args.output, 'w') as f:
        json.dump(scenes, f)


def cal_most_prob_scene(edges, num_objects, eps=1e-50):
    indexes = []
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                indexes.append(f'{i},{j}')

    variables = pulp.LpVariable.dict('', indexes, 0, 1, pulp.LpInteger)

    prob = pulp.LpProblem("sceneGraphProblem", pulp.LpMaximize)
    prob += pulp.lpSum([variables[i] * np.log(max(edges[i], eps)) +
                        (1 - variables[i]) * np.log(max(1 - edges[i], eps)) for i in indexes])

    # constriants
    # between two node, there exist one and only one edge
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                prob += variables[f'{i},{j}'] + variables[f'{j},{i}'] == 1

    # The transitivity must be followed
    for i in range(num_objects):
        for j in range(num_objects):
            for k in range(num_objects):
                if i != j and j != k and i != k:
                    prob += variables[f'{i},{j}'] + variables[f'{j},{k}'] - variables[f'{i},{k}'] <= 1
    # %%
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if status < 0:
        raise ValueError('The problem is unsolvable')

    rels = [[] for _ in range(num_objects)]
    for v in prob.variables():
        if v.varValue:
            edge = list(map(int, v.name.replace('_', '').split(',')))
            rels[edge[0]].append(edge[1])

    return rels


def reverse_relation(rels):
    all_nodes = set(range(len(rels)))

    ops_rels = []
    for source, targets in enumerate(rels):
        ops_rels.append(list(all_nodes.difference(set(targets + [source]))))

    return ops_rels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    main(args)
