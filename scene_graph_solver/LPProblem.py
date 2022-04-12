from itertools import product

from gurobipy import and_, or_, GRB, Model
import gurobipy
import numpy as np


class BlocksworldLPProblem:
    def __init__(self, attr_map, include_small_obj=False, include_large_cube=False, include_bottom_stack=False,
                 include_yellow=False):
        self.attr_map = attr_map
        self.include_small_obj = include_small_obj
        self.include_large_cube = include_large_cube
        self.include_bottom_stack = include_bottom_stack
        self.include_yellow = include_yellow

    def solve_for_scene(self, scene):
        model, attr_variables, rel_variables = self.get_LP_problem(scene)
        self.add_constraints(model, scene, rel_variables, attr_variables)
        model.setParam('OutputFlag', 0)
        model.optimize()
        predicted_scene = self.get_predicted_scene_for_block_world(attr_variables, rel_variables, scene)
        return predicted_scene

    def get_attribute_variables(self, objects):
        variables = []
        probabilities = {}
        num_objects = len(objects)
        attributes_map = self.attr_map['attributes']

        for name, values in attributes_map.items():
            for v, _ in enumerate(values):
                for i in range(num_objects):
                    variable_name = f'{name}_{i}_{v}'
                    variables.append(variable_name)
                    probabilities[variable_name] = objects[i][name][v]
        return variables, probabilities

    def get_relationship_variables(self, scene):
        variables = []
        probabilities = {}
        relationships = self.attr_map['relations']

        for rel in relationships:
            for target, sources in enumerate(scene['relationships'][rel]):
                for source, probability in sources:
                    # TODO: change the prediction for below and above (currently flipped)
                    variable_name = f'{rel}_{source}_{target}'
                    variables.append(variable_name)
                    probabilities[variable_name] = probability
        return variables, probabilities

    def get_LP_problem(self, scene, eps=1e-50):
        model = Model("scene_solver")
        attr_variables, attr_probabilities = self.get_attribute_variables(scene['objects'])
        rel_variables, rel_probabilities = self.get_relationship_variables(scene)

        attr_variables = model.addVars(attr_variables, vtype=GRB.BINARY, name='attr')
        rel_variables = model.addVars(rel_variables, vtype=GRB.BINARY, name='rel')

        attr_objective = [attr_variables[key] * np.log(max(attr_probabilities[key], eps)) for key in
                          attr_variables.keys()]
        rel_objective = [
            rel_variables[key] * np.log(max(rel_probabilities[key], eps)) + (1 - rel_variables[key]) * np.log(
                (max(1 - rel_probabilities[key], eps))) for key in rel_variables.keys()]

        model.setObjective(gurobipy.quicksum(attr_objective + rel_objective), GRB.MAXIMIZE)
        model.update()
        return model, attr_variables, rel_variables

    def get_predicted_scene_for_block_world(self, attr_variables, rel_variables, scene):
        predicted_scene = {
            'objects': [{'mask': o['mask']} for o in scene['objects']],
            'relationships': {rel: [[] for _ in range(len(scene['objects']))] for rel in self.attr_map['relations']}
        }

        for name, v in attr_variables.items():
            if v.x == 1:
                tokens = name.split('_')
                predicted_scene['objects'][int(tokens[1])][tokens[0]] = self.attr_map['attributes'][tokens[0]][
                    int(tokens[2])]

        for name, v in rel_variables.items():
            if v.x == 1:
                tokens = name.split('_')
                predicted_scene['relationships'][tokens[0]][int(tokens[2])].append(int(tokens[1]))
        return predicted_scene

    def add_constraints(self, model, scene, rel_variables, attr_variables):
        objects = scene['objects']
        num_objects = len(objects)

        def attribute_constraints():
            for attr in self.attr_map['attributes']:
                for idx in range(num_objects):
                    yield gurobipy.quicksum(
                        [attr_variables[f'{attr}_{idx}_{v}'] for v, _ in enumerate(objects[idx][attr])]) == 1

        model.addConstrs(attribute_constraints())

        # pairwise constraints
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    # choose exactly one from left, right, above and below
                    model.addConstr(rel_variables[f'above_{i}_{j}'] + rel_variables[f'below_{i}_{j}'] + rel_variables[
                        f'left_{i}_{j}'] + rel_variables[f'right_{i}_{j}'] == 1)

                    # left imply right, above imply below and vice versa
                    model.addConstr(rel_variables[f'above_{i}_{j}'] == rel_variables[f'below_{j}_{i}'])
                    model.addConstr(rel_variables[f'left_{i}_{j}'] == rel_variables[f'right_{j}_{i}'])

                    if self.include_small_obj:
                        model.addConstr(
                            1 - (attr_variables[f'size_{j}_1'] + rel_variables[f'above_{i}_{j}'] - 1) + attr_variables[
                                f'size_{i}_1'] >= 1)

        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # no opposite relationship of the same type
                for rel in self.attr_map['relations']:
                    model.addConstr(rel_variables[f'{rel}_{i}_{j}'] + rel_variables[f'{rel}_{j}_{i}'] <= 1)

        if self.include_bottom_stack:
            in_between = {}

        # triple constraints
        for i in range(num_objects):
            for j in range(num_objects):
                for k in range(num_objects):
                    if i == k or j == k or i == j:
                        continue
                    for rel in self.attr_map['relations']:
                        model.addConstr(
                            rel_variables[f'{rel}_{i}_{j}'] + rel_variables[f'{rel}_{j}_{k}'] - rel_variables[
                                f'{rel}_{i}_{k}'] <= 1)

                    model.addConstr(rel_variables[f'above_{i}_{j}'] + rel_variables[f'left_{i}_{k}'] - rel_variables[
                        f'left_{j}_{k}'] <= 1)

                    model.addConstr(rel_variables[f'below_{i}_{j}'] + rel_variables[f'left_{i}_{k}'] - rel_variables[
                        f'left_{j}_{k}'] <= 1)

                    if self.include_bottom_stack:
                        dummy = model.addVar(vtype=GRB.BINARY, name=f'in_between_{k}_{i}_{j}')
                        model.addConstr(dummy == and_(rel_variables[f'left_{i}_{k}'], rel_variables[f'left_{k}_{j}']))
                        in_between[f'in_between_{k}_{i}_{j}'] = dummy

        if self.include_bottom_stack:
            not_bottom = []

        for i in range(num_objects):
            dummy = model.addVar(vtype=GRB.BINARY, name=f'd_any_above_{i}')
            model.addConstr(dummy == or_([rel_variables[f'above_{j}_{i}'] for j in range(num_objects) if i != j]))

            # nothing can be on top of a sphere
            model.addConstr(2 - attr_variables[f'shape_{i}_0'] - dummy >= 1)

            if self.include_large_cube:
                model.addConstr(1 - (attr_variables[f'size_{i}_0'] + attr_variables[f'shape_{i}_1'] - 1) + dummy >= 1)

            if self.include_bottom_stack:
                have_above = model.addVar(vtype=GRB.BINARY, name=f'bottom_{i}')
                model.addConstr(
                    have_above == or_([rel_variables[f'below_{j}_{i}'] for j in range(num_objects) if i != j]))
                not_bottom.append(have_above)

            if self.include_yellow:
                model.addConstr((attr_variables[f'color_{i}_7'] == 1) >> (gurobipy.quicksum(
                    [rel_variables[f'right_{j}_{i}'] for j in range(num_objects) if i != j]) >= 1))

        # Two bottom objects of the stack next to each other cannot have the same color
        if self.include_bottom_stack:
            for i in range(num_objects):
                for j in range(num_objects):
                    if i == j:
                        continue
                    have_in_between = model.addVar(vtype=GRB.BINARY, name=f'have_in_between_{i}_{j}')
                    model.addConstr(have_in_between == or_([in_between[f'in_between_{k}_{i}_{j}']
                                                            for k in range(num_objects) if k != i and k != j]))

                    for c in range(len(self.attr_map['attributes']['color'])):
                        model.addConstr(
                            (1 - not_bottom[i]) + (1 - not_bottom[j]) + (1 - have_in_between) + attr_variables[
                                f'color_{i}_{c}'] + attr_variables[f'color_{j}_{c}'] <= 4)


class ClevrLPProblem:
    def __init__(self, attr_map, color_between=False, material_order=False, large_cube=False,
                 object_behind=False):
        self.attr_map = attr_map
        self.color_between = color_between
        self.material_order = material_order
        self.large_cube = large_cube
        self.object_behind = object_behind
        self.M = 100

    def solve_for_scene(self, scene):
        model, attr_variables, rel_variables = self.get_LP_problem(scene)
        self.add_constraints(model, scene, rel_variables, attr_variables)
        model.setParam('OutputFlag', 0)
        model.optimize()
        predicted_scene = self.get_predicted_scene(attr_variables, rel_variables, scene)
        return predicted_scene

    def get_attribute_variables(self, objects):
        variables = []
        probabilities = {}
        num_objects = len(objects)
        attributes_map = self.attr_map['attributes']

        for name, values in attributes_map.items():
            for v, _ in enumerate(values):
                for i in range(num_objects):
                    variable_name = f'{name}_{i}_{v}'
                    variables.append(variable_name)
                    probabilities[variable_name] = objects[i][name][v]
        return variables, probabilities

    def get_relationship_variables(self, scene):
        variables = []
        probabilities = {}
        relationships = self.attr_map['relations']

        for rel in relationships:
            for target, sources in enumerate(scene['relationships'][rel]):
                for source, probability in sources:
                    # TODO: change the prediction for below and above (currently flipped)
                    variable_name = f'{rel}_{source}_{target}'
                    variables.append(variable_name)
                    probabilities[variable_name] = probability
        return variables, probabilities

    def get_predicted_scene(self, attr_variables, rel_variables, scene):
        predicted_scene = {
            'objects': [{'mask': o['mask']} for o in scene['objects']],
            'relationships': {rel: [[] for _ in range(len(scene['objects']))] for rel in self.attr_map['relations']}
        }

        for name, v in attr_variables.items():
            if v.x == 1:
                tokens = name.split('_')
                predicted_scene['objects'][int(tokens[1])][tokens[0]] = self.attr_map['attributes'][tokens[0]][
                    int(tokens[2])]

        for name, v in rel_variables.items():
            if v.x == 1:
                tokens = name.split('_')
                predicted_scene['relationships'][tokens[0]][int(tokens[2])].append(int(tokens[1]))
        return predicted_scene

    def get_LP_problem(self, scene, eps=1e-50):
        model = Model("scene_solver")
        attr_variables, attr_probabilities = self.get_attribute_variables(scene['objects'])
        rel_variables, rel_probabilities = self.get_relationship_variables(scene)

        attr_variables = model.addVars(attr_variables, vtype=GRB.BINARY, name='attr')
        rel_variables = model.addVars(rel_variables, vtype=GRB.BINARY, name='rel')

        attr_objective = [attr_variables[key] * np.log(max(attr_probabilities[key], eps)) for key in
                          attr_variables.keys()]
        rel_objective = [
            rel_variables[key] * np.log(max(rel_probabilities[key], eps)) + (1 - rel_variables[key]) * np.log(
                (max(1 - rel_probabilities[key], eps))) for key in rel_variables.keys()]

        model.setObjective(gurobipy.quicksum(attr_objective + rel_objective), GRB.MAXIMIZE)
        model.update()
        return model, attr_variables, rel_variables

    def add_constraints(self, model, scene, rel_variables, attr_variables):
        # only choose one attribute
        objects = scene['objects']
        num_objects = len(objects)

        def attribute_constraints():
            for attr in self.attr_map['attributes']:
                for idx in range(num_objects):
                    yield gurobipy.quicksum(
                        [attr_variables[f'{attr}_{idx}_{v}'] for v, _ in enumerate(objects[idx][attr])]) == 1

        model.addConstrs(attribute_constraints())

        # pairwise constraints
        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    continue
                # exactly choose from one of left, right, one from front, behind
                model.addConstr(rel_variables[f'left_{i}_{j}'] + rel_variables[f'right_{i}_{j}'] == 1)
                model.addConstr(rel_variables[f'front_{i}_{j}'] + rel_variables[f'behind_{i}_{j}'] == 1)
                # left imply right, and front imply behind and vice versa
                model.addConstr(rel_variables[f'left_{i}_{j}'] == rel_variables[f'right_{j}_{i}'])
                model.addConstr(rel_variables[f'front_{i}_{j}'] == rel_variables[f'behind_{j}_{i}'])

                if self.material_order:
                    model.addConstr(attr_variables[f'material_{i}_{0}'] + attr_variables[f'material_{j}_{1}'] +
                                    rel_variables[f'behind_{i}_{j}'] <= 2)

        attr_names = sorted(self.attr_map['attributes'].keys())
        values = [range(len(self.attr_map['attributes'][key])) for key in attr_names]
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # no opposite relationship of the same type
                for rel in self.attr_map['relations']:
                    model.addConstr(rel_variables[f'{rel}_{i}_{j}'] + rel_variables[f'{rel}_{j}_{i}'] == 1)

                # no two objects can have exactly the same attribute combination
                # for attr, values in self.attr_map['attributes'].items():
                attr_combs = product(*values)
                for comb in attr_combs:
                    model.addConstr(
                        gurobipy.quicksum([attr_variables[f'{attr}_{i}_{v}'] + attr_variables[f'{attr}_{j}_{v}']
                                           for attr, v in zip(attr_names, comb)])
                        <= 2 * len([1 for _, _ in self.attr_map['attributes'].items()]) - 1)

        # triple constraints
        for i in range(num_objects):
            for j in range(num_objects):
                if i == j:
                    continue
                for k in range(num_objects):
                    if i == k or j == k:
                        continue
                    for rel in self.attr_map['relations']:
                        model.addConstr(
                            rel_variables[f'{rel}_{i}_{j}'] + rel_variables[f'{rel}_{j}_{k}'] - rel_variables[
                                f'{rel}_{i}_{k}'] <= 1)

        if self.color_between:
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    in_between_objs = []
                    for k in range(num_objects):
                        if i == k or j == k:
                            continue
                        dummy = model.addVar(vtype=GRB.BINARY, name=f'between_{k}_{i}_{j}')
                        i_j_between = model.addVar(vtype=GRB.BINARY, name=f'in_between_{k}_{i}_{j}')
                        j_i_between = model.addVar(vtype=GRB.BINARY, name=f'in_between_{k}_{j}_{i}')
                        model.addConstr(
                            i_j_between == and_(rel_variables[f'left_{i}_{k}'], rel_variables[f'left_{k}_{j}']))
                        model.addConstr(
                            j_i_between == and_(rel_variables[f'left_{j}_{k}'], rel_variables[f'left_{k}_{i}']))
                        model.addConstr(dummy == or_(i_j_between, j_i_between))
                        in_between_objs.append(dummy)

                    separated = model.addVar(vtype=GRB.BINARY, name=f'separated_{i}_{j}')
                    model.addConstr(separated == or_([in_between for in_between in in_between_objs]))
                    for c in range(len(self.attr_map['attributes']['color'])):
                        model.addConstr(attr_variables[f'color_{i}_{c}'] + attr_variables[f'color_{j}_{c}'] +
                                        (1 - separated) <= 2)

        # a large cube must have a small cylinder behind it
        if self.large_cube:
            for i in range(num_objects):
                small_cylinder_behinds = []
                for j in range(num_objects):
                    if i == j:
                        continue
                    behind = model.addVar(vtype=GRB.BINARY, name=f'small_cylinder_behind_{j}_{i}')
                    model.addConstr(behind == and_(attr_variables[f'size_{j}_1'], attr_variables[f'shape_{j}_2'],
                                                   rel_variables[f'behind_{j}_{i}']))
                    small_cylinder_behinds.append(behind)
                has_cylinder_behind = model.addVar(vtype=GRB.BINARY, name=f'has_small_cylinder_behind_{i}')
                model.addConstr(has_cylinder_behind == or_([behind for behind in small_cylinder_behinds]))
                model.addConstr(1 - (attr_variables[f'size_{i}_0'] + attr_variables[f'shape_{i}_1'] - 1) +
                                has_cylinder_behind >= 1)

        # If there are a cyan metal object and a red sphere object, then there cannot be anything behind them.
        if self.object_behind:
            last_two_objects = []
            cyan_metals = []
            red_spheres = []
            for i in range(num_objects):
                last_two_object = model.addVar(vtype=GRB.BINARY, name=f'last_two_objects_{i}')
                cyan_metal = model.addVar(vtype=GRB.BINARY, name=f'cyan_metal_{i}')
                red_sphere = model.addVar(vtype=GRB.BINARY, name=f'red_sphere_{i}')
                behinds = [rel_variables[f'behind_{j}_{i}'] for j in range(num_objects) if j != i]
                model.addConstr(gurobipy.quicksum(behinds) >= 2 - self.M * last_two_object)
                model.addConstr(gurobipy.quicksum(behinds) <= 1 + self.M * (1 - last_two_object))
                model.addConstr(cyan_metal == and_(attr_variables[f'color_{i}_{2}'],
                                                   attr_variables[f'material_{i}_{0}']))
                model.addConstr(red_sphere == and_(attr_variables[f'color_{i}_{6}'],
                                                   attr_variables[f'shape_{i}_{0}']))
                last_two_objects.append(last_two_object)
                cyan_metals.append(cyan_metal)
                red_spheres.append(red_sphere)

            post_conditions = []
            for i in range(num_objects):
                for j in range(num_objects):
                    if i == j:
                        continue
                    post_condition = model.addVar(vtype=GRB.BINARY, name=f'post_condition_{i}_{j}')
                    model.addConstr(post_condition == and_(cyan_metals[i], red_spheres[j],
                                                           last_two_objects[i], last_two_objects[j]))
                    post_conditions.append(post_condition)

            has_cyan_metal = model.addVar(vtype=GRB.BINARY, name=f'has_cyan_metal')
            has_red_sphere = model.addVar(vtype=GRB.BINARY, name=f'has_red_sphere')
            post_condition_met = model.addVar(vtype=GRB.BINARY, name=f'post_condition_met')
            model.addConstr(has_cyan_metal == or_([v for v in cyan_metals]))
            model.addConstr(has_red_sphere == or_([v for v in red_spheres]))
            model.addConstr(post_condition_met == or_([v for v in post_conditions]))
            # actual constraint
            model.addConstr(1 - (has_red_sphere + has_cyan_metal - 1) + post_condition_met >= 1)
