import pulp


def get_clevr_block_world_constraint(prob, scene, rel_variables, attr_variables, attr_map, M=100,
                                     include_small_obj=False, include_large_cube=False, include_bottom_stack=False):
    num_objects = len(scene['objects'])
    objects = scene['objects']

    for attr in attr_map['attributes']:
        for i in range(num_objects):
            prob += pulp.lpSum([attr_variables[f'{attr}_{i}_{v}'] for v, _ in enumerate(objects[i][attr])]) == 1

    # pairwise constraints
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                # choose exactly one from left, right, above and below
                prob += rel_variables[f'above_{i}_{j}'] + rel_variables[f'below_{i}_{j}'] + rel_variables[
                    f'left_{i}_{j}'] + rel_variables[f'right_{i}_{j}'] == 1

                # left imply right, above imply below and vice versa
                prob += rel_variables[f'above_{i}_{j}'] == rel_variables[f'below_{j}_{i}']
                prob += rel_variables[f'left_{i}_{j}'] == rel_variables[f'right_{j}_{i}']

                if include_small_obj:
                    prob += 1 - (attr_variables[f'size_{j}_1'] + rel_variables[f'above_{i}_{j}'] - 1) + \
                        attr_variables[f'size_{i}_1'] >= 1

    if include_bottom_stack:
        next_to = {}
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            # no opposite relationship of the same type
            for rel in attr_map['relations']:
                prob += rel_variables[f'{rel}_{i}_{j}'] + rel_variables[f'{rel}_{j}_{i}'] <= 1

            # if include_bottom_stack:
            #     # TODO: Continue here
            #     v = pulp.LpVariable(f'next_to_{i}_{j}')
            #     prob += rel_variables[f'left_{i}_{j}'] +

    # triple constraints
    for i in range(num_objects):
        for j in range(num_objects):
            for k in range(num_objects):
                if i == k or j == k or i == j:
                    continue
                for rel in attr_map['relations']:
                    prob += rel_variables[f'{rel}_{i}_{j}'] + rel_variables[f'{rel}_{j}_{k}'] - rel_variables[
                        f'{rel}_{i}_{k}'] <= 1

                # objects on top of each other should have  the same left / right relations
                prob += rel_variables[f'above_{i}_{j}'] + rel_variables[f'left_{i}_{k}'] - rel_variables[
                    f'left_{j}_{k}'] <= 1

                prob += rel_variables[f'below_{i}_{j}'] + rel_variables[f'left_{i}_{k}'] - rel_variables[
                    f'left_{j}_{k}'] <= 1

    if include_bottom_stack:
        is_bottom = []

    for i in range(num_objects):
        dummy = pulp.LpVariable(f'd_any_above_{i}')
        prob += pulp.lpSum([rel_variables[f'above_{j}_{i}'] for j in range(num_objects) if i != j]) >= 1 - M * (
                1 - dummy)
        prob += pulp.lpSum([rel_variables[f'above_{j}_{i}'] for j in range(num_objects) if i != j]) <= M * dummy
        prob += 2 - attr_variables[f'shape_{i}_0'] - dummy >= 1
        if include_large_cube:
            prob += 1 - (attr_variables[f'size_{i}_0'] + attr_variables[f'shape_{i}_1'] - 1) + dummy >= 1

        if include_bottom_stack:
            v = pulp.LpVariable(f'bottom_{i}')
            prob += pulp.lpSum([rel_variables[f'below_{j}_i'] for j in range(num_objects) if i != j]) <= M * (1 - v)
            prob += pulp.lpSum([rel_variables[f'below_{j}_i'] for j in range(num_objects) if i != j]) >= 1 - M * v
            is_bottom.append(v)
