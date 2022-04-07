from abc import ABC, abstractmethod

from blocks import State, Block


def has_small_below(state: State, obj: Block):
    for obj in state.objects_below(obj):
        if obj.size == 'small':
            return True
    return False


def get_constraint_map():
    return {
        'small_stack': StackConstraint(),
        'large_cube': LargeCubeConstraint(),
        'neighbor_stack': NeighborConstraint(),
        'yellow_object': YellowConstraint()
    }


def get_clevr_constraint_map():
    return {
        'same_color': ClevrSameColorConstraint(),
        'material': ClevrMaterialConstraint(),
        'large_cube': ClevrLargeCubeConstraint(),
        'pair_behind': ClevrPairBehindConstraint(),
        'identity': ClevrObjectIdentityConstraint()
    }


class SceneConstraint(ABC):
    @abstractmethod
    def evaluate(self, state):
        """
        Evaluate the constraint for the scene.
        :param state: state to be evaluated
        :return: whether the constraint is satisfied. True means satisfied. False means not.
        """
        pass


class StackConstraint(SceneConstraint):
    """
    An object cannot hold another object larger than it
    """

    def evaluate(self, state: State):
        for obj in state.objects:
            if obj.size == 'large' and has_small_below(state, obj):
                return False
        return True


class LargeCubeConstraint(SceneConstraint):
    """
    A large cube has to hold something on top of it
    """

    def evaluate(self, state: State):
        for obj in state.objects:
            if obj.size == 'large' and len(state.objects_above(obj)) == 0:
                return False
        return True


class NeighborConstraint(SceneConstraint):
    """
    Two neighboring stacks bottom of the stacks cannot have the same color
    """

    def evaluate(self, state: State):
        is_bottoms = [state.is_bottom(o) for o in state.objects]
        n = len(state.objects)

        for i in range(n):
            for j in range(i + 1, n):
                if is_bottoms[i] and is_bottoms[j] and state.is_direct_next(state.objects[i], state.objects[j]) \
                        and state.objects[i].color == state.objects[j].color:
                    return False
        return True


class YellowConstraint(SceneConstraint):
    """
    For each yellow object there has to have something on the right
    """

    def evaluate(self, state: State):
        for obj in state.right_most_objects():
            if obj.color == 'yellow':
                return False
        return True


class ClevrSameColorConstraint(SceneConstraint):
    """
    Between any pairs of objects with the same color, there must be an object with a different color placed in between horizontally.
    """

    def __exist_object_inbetween(self, scene, o1, o2):
        o1_idx, o2_idx = o1['id'], o2['id']
        left = o1_idx if o1_idx in scene['relationships']['left'][o2_idx] else o2_idx
        right = o2_idx if o1_idx in scene['relationships']['left'][o2_idx] else o1_idx

        # check all objects in between
        for o_idx in set(scene['relationships']['right'][left]).intersection(
                set(scene['relationships']['left'][right])):
            if scene['objects'][o_idx]['color'] != o1['color']:
                return True
        return False

    def evaluate(self, scene):
        objects = scene['objects']
        n = len(objects)
        for i in range(n):
            o1 = objects[i]
            for j in range(i + 1, n):
                o2 = objects[j]
                if o1['color'] == o2['color'] and not self.__exist_object_inbetween(scene, o1, o2):
                    return False
        return True


class ClevrMaterialConstraint(SceneConstraint):
    """
    A metal object cannot be behind a rubber object
    """

    def __rubber_front(self, scene, o1):
        objects_front = scene['relationships']['front'][o1['id']]
        for o_idx in objects_front:
            if scene['objects'][o_idx]['material'] == 'rubber':
                return True
        return False

    def evaluate(self, scene):
        for o in scene['objects']:
            if o['material'] == 'metal' and self.__rubber_front(scene, o):
                return False
        return True


class ClevrLargeCubeConstraint(SceneConstraint):
    """
    For each large cube, there must be a small cylinder behind it
    """

    def __exist_small_cylinder_behind(self, scene, o1):
        objects_behind = scene['relationships']['behind'][o1['id']]
        for o_idx in objects_behind:
            if scene['objects'][o_idx]['shape'] == 'cylinder' and scene['objects'][o_idx]['size'] == 'small':
                return True
        return False

    def evaluate(self, scene):
        for o in scene['objects']:
            if o['size'] == 'large' and o['shape'] == 'cube' and not self.__exist_small_cylinder_behind(scene, o):
                return False
        return True


class ClevrPairBehindConstraint(SceneConstraint):
    """
    If there are a cyan metal object and a red sphere object, then there cannot be anything behind them
    """

    def evaluate(self, scene):
        cyan_metal = None
        red_sphere = None
        behind_rel = scene['relationships']['behind']

        for o in scene['objects']:
            if o['color'] == 'cyan' and o['material'] == 'metal':
                cyan_metal = o
            if o['color'] == 'red' and o['shape'] == 'sphere':
                red_sphere = o
        if cyan_metal is not None and red_sphere is not None and \
                (len(behind_rel[cyan_metal['id']]) > 1 or len(behind_rel[red_sphere['id']]) > 1):
            return False
        else:
            return True


class ClevrObjectIdentityConstraint(SceneConstraint):
    """
    There cannot be two identical objects
    """

    def evaluate(self, scene):
        objects = scene['objects']
        n = len(objects)
        for i in range(n):
            o1 = objects[i]
            for j in range(i + 1, n):
                o2 = objects[j]
                if o1['color'] == o2['color'] and o1['material'] == o2['material'] and o1['shape'] == o2['shape'] and \
                        o1['size'] == o2['size']:
                    return False
        return True
