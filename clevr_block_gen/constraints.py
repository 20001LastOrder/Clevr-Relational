from abc import ABC, abstractmethod

from clevr_block_gen.blocks import State, Block


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


class SceneConstraint(ABC):
    @abstractmethod
    def evaluate(self, state: State):
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
