from .constraint_evaluation import SceneConstraint


class SameColorConstraint(SceneConstraint):
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
        count = 0
        for i in range(n):
            o1 = objects[i]
            for j in range(i + 1, n):
                o2 = objects[j]
                if o1['color'] == o2['color'] and not self.__exist_object_inbetween(scene, o1, o2):
                    count += 1
        return count


class MaterialConstraint(SceneConstraint):
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
        count = 0
        for o in scene['objects']:
            if o['material'] == 'metal' and self.__rubber_front(scene, o):
                count += 1
        return count


class LargeCubeConstraint(SceneConstraint):
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
        count = 0
        for o in scene['objects']:
            if o['size'] == 'large' and o['shape'] == 'cube' and not self.__exist_small_cylinder_behind(scene, o):
                count += 1
        return count


class PairBehindConstraint(SceneConstraint):
    """
    If there are a cyan metal object and a red sphere object, then there cannot be anything behind them
    """

    def evaluate(self, scene):
        cyan_metals = []
        red_spheres = []
        behind_rel = scene['relationships']['behind']

        for o in scene['objects']:
            if o['color'] == 'cyan' and o['material'] == 'metal':
                cyan_metals.append(o)
            if o['color'] == 'red' and o['shape'] == 'sphere':
                red_spheres.append(o)

        count = 0
        for cyan_metal in cyan_metals:
            for red_sphere in red_spheres:
                if len(behind_rel[cyan_metal['id']]) <= 1 and len(behind_rel[red_sphere['id']]) <= 1:
                    return 0
        return len(cyan_metals) != 0 and len(red_spheres) != 0


class ObjectIdentityConstraint(SceneConstraint):
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