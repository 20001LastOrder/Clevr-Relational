from typing import Dict, List
from .constraint_evaluation import SceneConstraint


def get_objects(scene: Dict, attributes: Dict[str, str]) -> List[int]:
    result = []

    for i, obj in enumerate(scene['objects']):
        if all([obj[attr] == value for attr, value in attributes.items()]):
            result.append(i)
    return result


def get_related_objects(scene: Dict, obj_id: int, relation_name: str) -> List[int]:
    return scene['relationships'][relation_name][obj_id]


def get_bottom_objects(scene: Dict):
    result = []
    for i, obj in enumerate(scene['objects']):
        if len(get_related_objects(scene, i, 'below')) == 0:
            result.append(i)
    return result


def next_to(scene: Dict, obj1: int, obj2: int):
    relationships = scene['relationships']['left']
    left_obj = obj1 if obj1 in relationships[obj2] else obj2
    right_obj = obj2 if obj1 in relationships[obj2] else obj1

    for i, _ in enumerate(scene['objects']):
        if i in relationships[right_obj] and left_obj in relationships[i]:
            return False

    return True


class SmallObjectConstraint(SceneConstraint):
    """
    A small object cannot hold a large object on top of it
    """

    def evaluate(self, scene) -> int:
        count = 0
        small_objects = get_objects(scene, {'size': 'small'})
        for obj_id in small_objects:
            for above_obj_id in get_related_objects(scene, obj_id, 'above'):
                if scene['objects'][above_obj_id]['size'] == 'large':
                    count += 1
        return count


class LargeCubeConstraint(SceneConstraint):
    """
    A large cube must have something on it
    """

    def evaluate(self, scene) -> int:
        count = 0
        objects = get_objects(scene, {'size': 'large', 'shape': 'cube'})
        for obj_id in objects:
            if len(get_related_objects(scene, obj_id, 'above')) == 0:
                count += 1
        return count


class SphereConstraint(SceneConstraint):
    """
    A sphere cannot have anything on top of it
    """

    def evaluate(self, scene) -> int:
        count = 0
        objects = get_objects(scene, {'shape': 'sphere'})
        for obj_id in objects:
            if len(get_related_objects(scene, obj_id, 'above')) > 0:
                count += 1
        return count


class BottomConstraint(SceneConstraint):
    """
    Two bottom objects next to each other cannot have the same color
    """
    def evaluate(self, scene) -> int:
        count = 0
        objects = get_bottom_objects(scene)
        n = len(objects)
        for i in range(n):
            idx1 = objects[i]
            for j in range(i + 1, n):
                idx2 = objects[j]
                if next_to(scene, idx1, idx2) and scene['objects'][idx1]['color'] == scene['objects'][idx2]['color']:
                    count += 1

        return count


class YellowObjectConstraint(SceneConstraint):
    """
    A yellow object must have something on its right
    """
    def evaluate(self, scene) -> int:
        count = 0
        objects = get_objects(scene, {'color': 'yellow'})
        for obj_id in objects:
            if len(get_related_objects(scene, obj_id, 'right')) == 0:
                count += 1
        return count
