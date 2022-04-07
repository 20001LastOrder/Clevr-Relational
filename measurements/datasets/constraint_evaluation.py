from abc import ABC, abstractmethod
from typing import List, Dict


class SceneConstraint(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, scene) -> int:
        """
        evaluate the scene and output the number of constraint violations
        :param scene: scene to be evaluated
        :return: number of violations
        """
        pass


class ConstraintEvaluator:
    def __init__(self, constraints: List[SceneConstraint]):
        self.constraints = constraints

    def evaluate(self, scene) -> Dict[str, int]:
        """
        evaluate the scene and output a dict of constraint type to the number of violations of that constraint
        :param scene:
        :return:
        """
        return {constraint.name: constraint.evaluate(scene) for constraint in self.constraints}