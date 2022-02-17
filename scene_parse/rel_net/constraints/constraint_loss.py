from abc import ABC, abstractmethod
from typing import Callable
import torch
from torch.nn import functional


class ConstraintLoss(ABC):
    @abstractmethod
    def cal_loss(self, results: torch.Tensor) -> torch.Tensor:
        pass


class ThresholdConstraint(ConstraintLoss):
    """
    Express the loss for constraint value_cal(results) <= epsilon with max(value_cal(results) - epsilon, 0)
    """
    def __init__(self, epsilon: float, value_cal: Callable[[torch.Tensor], torch.Tensor], ignore_diag=False):
        self.epsilon = epsilon
        self.value_cal = value_cal
        self.ignore_diag = ignore_diag

    def cal_loss(self, results: torch.Tensor) -> torch.Tensor:
        if self.ignore_diag:
            return functional.relu(self.value_cal(results) - self.epsilon)\
                .mul(torch.triu(1 - torch.eye(results.shape[-1], device=results.device))).sum()
        else:
            return functional.relu(self.value_cal(results) - self.epsilon).sum()


def get_anti_symmetry_constraint(epsilon=-0.7):
    def anti_symmetry(adj: torch.Tensor):
        """
        :param adj: in shape (R, N, N) R is the number of relationships
        :return: a matrix of loss
        """
        return -(adj - adj.transpose(1, 2)).abs().mul(1 - torch.eye(adj.shape[-1], device=adj.device))
    return ThresholdConstraint(epsilon, anti_symmetry, ignore_diag=True)


def get_dag_constraint(epsilon=0):
    def dag(adj: torch.Tensor):
        """
        :param adj: in shape (R, N, N) R is the number of relationships
        :return: a matrix of loss
        """
        result = adj
        cumulator = adj

        for _ in range(1, adj.shape[-1]):
            cumulator = cumulator @ adj
            result = result + cumulator

        return result.mul(torch.eye(adj.shape[-1], device=result.device))

    return ThresholdConstraint(epsilon, dag)
