from abc import ABC, abstractmethod
from typing import Callable, Tuple, Any
import torch
from torch.nn import functional


class ConstraintLoss(ABC):
    @abstractmethod
    def __call__(self, results: torch.Tensor) -> torch.Tensor:
        pass


class ThresholdConstraint(ConstraintLoss):
    """
    Express the loss for constraint value_cal(results) <= epsilon with max(value_cal(results) - epsilon, 0)
    """

    def __init__(self, epsilon: float, value_cal: Callable[[torch.Tensor], torch.Tensor], ignore_diag=False):
        self.epsilon = epsilon
        self.value_cal = value_cal
        self.ignore_diag = ignore_diag

    def __call__(self, results: torch.Tensor) -> torch.Tensor:
        error_results = self.value_cal(results)
        if self.ignore_diag:
            return functional.relu(error_results - self.epsilon) \
                .mul(torch.triu(1 - torch.eye(error_results.shape[-1], device=results.device))).sum()
        else:
            return functional.relu(error_results - self.epsilon).sum()


class InternalLossConstraint(ConstraintLoss):
    """
    This class calculate the internal loss for a batch of results. It assumes that value_cal gives back a two output
    values and use criteria to calculate a loss from the two values
    """

    def __init__(self, value_cal: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], criteria: Callable):
        self.value_cal = value_cal
        self.criteria = criteria

    def __call__(self, results: torch.Tensor) -> torch.Tensor:
        value1, value2 = self.value_cal(results)
        return self.criteria(value1, value2)


def get_anti_symmetry_constraint(epsilon=-0.5) -> ConstraintLoss:
    def anti_symmetry(adj: torch.Tensor):
        """
        :param adj: in shape (R, N, N) R is the number of relationships
        :return: return: a matrix of loss
        """
        return -(adj - adj.transpose(1, 2)).abs()

    return ThresholdConstraint(epsilon, anti_symmetry, ignore_diag=True)


def get_deduct_constraint(dimension_pairs, epsilon=-0.5):
    def deduct(adj: torch.Tensor):
        deduct_pairs = []
        for i, j in dimension_pairs:
            deduct_pairs.append(-(adj[i] - adj[j]).abs())
        return torch.stack(deduct_pairs)
    return ThresholdConstraint(epsilon, deduct, ignore_diag=True)


def get_dag_constraint(epsilon=0) -> ConstraintLoss:
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


def get_transitivity_constraint(epsilon=0.5, expected=1):
    def transitivity(adj: torch.Tensor):
        adj_sq = adj @ adj
        return torch.masked_select(torch.relu(-adj + epsilon), adj_sq >= expected).sum()

    return transitivity
