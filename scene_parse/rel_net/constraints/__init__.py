import torch
from typing import List
from .constraint_loss import ConstraintLoss


def build_adjacency_matrix(edge_list, num_nodes):
    """
    create an adjacency matrix from a list of edges of a scene
    :param edge_list: a list of edges of a scene in shape (N*N, R) where N is the number of nodes and R
    is the number of relationship types
    :param num_nodes number of nodes in the scene graph (N)
    :return: Adjacency matrices with shape (R, N, N)
    """
    return edge_list.T.view(-1, num_nodes, num_nodes)


def build_adjacency_matrix_from_edge_list(sources, targets, preds, num_nodes):
    adj = torch.zeros(preds.shape[-1], num_nodes, num_nodes)
    for i, (source, target) in enumerate(zip(sources, targets)):
        adj[:, source, target] = preds[i]
    return adj


def adj_probability(adj):
    """
    Given an adjacency matrix of logits, construct the probabilities of each edge
    :param adj:
    :return: adj matrix with probability
    """
    logits_diff = (adj - adj.transpose(1, 2))
    return torch.sigmoid(logits_diff).mul(1 - torch.eye(adj.shape[-1], device=adj.device))


def get_logic_loss(edge_list, num_nodes, constraints: List[ConstraintLoss]):
    adjacency = build_adjacency_matrix(edge_list, num_nodes)

    loss = sum([constraint.cal_loss(adjacency) for constraint in constraints])
    return loss

