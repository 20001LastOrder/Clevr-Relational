from .constraint_loss import get_dag_constraint, get_anti_symmetry_constraint
from . import build_adjacency_matrix
import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)


def test_anti_symmetry():
    node_size = 5

    network = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, node_size * node_size * 2),
        nn.Sigmoid()
    )

    constraints = [get_dag_constraint(1), get_anti_symmetry_constraint()]
    criteria = lambda x:  constraints[0](x) + constraints[1](x)

    x = torch.ones(1, 32)
    optimizer = torch.optim.SGD(params=network.parameters(), lr=0.01)

    last_loss = 0
    targets = torch.triu(torch.ones((2, node_size, node_size)), diagonal=1)
    loss_criteria = torch.nn.BCELoss()

    change = 10
    for i in range(1000):
        a = network(x).reshape(-1, 2)

        adj: torch.Tensor = build_adjacency_matrix(a, node_size)
        network.requires_grad = True
        loss = 0.1 * criteria(adj) / torch.numel(adj) + 0.9 * loss_criteria(adj, targets)
        loss.backward()

        change = abs(loss - last_loss)
        print(loss.item())
        optimizer.step()
        last_loss = loss.item()

    adj = build_adjacency_matrix(network(x).reshape(-1, 2), node_size)
    print(adj)
    # result = adj
    # cumulator = adj

    # for _ in range(1, adj.shape[-1]):
    #     cumulator = cumulator @ adj
    #     result = result + cumulator
    # print(result)


if __name__ == '__main__':
    test_anti_symmetry()