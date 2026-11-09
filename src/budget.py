"""Neighborhood budgets: a model may only see k hops, f sampled neighbors per hop."""
from torch_geometric.loader import NeighborLoader

ALL = -1  # NeighborLoader convention: take every neighbor

BUDGETS = [(1, 10), (2, 10), (2, ALL)]


def budget_name(k, f):
    return f"k={k}, f={'all' if f == ALL else f}"


def make_loader(data, nodes, k, f, batch_size, shuffle=False):
    """NeighborLoader that samples f neighbors per node at each of k hops."""
    return NeighborLoader(data, num_neighbors=[f] * k, input_nodes=nodes,
                          batch_size=batch_size, shuffle=shuffle)


def fetch_cost(batch):
    """Distinct accounts fetched for a single target node (batch_size=1)."""
    assert batch.batch_size == 1
    return batch.num_nodes - 1
