"""Check RGCNConv(aggr="mean") against a hand-written per-relation mean."""
import os
import sys

import pytest
import torch
import torch_geometric.backend
import torch_geometric.typing
from torch_geometric.nn import RGCNConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import models  # noqa: E402,F401  (importing it applies our fix)

NUM_RELATIONS = 3


def tiny_graph():
    """5 nodes, 3 relations, with nodes that have unequal per-relation degrees."""
    torch.manual_seed(0)
    x = torch.randn(5, 4)
    edges = [  # (src, dst, relation)
        (1, 0, 0), (4, 0, 1), (2, 0, 0), (3, 0, 0),
        (0, 1, 2), (2, 1, 0), (3, 1, 2),
        (4, 2, 1),
        (1, 3, 1), (0, 3, 1), (2, 3, 2),
    ]
    edge_index = torch.tensor([[s for s, _, _ in edges], [d for _, d, _ in edges]])
    edge_type = torch.tensor([r for _, _, r in edges])
    return x, edge_index, edge_type


def reference_rgcn(conv, x, edge_index, edge_type):
    """x_i' = x_i W_root + bias + sum_r mean_{j in N_r(i)} x_j W_r, by hand."""
    out = x @ conv.root + conv.bias
    for i in range(x.size(0)):
        for r in range(NUM_RELATIONS):
            nbrs = edge_index[0][(edge_index[1] == i) & (edge_type == r)]
            if len(nbrs):
                out[i] += x[nbrs].mean(dim=0) @ conv.weight[r]
    return out


@pytest.fixture
def setup():
    x, edge_index, edge_type = tiny_graph()
    conv = RGCNConv(4, 6, NUM_RELATIONS, aggr="mean")
    with torch.no_grad():
        expected = reference_rgcn(conv, x, edge_index, edge_type)
    return conv, x, edge_index, edge_type, expected


def run_conv(conv, x, edge_index, edge_type, use_segment_matmul, is_sorted):
    """Run conv with the given backend setting, restoring it afterwards."""
    saved = torch_geometric.backend.use_segment_matmul
    torch_geometric.backend.use_segment_matmul = use_segment_matmul
    conv.is_sorted = is_sorted
    if is_sorted:  # is_sorted=True promises edges are already sorted by type
        perm = edge_type.argsort(stable=True)
        edge_index, edge_type = edge_index[:, perm], edge_type[perm]
    try:
        with torch.no_grad():
            return conv(x, edge_index, edge_type)
    finally:
        torch_geometric.backend.use_segment_matmul = saved


def test_models_py_disables_segment_matmul():
    assert torch_geometric.backend.use_segment_matmul is False


@pytest.mark.parametrize("is_sorted", [False, True])
def test_per_relation_path_matches_reference(setup, is_sorted):
    """The setting models.py uses gives the documented per-relation mean."""
    conv, x, edge_index, edge_type, expected = setup
    out = run_conv(conv, x, edge_index, edge_type, False, is_sorted)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch_geometric.typing.WITH_SEGMM,
                    reason="segment_matmul needs pyg-lib")
@pytest.mark.parametrize("is_sorted", [False, True])
def test_segment_matmul_path_does_not_match_reference(setup, is_sorted):
    """Documents the PyG issue: the fast kernel averages across relations."""
    conv, x, edge_index, edge_type, expected = setup
    out = run_conv(conv, x, edge_index, edge_type, True, is_sorted)
    assert not torch.allclose(out, expected, atol=1e-4)

    all_nbr_mean = x @ conv.root + conv.bias
    with torch.no_grad():
        for i in range(x.size(0)):
            mask = edge_index[1] == i
            if mask.any():
                msgs = torch.stack([x[j] @ conv.weight[r] for j, r in
                                    zip(edge_index[0][mask], edge_type[mask])])
                all_nbr_mean[i] += msgs.mean(dim=0)
    torch.testing.assert_close(out, all_nbr_mean)
