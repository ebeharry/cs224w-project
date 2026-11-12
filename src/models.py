"""A small relational GNN for bot detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.backend
from torch_geometric.nn import RGCNConv

from data import NUM_RELATIONS, PROFILE_COLS, TEXT_COLS

# PyG's segment_matmul path averages across relations; force the per-relation mean.
torch_geometric.backend.use_segment_matmul = False


class RelGNN(nn.Module):
    """Profile/text encoders -> 2 x RGCNConv -> linear head."""

    def __init__(self, hidden=128, num_relations=NUM_RELATIONS, dropout=0.3):
        super().__init__()
        self.register_buffer("profile_cols", torch.tensor(PROFILE_COLS))
        self.register_buffer("text_cols", torch.tensor(TEXT_COLS))
        self.profile_enc = nn.Linear(len(PROFILE_COLS), hidden // 2)
        self.text_enc = nn.Linear(len(TEXT_COLS), hidden // 2)
        self.conv1 = RGCNConv(hidden, hidden, num_relations, aggr="mean")
        self.conv2 = RGCNConv(hidden, hidden, num_relations, aggr="mean")
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        h = torch.cat([
            F.leaky_relu(self.profile_enc(x[:, self.profile_cols])),
            F.leaky_relu(self.text_enc(x[:, self.text_cols])),
        ], dim=1)
        h = F.dropout(h, self.dropout, self.training)
        for conv in (self.conv1, self.conv2):
            h = F.relu(conv(h, edge_index, edge_type))
            h = F.dropout(h, self.dropout, self.training)
        return self.head(h)


class MLP(nn.Module):
    """RelGNN with each RGCNConv replaced by a Linear layer (no graph)."""

    def __init__(self, hidden=128, dropout=0.3):
        super().__init__()
        self.register_buffer("profile_cols", torch.tensor(PROFILE_COLS))
        self.register_buffer("text_cols", torch.tensor(TEXT_COLS))
        self.profile_enc = nn.Linear(len(PROFILE_COLS), hidden // 2)
        self.text_enc = nn.Linear(len(TEXT_COLS), hidden // 2)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x):
        h = torch.cat([
            F.leaky_relu(self.profile_enc(x[:, self.profile_cols])),
            F.leaky_relu(self.text_enc(x[:, self.text_cols])),
        ], dim=1)
        h = F.dropout(h, self.dropout, self.training)
        for lin in (self.lin1, self.lin2):
            h = F.relu(lin(h))
            h = F.dropout(h, self.dropout, self.training)
        return self.head(h)
