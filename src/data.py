"""Load MGTAB into a PyG Data object."""
import torch
from torch_geometric.data import Data

from dataset import MGTAB

DATA_DIR = "data/mgtab"

RELATIONS = ["followers", "friends", "mention", "reply", "quoted", "URL", "hashtag"]
NUM_RELATIONS = len(RELATIONS)

# 20 profile features in [0, 1], then the 768-dim LaBSE text embedding.
PROFILE_COLS = list(range(0, 20))
TEXT_COLS = list(range(20, 788))


def inspect_feature_layout(x):
    """Find which columns are profile features vs. text embedding from value ranges."""
    col_min, col_max = x.min(dim=0).values, x.max(dim=0).values
    in_unit = ((col_min >= 0) & (col_max <= 1)).nonzero().flatten().tolist()
    other = ((col_min < 0) | (col_max > 1)).nonzero().flatten().tolist()
    num_binary = sum(len(torch.unique(x[:, j])) == 2 for j in in_unit)
    num_negative = int((col_min[other] < 0).sum())

    print(f"Columns in [0, 1] (profile): {len(in_unit)} -> {in_unit[0]}..{in_unit[-1]}"
          f"  ({num_binary} binary)")
    print(f"Columns outside [0, 1] (text): {len(other)} -> {other[0]}..{other[-1]}"
          f"  ({num_negative} with negative values)")
    assert in_unit == PROFILE_COLS, "profile columns are not 0..19 as assumed"
    assert other == TEXT_COLS, "text columns are not 20..787 as assumed"

    zero_text = (x[:, TEXT_COLS] == 0).all(dim=1)
    print(f"Users with an all-zero text embedding: {int(zero_text.sum())}")
    return zero_text


def load_mgtab(data_dir=DATA_DIR, add_reverse=True):
    """Return a PyG Data object with x, edge_index, edge_type and y (bot labels)."""
    raw = MGTAB(data_dir)[0]  # y1 = stance labels, y2 = bot labels
    x, edge_index, edge_type, y = raw.x, raw.edge_index, raw.edge_type, raw.y2

    if add_reverse:
        not_loop = edge_index[0] != edge_index[1]
        reverse = edge_index[:, not_loop].flip(0)
        edge_index = torch.cat([edge_index, reverse], dim=1)
        edge_type = torch.cat([edge_type, edge_type[not_loop]])

    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)


def print_stats(data):
    num_bots = int((data.y == 1).sum())
    num_humans = int((data.y == 0).sum())
    print(f"Nodes: {data.num_nodes}   features: {data.num_features}")
    print(f"Bots: {num_bots} ({num_bots / data.num_nodes:.1%})   Humans: {num_humans}")
    print(f"Edges: {data.num_edges:,}")
    counts = torch.bincount(data.edge_type, minlength=NUM_RELATIONS)
    for r, name in enumerate(RELATIONS):
        print(f"  {r} {name:<10} {int(counts[r]):>9,}")


if __name__ == "__main__":
    raw = load_mgtab(add_reverse=False)
    zero_text = inspect_feature_layout(raw.x)
    print(f"  of which bots: {int(raw.y[zero_text].sum())}")
    print("\n-- Original edges --")
    print_stats(raw)
    loops = raw.edge_index[0] == raw.edge_index[1]
    print(f"Self-loops: {int(loops.sum()):,} (per relation: "
          f"{torch.bincount(raw.edge_type[loops], minlength=NUM_RELATIONS).tolist()})")
    print("\n-- With reverse edges --")
    print_stats(load_mgtab())
