"""Train/val/test splits: 70/20/10 over all nodes, exactly as in MGTAB-ML.py."""
import numpy as np
import torch
from sklearn.utils import shuffle

SEEDS = [0, 1, 2, 3, 4]


def get_split(num_nodes, seed):
    """Shuffle node ids with the given seed and cut them 70/20/10."""
    idx = shuffle(np.arange(num_nodes), random_state=seed)
    train = idx[:int(0.7 * num_nodes)]
    val = idx[int(0.7 * num_nodes):int(0.9 * num_nodes)]
    test = idx[int(0.9 * num_nodes):]
    return tuple(torch.from_numpy(np.sort(part)) for part in (train, val, test))
