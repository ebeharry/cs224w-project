"""
This sample code is taken directly from https://github.com/GraphDetec/MGTAB/tree/main
"""
import torch


def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)