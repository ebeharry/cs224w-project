"""MLP on scaled node features only (no graph, so cost = 0), over 5 seeds."""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import load_mgtab, scale_features
from evaluate import METRICS, evaluate, save_result, summarize
from models import MLP
from run_gnn import BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY
from splits import SEEDS, get_split


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x).argmax(dim=1).numpy()


def run_one(x, y, seed, scaling):
    """Train on one split with early stopping on val F1, then evaluate on test."""
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = get_split(x.size(0), seed)
    x = scale_features(x, train_idx, scaling)

    train_loader = DataLoader(TensorDataset(x[train_idx], y[train_idx]),
                              batch_size=BATCH_SIZE, shuffle=True)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_f1 = evaluate(y[val_idx].numpy(), predict(model, x[val_idx]))["f1"]
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            best_state = {n: p.clone() for n, p in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)

    metrics = evaluate(y[test_idx].numpy(), predict(model, x[test_idx]))
    print(f"seed {seed}: stopped at epoch {epoch} (best val F1 {best_f1:.4f})  "
          + "  ".join(f"{m}={metrics[m]:.4f}" for m in METRICS), flush=True)
    return metrics


MODEL_NAMES = {"standard": "MLP", "text-minmax": "MLP (text minmax)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", choices=list(MODEL_NAMES), default="standard")
    args = parser.parse_args()

    data = load_mgtab(add_reverse=False)  # edges are not used
    per_seed = [run_one(data.x, data.y, seed, args.scaling) for seed in SEEDS]

    summary = summarize(per_seed)
    summary["avg_cost"] = (0.0, 0.0)
    print("\nmean ± std: " + "  ".join(
        f"{m}={summary[m][0]:.4f}±{summary[m][1]:.4f}" for m in METRICS))
    save_result(MODEL_NAMES[args.scaling], "k=0 (no graph)", summary, len(SEEDS))


if __name__ == "__main__":
    main()
