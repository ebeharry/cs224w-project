"""Train and evaluate RelGNN under each neighborhood budget."""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from budget import BUDGETS, budget_name, fetch_cost, make_loader
from data import SCALINGS, load_mgtab, scale_features
from evaluate import METRICS, evaluate, save_result, summarize
from models import RelGNN
from splits import SEEDS, get_split

BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 100
PATIENCE = 10  # epochs without val-F1 improvement before stopping
MODEL_NAMES = {"none": "RelGNN", "standard": "RelGNN (scaled)",
               "text-minmax": "RelGNN (text minmax)"}


def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_type)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, loader, record_cost=False):
    """Predict the seed nodes of every batch; optionally record fetch cost."""
    model.eval()
    preds, labels, costs = [], [], []
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.edge_type)[:batch.batch_size]
        preds.append(out.argmax(dim=1))
        labels.append(batch.y[:batch.batch_size])
        if record_cost:
            costs.append(fetch_cost(batch))
    return torch.cat(labels).numpy(), torch.cat(preds).numpy(), costs


def run_one(data, k, f, seed, scaling="none"):
    """Train on one split with early stopping on val F1, then test node by node."""
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = get_split(data.num_nodes, seed)
    if scaling != "none":
        data = copy.copy(data)
        data.x = scale_features(data.x, train_idx, scaling)
    train_loader = make_loader(data, train_idx, k, f, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(data, val_idx, k, f, BATCH_SIZE)

    model = RelGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer)
        val_f1 = evaluate(*predict(model, val_loader)[:2])["f1"]
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            best_state = {n: p.clone() for n, p in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)

    test_loader = make_loader(data, test_idx, k, f, batch_size=1)
    y_true, y_pred, costs = predict(model, test_loader, record_cost=True)
    metrics = evaluate(y_true, y_pred)
    metrics["avg_cost"] = float(np.mean(costs))
    print(f"  seed {seed}: stopped at epoch {epoch} (best val F1 {best_f1:.4f})  "
          + "  ".join(f"{m}={metrics[m]:.4f}" for m in METRICS)
          + f"  avg_cost={metrics['avg_cost']:.1f}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--budget", type=int, nargs=2, metavar=("K", "F"))
    parser.add_argument("--scaling", choices=SCALINGS, default="none")
    args = parser.parse_args()
    budgets = [tuple(args.budget)] if args.budget else BUDGETS
    model_name = MODEL_NAMES[args.scaling]

    data = load_mgtab()
    for k, f in budgets:
        name = budget_name(k, f)
        print(f"\n=== {model_name}, budget {name} ===", flush=True)
        start = time.time()
        per_seed = [run_one(data, k, f, seed, args.scaling) for seed in args.seeds]
        summary = summarize(per_seed)
        print(f"  mean: " + "  ".join(f"{m}={summary[m][0]:.4f}±{summary[m][1]:.4f}"
                                      for m in METRICS + ["avg_cost"])
              + f"   ({time.time() - start:.0f}s)", flush=True)
        save_result(model_name, name, summary, len(args.seeds))


if __name__ == "__main__":
    main()
