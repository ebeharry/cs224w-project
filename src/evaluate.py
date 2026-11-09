"""Classification metrics and the shared results table."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1",
           "macro_precision", "macro_recall", "macro_f1"]
RESULTS_CSV = "results/results.csv"
RESULTS_MD = "results/results.md"


def evaluate(y_true, y_pred):
    """Return a dict with every metric in METRICS."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def summarize(per_seed):
    """Turn a list of per-seed metric dicts into {metric: (mean, std)}."""
    return {m: (np.mean([r[m] for r in per_seed]), np.std([r[m] for r in per_seed]))
            for m in per_seed[0]}


def save_result(model, budget, summary, num_seeds):
    """Insert or replace one row of results.csv and regenerate results.md."""
    row = {"model": model, "budget": budget, "seeds": num_seeds}
    for m in METRICS + ["avg_cost"]:
        row[f"{m}_mean"], row[f"{m}_std"] = summary[m]

    df = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    if len(df):
        df = df[~((df.model == model) & (df.budget == budget))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    write_markdown(df)


def write_markdown(df):
    pm = lambda r, m: f"{100 * r[f'{m}_mean']:.2f} ± {100 * r[f'{m}_std']:.2f}"
    lines = ["| Model | Budget | Seeds | Accuracy | Balanced acc. | Bot F1 | Macro F1 | Avg. cost |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.model} | {r.budget} | {r.seeds} | {pm(r, 'accuracy')} | "
                     f"{pm(r, 'balanced_accuracy')} | {pm(r, 'f1')} | {pm(r, 'macro_f1')} | "
                     f"{r.avg_cost_mean:.1f} |")
    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
