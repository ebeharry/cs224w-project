"""Plot balanced accuracy (mean ± std) against average fetch cost (log scale)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES = "#2a78d6"     # GNN points
REFERENCE = "#52514e"  # random-forest reference line
TEXT = "#0b0b0b"
MUTED = "#8a8985"


def main():
    df = pd.read_csv("results/results.csv")
    rf = df[df.model == "RandomForest"].iloc[0]
    gnn = df[df.model == "RelGNN"].sort_values("avg_cost_mean")

    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(4.6, 2.7))

    rf_mean, rf_std = 100 * rf.balanced_accuracy_mean, 100 * rf.balanced_accuracy_std
    ax.axhspan(rf_mean - rf_std, rf_mean + rf_std, color=REFERENCE, alpha=0.08, lw=0)
    ax.axhline(rf_mean, color=REFERENCE, lw=1.2, ls=(0, (4, 3)))
    ax.text(5, rf_mean - 0.25, "random forest (no graph, cost 0)",
            color=REFERENCE, fontsize=8, va="top")

    x = gnn.avg_cost_mean
    y = 100 * gnn.balanced_accuracy_mean
    err = 100 * gnn.balanced_accuracy_std
    ax.errorbar(x, y, yerr=err, color=SERIES, lw=2, marker="o", ms=6,
                mec="white", mew=1.5, capsize=3, elinewidth=1.2, zorder=3)
    for xi, yi, budget in zip(x, y, gnn.budget):
        ax.annotate(budget, (xi, yi), xytext=(6, 7), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8, color=TEXT)

    ax.set_xscale("log")
    ax.set_xlim(4, 2e4)
    ax.set_ylim(85, 91.5)
    ax.set_xlabel("Average accounts fetched per target (log scale)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("results/budget_vs_cost.pdf")


if __name__ == "__main__":
    main()
