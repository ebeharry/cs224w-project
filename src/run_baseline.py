"""Random forest on node features only (no graph, so cost = 0), over 5 seeds."""
from sklearn.ensemble import RandomForestClassifier

from data import load_mgtab
from evaluate import METRICS, evaluate, save_result, summarize
from splits import SEEDS, get_split


def main():
    data = load_mgtab(add_reverse=False)  # edges are not used
    x, y = data.x.numpy(), data.y.numpy()

    per_seed = []
    for seed in SEEDS:
        train, _, test = get_split(data.num_nodes, seed)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf.fit(x[train], y[train])
        metrics = evaluate(y[test], rf.predict(x[test]))
        per_seed.append(metrics)
        print(f"seed {seed}: " + "  ".join(f"{m}={metrics[m]:.4f}" for m in METRICS))

    summary = summarize(per_seed)
    summary["avg_cost"] = (0.0, 0.0)
    print("\nmean ± std: " + "  ".join(
        f"{m}={summary[m][0]:.4f}±{summary[m][1]:.4f}" for m in METRICS))
    save_result("RandomForest", "k=0 (no graph)", summary, len(SEEDS))


if __name__ == "__main__":
    main()
