"""Split conformal prediction on a kNN hypergraph under structure corruption.

Baseline: MLP on clean hypergraph-smoothed features, LAC conformal calibrated on clean structure.
Improved: HistGradientBoosting trained on clean + corrupted-structure copies, conformal calibrated
on pooled clean + corrupted calibration structures.
Main metric: |coverage on corrupted test structure - (1 - alpha)| (lower is better).
"""
import json
import math
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=".*did not converge.*")
SEED = 42
SEEDS = [42, 43, 44]
K = 10          # hyperedge = node itself + K nearest neighbours
ALPHA = 0.1     # target coverage 1 - ALPHA
CORRUPT = 0.2   # fraction of hyperedge memberships replaced at random


def build_incidence(X):
    """(rows, cols) of incidence entries: hyperedge i = {i} + its K nearest neighbours."""
    n = len(X)
    idx = NearestNeighbors(n_neighbors=K + 2).fit(X).kneighbors(X, return_distance=False)
    ar = np.arange(n)
    # stable sort moves the self-match to the end, so the first K entries are neighbours
    order = np.argsort(idx == ar[:, None], axis=1, kind="stable")
    nb = np.take_along_axis(idx, order, axis=1)[:, :K]
    members = np.column_stack([ar, nb])
    return members.ravel(), np.repeat(ar, K + 1)


def make_H(rows, cols, n):
    """Binary sparse incidence matrix H (nodes x hyperedges)."""
    H = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    H.sum_duplicates()
    H.data[:] = 1.0
    return H


def corrupt(rows, cols, n, seed):
    """Replace a random CORRUPT fraction of memberships with uniformly random nodes."""
    rng = np.random.default_rng(seed)
    r = rows.copy()
    m = rng.random(len(r)) < CORRUPT
    r[m] = rng.integers(0, n, m.sum())
    return make_H(r, cols, n)


def smooth(H, X):
    """Hypergraph smoothing Dv^-1 H De^-1 H^T X, computed as two sparse products."""
    dv = np.asarray(H.sum(1)).ravel()
    de = np.asarray(H.sum(0)).ravel()
    dv[dv == 0] = 1
    de[de == 0] = 1
    return (H @ ((H.T @ X) / de[:, None])) / dv[:, None]


def features(H, Xs):
    """Concatenate node features with hypergraph-smoothed features."""
    return np.hstack([Xs, smooth(H, Xs)])


def lac_qhat(P, y):
    """Finite-sample (1 - ALPHA) quantile of LAC scores 1 - p(true class)."""
    s = 1 - P[np.arange(len(y)), y]
    n = len(s)
    level = min(math.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
    return np.quantile(s, level, method="higher")


def evaluate(P, y, qhat):
    """Return (coverage, mean set size, macro F1) for LAC prediction sets."""
    sets = (1 - P) <= qhat
    cov = sets[np.arange(len(y)), y].mean()
    return cov, sets.sum(1).mean(), f1_score(y, P.argmax(1), average="macro")


def run(seed, Xs, y, splits, rows, cols, n):
    """One full run of both methods for a given seed."""
    tr, cal, te = splits
    F_clean = features(make_H(rows, cols, n), Xs)
    F_tr_c = features(corrupt(rows, cols, n, seed + 2000), Xs)   # training corrupted copy
    F_cal_c = features(corrupt(rows, cols, n, seed + 3000), Xs)  # calibration corrupted copy
    F_te_c = features(corrupt(rows, cols, n, seed + 1000), Xs)   # test corrupted structure

    # Baseline: MLP on clean features; conformal calibrated on clean structure only
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed)
    mlp.fit(F_clean[tr], y[tr])
    q_b = lac_qhat(mlp.predict_proba(F_clean[cal]), y[cal])
    b_cov_c, b_size, b_f1 = evaluate(mlp.predict_proba(F_te_c[te]), y[te], q_b)
    b_cov_cl, _, _ = evaluate(mlp.predict_proba(F_clean[te]), y[te], q_b)

    # Improved: HGB on clean + corrupted training copies; conformal pooled over clean + corrupted cal
    hgb = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    hgb.fit(np.vstack([F_clean[tr], F_tr_c[tr]]), np.concatenate([y[tr], y[tr]]))
    P_cal = np.vstack([hgb.predict_proba(F_clean[cal]), hgb.predict_proba(F_cal_c[cal])])
    q_i = lac_qhat(P_cal, np.concatenate([y[cal], y[cal]]))
    i_cov_c, i_size, i_f1 = evaluate(hgb.predict_proba(F_te_c[te]), y[te], q_i)
    i_cov_cl, _, _ = evaluate(hgb.predict_proba(F_clean[te]), y[te], q_i)

    return dict(b_cov_c=b_cov_c, b_cov_cl=b_cov_cl, b_size=b_size, b_f1=b_f1,
                i_cov_c=i_cov_c, i_cov_cl=i_cov_cl, i_size=i_size, i_f1=i_f1)


def main():
    X, y = load_digits(return_X_y=True)
    n = len(y)
    idx = np.arange(n)
    # Split ONCE: 50% train / 25% calibration / 25% test, stratified
    tr, rest = train_test_split(idx, test_size=0.5, stratify=y, random_state=SEED)
    cal, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=SEED)
    print(f"rows={n} train={len(tr)} cal={len(cal)} test={len(te)}")
    Xs = StandardScaler().fit(X[tr]).transform(X)  # scaler fit on train only
    rows, cols = build_incidence(Xs)               # structure uses features only (no labels)
    print("hypergraph built")

    res = {}
    for s in SEEDS:
        print(f"seed {s}: running baseline and improved models")
        r = run(s, Xs, y, (tr, cal, te), rows, cols, n)
        r["b_score"] = abs(r["b_cov_c"] - (1 - ALPHA))
        r["i_score"] = abs(r["i_cov_c"] - (1 - ALPHA))
        for k, v in r.items():
            res.setdefault(k, []).append(float(v))
        print(f"  baseline gap={r['b_score']:.4f} improved gap={r['i_score']:.4f}")

    m = {k: float(np.mean(v)) for k, v in res.items()}
    metrics = {
        "baseline_score": m["b_score"], "improved_score": m["i_score"],
        "improvement": m["b_score"] - m["i_score"],
        "baseline_score_std": float(np.std(res["b_score"])),
        "improved_score_std": float(np.std(res["i_score"])),
        "baseline_corrupted_coverage": m["b_cov_c"], "improved_corrupted_coverage": m["i_cov_c"],
        "baseline_clean_coverage": m["b_cov_cl"], "improved_clean_coverage": m["i_cov_cl"],
        "baseline_set_size": m["b_size"], "improved_set_size": m["i_size"],
        "baseline_f1": m["b_f1"], "improved_f1": m["i_f1"],
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        print("non-finite metrics; failing")
        sys.exit(1)

    # Chart: corrupted-test coverage vs target
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cov_b = np.array(res["b_cov_c"]); cov_i = np.array(res["i_cov_c"])
    ax.bar(["baseline", "improved"], [cov_b.mean(), cov_i.mean()],
           yerr=[cov_b.std(), cov_i.std()], capsize=5, color=["gray", "tab:blue"])
    ax.axhline(1 - ALPHA, color="red", ls="--", label="target")
    ax.set_ylabel("coverage on corrupted test structure")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/coverage_corrupted.png", dpi=120)

    out = {
        "question": ("Can split conformal prediction calibrated on noise-perturbed hypergraph "
                     "structure keep its nominal coverage better than standard clean-calibrated "
                     "conformal prediction when the test hypergraph is corrupted?"),
        "method": ("kNN hypergraph on standardized digits features; node features concatenated with "
                   "hypergraph-smoothed features; split conformal (LAC) with alpha=0.1; "
                   "3 seeds, mean and std reported."),
        "dataset": {
            "name": "scikit-learn digits (load_digits)",
            "source_url": "https://scikit-learn.org/stable/datasets/toy_dataset.html",
            "n_rows": int(n), "n_train": int(len(tr)), "n_test": int(len(te)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "abs(coverage on corrupted test structure - 0.90)",
        "higher_is_better": False,
        "baseline": ("MLP (one hidden layer of 128) on clean hypergraph-smoothed features with "
                     "split conformal prediction calibrated on the clean hypergraph."),
        "improved": ("HistGradientBoosting trained on clean plus structure-corrupted copies "
                     "(20% of memberships replaced), conformal calibration pooled over clean and "
                     "corrupted calibration structures."),
        "metrics": metrics,
        "notes": ("Single stratified 50/25/25 split; test set used once per seed for final evaluation. "
                  "No hyper-parameter search was run, so no CV-based selection occurred. "
                  "Corruption is uniform random, not adversarial. Coverage is averaged over 3 seeds "
                  "which vary model init and corruption draws; the std is over these seeds. "
                  "Hypergraph is built transductively over all nodes using features only (no labels)."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
