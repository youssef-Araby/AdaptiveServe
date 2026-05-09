"""C7: prompt-feature router.

Trains a classifier to predict the best KV-cache config per prompt.
Evaluates with leave-one-task-out cross-validation so we measure
generalization to *unseen* tasks (not just unseen prompts within
known tasks).

Compares classifier against:
  - always-C0 (no compression)
  - always-Cx for the best fixed config
  - per-prompt oracle (upper bound)

Reports mean quality and harmonic-mean compression for each policy.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import harmonic_mean, mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
FEATURE_KEYS = [
    "seq_len_tokens",
    "seq_len_chars",
    "token_entropy",
    "gzip_ratio",
    "unique_token_ratio",
    "question_position",
    "newline_density",
]
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]


def load_dataset(model: str):
    path = REPO / "runs" / "dataset" / f"{model}.jsonl"
    return [json.loads(l) for l in open(path)]


def featurize(rows):
    X = np.zeros((len(rows), len(FEATURE_KEYS)), dtype=np.float32)
    for i, r in enumerate(rows):
        f = r["features"]
        for j, k in enumerate(FEATURE_KEYS):
            v = f.get(k)
            # question_position can be None when no '?' present; impute 0.0
            X[i, j] = -1.0 if v is None else float(v)
    return X


def policy_metrics(rows, picks):
    """Given a per-prompt list of chosen configs, return (mean quality, hmean compression, pick counts)."""
    qs = []
    crs = []
    for r, c in zip(rows, picks):
        qs.append(r["scores"][c])
        crs.append(r["compression"][c])
    return mean(qs), harmonic_mean(crs), Counter(picks)


def oracle_iso_quality(rows, tau: float):
    picks = []
    for r in rows:
        c0 = r["scores"]["C0"]
        thresh = tau * c0
        best_c, best_cr = "C0", r["compression"]["C0"]
        for c in CONFIGS:
            if r["scores"][c] >= thresh and r["compression"][c] > best_cr:
                best_c, best_cr = c, r["compression"][c]
        picks.append(best_c)
    return picks


def loto_eval(rows, X, y, tau: float):
    """Leave-one-task-out CV. Returns predicted picks aligned with rows."""
    tasks = sorted({r["task"] for r in rows})
    preds = [None] * len(rows)
    for held in tasks:
        train_idx = [i for i, r in enumerate(rows) if r["task"] != held]
        test_idx = [i for i, r in enumerate(rows) if r["task"] == held]
        # Need at least 2 classes in train
        y_train = [y[i] for i in train_idx]
        if len(set(y_train)) < 2:
            for i in test_idx:
                preds[i] = Counter(y_train).most_common(1)[0][0]
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.08, random_state=0
        )
        clf.fit(Xtr, y_train)
        yhat = clf.predict(Xte)
        for i, p in zip(test_idx, yhat):
            preds[i] = p
    return preds


def random_split_eval(rows, X, y, seed: int = 0, frac: float = 0.7):
    rng = np.random.default_rng(seed)
    n = len(rows)
    idx = rng.permutation(n)
    n_train = int(frac * n)
    tr, te = idx[:n_train].tolist(), idx[n_train:].tolist()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.08, random_state=0
    )
    clf.fit(Xtr, [y[i] for i in tr])
    yhat = clf.predict(Xte)
    preds = [None] * n
    for i in tr:
        preds[i] = y[i]  # train: oracle (used only for full-policy view)
    for i, p in zip(te, yhat):
        preds[i] = p
    test_acc = mean(int(p == y[i]) for i, p in zip(te, yhat))
    return preds, test_acc, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="best_iso_quality_99",
                    choices=["best_quality", "best_iso_quality_99",
                             "best_iso_quality_95", "best_iso_quality_90"])
    args = ap.parse_args()

    rows = load_dataset(args.model)
    X = featurize(rows)
    y = [r[args.label] for r in rows]

    print(f"\n=== C7 router  model={args.model}  label={args.label}  n={len(rows)} ===\n")
    print("label distribution:", dict(Counter(y)))
    print()

    # ---- baselines ----
    print("baselines:")
    fmt = "  {:24s} q={:.4f}  cr={:6.2f}x  picks={}"
    for c in CONFIGS:
        picks = [c] * len(rows)
        q, cr, _ = policy_metrics(rows, picks)
        print(fmt.format(f"always-{c}", q, cr, "-"))

    # iso-quality oracle (training label, upper bound)
    oracle_picks = [r[args.label] for r in rows]
    qo, cro, po = policy_metrics(rows, oracle_picks)
    tau = {"best_iso_quality_99": 0.99, "best_iso_quality_95": 0.95,
           "best_iso_quality_90": 0.90, "best_quality": 1.00}[args.label]
    print(fmt.format(f"ORACLE (tau={tau})", qo, cro, dict(po)))
    print()

    # ---- random 70/30 split (in-distribution) ----
    preds_rand, test_acc, test_idx = random_split_eval(rows, X, y, seed=0)
    test_rows = [rows[i] for i in test_idx]
    test_preds = [preds_rand[i] for i in test_idx]
    test_oracle = [y[i] for i in test_idx]
    qp, crp, pp = policy_metrics(test_rows, test_preds)
    qo2, cro2, _ = policy_metrics(test_rows, test_oracle)
    always_c1 = [int(y[i] == "C1") for i in test_idx]
    print("=== random 70/30 split (in-distribution) ===")
    print(f"  test top-1 acc:        {test_acc:.3f}   (always-C1 baseline = {mean(always_c1):.3f})")
    print(f"  classifier (test set): q={qp:.4f}  cr={crp:.2f}x  picks={dict(pp)}")
    print(f"  oracle      (test set): q={qo2:.4f}  cr={cro2:.2f}x")
    print()

    # ---- leave-one-task-out (OOD) ----
    preds_loto = loto_eval(rows, X, y, tau)
    acc = mean(int(p == yi) for p, yi in zip(preds_loto, y))
    always_c1_full = mean(int(yi == "C1") for yi in y)
    q_l, cr_l, p_l = policy_metrics(rows, preds_loto)
    print("=== leave-one-task-out (OOD by task) ===")
    print(f"  top-1 acc:             {acc:.3f}   (always-C1 baseline = {always_c1_full:.3f})")
    print(f"  classifier policy:     q={q_l:.4f}  cr={cr_l:.2f}x  picks={dict(p_l)}")
    print(f"  oracle policy:         q={qo:.4f}  cr={cro:.2f}x  picks={dict(po)}")
    print()

    # per-held-task breakdown
    print("  per-held-task accuracy & policy outcome:")
    by_task = defaultdict(list)
    for r, yhat, yi in zip(rows, preds_loto, y):
        by_task[r["task"]].append((r, yhat, yi))
    print("    task           n   acc    cls_q    cls_cr    orc_q    orc_cr")
    for t in sorted(by_task):
        items = by_task[t]
        rs = [x[0] for x in items]
        ph = [x[1] for x in items]
        po_ = [x[2] for x in items]
        a = mean(int(p == y_) for _, p, y_ in items)
        qc, crc, _ = policy_metrics(rs, ph)
        qor, cror, _ = policy_metrics(rs, po_)
        print(f"    {t:14s} {len(items):3d}  {a:.3f}  {qc:.4f}  {crc:6.2f}x  {qor:.4f}  {cror:6.2f}x")


if __name__ == "__main__":
    main()
