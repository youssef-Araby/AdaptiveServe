"""C7 router — regression-based variant.

Trains 6 regressors (one per config) to predict the per-prompt quality
score from the 7 features. At inference, the router picks the config
with the highest compression whose predicted quality is at least
`tau * predicted_quality(C0)`.

This is more sample-efficient than the classification head because:
- Every prompt provides 6 supervised regression targets, not 1 label.
- The model is invariant to the ratio of "C1-preferring" to
  "C0-preferring" tasks in the training set: it learns the quality
  surface of each config independently.

Reports random-split + leave-one-task-out evaluation, same as
train_c7_router.py.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import harmonic_mean, mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
FEATURE_KEYS = [
    "seq_len_tokens", "seq_len_chars", "token_entropy",
    "gzip_ratio", "unique_token_ratio", "question_position",
    "newline_density",
]
CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]


def load(model: str):
    return [json.loads(l) for l in open(REPO / "runs/dataset" / f"{model}.jsonl")]


def featurize(rows):
    X = np.zeros((len(rows), len(FEATURE_KEYS)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, k in enumerate(FEATURE_KEYS):
            v = r["features"].get(k)
            X[i, j] = -1.0 if v is None else float(v)
    return X


def fit_regressors(X_train, rows_train):
    """Fit one regressor per config; returns dict cfg -> (scaler, regressor)."""
    out = {}
    for c in CONFIGS:
        y = np.array([r["scores"][c] for r in rows_train], dtype=np.float32)
        sc = StandardScaler()
        Xs = sc.fit_transform(X_train)
        reg = HistGradientBoostingRegressor(
            max_iter=300, max_depth=4, learning_rate=0.05, random_state=0
        )
        reg.fit(Xs, y)
        out[c] = (sc, reg)
    return out


def predict_quality(regs, X):
    """Returns dict cfg -> array of predicted qualities."""
    return {c: reg.predict(sc.transform(X)) for c, (sc, reg) in regs.items()}


def policy_from_predictions(rows, q_pred, tau: float):
    """For each prompt: pick config with max compression s.t.
    predicted_q[c] >= tau * predicted_q[C0]."""
    n = len(rows)
    picks = []
    for i, r in enumerate(rows):
        thresh = tau * q_pred["C0"][i]
        best_c, best_cr = "C0", r["compression"]["C0"]
        for c in CONFIGS:
            if q_pred[c][i] >= thresh and r["compression"][c] > best_cr:
                best_c, best_cr = c, r["compression"][c]
        picks.append(best_c)
    return picks


def metrics(rows, picks):
    qs = [r["scores"][c] for r, c in zip(rows, picks)]
    crs = [r["compression"][c] for r, c in zip(rows, picks)]
    return mean(qs), harmonic_mean(crs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tau", type=float, default=0.99)
    args = ap.parse_args()

    rows = load(args.model)
    X = featurize(rows)
    tau = args.tau

    print(f"\n=== C7 router (regression)  model={args.model}  tau={tau}  n={len(rows)} ===\n")

    # Baselines
    print("baselines:")
    for c in CONFIGS:
        q, cr = metrics(rows, [c] * len(rows))
        print(f"  always-{c:3s}  q={q:.4f}  cr={cr:6.2f}x")

    # Oracle (true per-prompt iso-quality)
    label_key = {1.00: "best_iso_quality_99", 0.99: "best_iso_quality_99",
                 0.95: "best_iso_quality_95", 0.90: "best_iso_quality_90"}.get(tau)
    oracle_picks = [r[label_key] for r in rows]
    qo, cro = metrics(rows, oracle_picks)
    print(f"  ORACLE         q={qo:.4f}  cr={cro:6.2f}x")
    print()

    # ---- random 70/30 split ----
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    n_tr = int(0.7 * len(rows))
    tr, te = idx[:n_tr].tolist(), idx[n_tr:].tolist()
    rows_tr = [rows[i] for i in tr]
    rows_te = [rows[i] for i in te]
    regs = fit_regressors(X[tr], rows_tr)
    q_pred_te = predict_quality(regs, X[te])
    picks_te = policy_from_predictions(rows_te, q_pred_te, tau)
    q_te, cr_te = metrics(rows_te, picks_te)
    qo_te, cro_te = metrics(rows_te, [rows[i][label_key] for i in te])
    qc1_te, crc1_te = metrics(rows_te, ["C1"] * len(te))
    qc0_te, _ = metrics(rows_te, ["C0"] * len(te))
    print("=== random 70/30 split ===")
    print(f"  classifier (test): q={q_te:.4f}  cr={cr_te:6.2f}x  picks={dict(Counter(picks_te))}")
    print(f"  always-C0  (test): q={qc0_te:.4f}")
    print(f"  always-C1  (test): q={qc1_te:.4f}  cr={crc1_te:6.2f}x")
    print(f"  oracle     (test): q={qo_te:.4f}  cr={cro_te:6.2f}x")
    print()

    # ---- leave-one-task-out ----
    tasks = sorted({r["task"] for r in rows})
    preds = [None] * len(rows)
    for held in tasks:
        tr_i = [i for i, r in enumerate(rows) if r["task"] != held]
        te_i = [i for i, r in enumerate(rows) if r["task"] == held]
        regs = fit_regressors(X[tr_i], [rows[i] for i in tr_i])
        q_pred = predict_quality(regs, X[te_i])
        picks = policy_from_predictions([rows[i] for i in te_i], q_pred, tau)
        for i, p in zip(te_i, picks):
            preds[i] = p

    q_l, cr_l = metrics(rows, preds)
    qc1_full, crc1_full = metrics(rows, ["C1"] * len(rows))
    qc0_full, _ = metrics(rows, ["C0"] * len(rows))
    print("=== leave-one-task-out (OOD by task) ===")
    print(f"  classifier policy: q={q_l:.4f}  cr={cr_l:6.2f}x  picks={dict(Counter(preds))}")
    print(f"  always-C0:         q={qc0_full:.4f}")
    print(f"  always-C1:         q={qc1_full:.4f}  cr={crc1_full:6.2f}x")
    print(f"  oracle:            q={qo:.4f}  cr={cro:6.2f}x")
    print()

    print("  per-held-task outcome:")
    print("    task             n   cls_q   cls_cr   orc_q   orc_cr   alwaysC1_q")
    by_task = defaultdict(list)
    for r, p in zip(rows, preds):
        by_task[r["task"]].append((r, p))
    for t in sorted(by_task):
        items = by_task[t]
        rs = [x[0] for x in items]
        ph = [x[1] for x in items]
        po = [r[label_key] for r in rs]
        qc, crc = metrics(rs, ph)
        qor, cror = metrics(rs, po)
        qc1, _ = metrics(rs, ["C1"] * len(rs))
        print(f"    {t:16s} {len(items):3d}  {qc:.4f}  {crc:6.2f}x  {qor:.4f}  {cror:6.2f}x  {qc1:.4f}")


if __name__ == "__main__":
    main()
