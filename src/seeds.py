"""How much do the headline numbers move if we train again with a different random seed?

Run:  .venv/bin/python src/seeds.py      (after download.py; model C makes this take a while)
Writes results_seeds.md.

Our 95% ranges (fix.py) only resample the 2023 TEST rows. They don't include training
randomness: XGBoost samples rows and columns while it trains, and model C trains on a random
60,000-row sample. Here we retrain models A, B and C with 5 different seeds, everything else
identical to fix.py (same inputs, same cut-off rule, model C with the "ratio >= 0.9" rule that
fix.py chose on 2022 validation data), and report the mean and spread across the 5 runs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, FalsePositiveRateParity
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from fix import APPLICANT_FEATURES, DATA, false_alarm_ratio, to_matrix
from prep import NUMERIC_FEATURES, XGB_PARAMS

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2, 3, 4]


def main():
    df = pd.read_parquet(DATA)
    train, test = df[df.year == 2022].reset_index(drop=True), df[df.year == 2023].reset_index(drop=True)
    y_tr, y_te = (1 - train.approved).to_numpy(), (1 - test.approved).to_numpy()
    flag_share = y_tr.mean()
    XA = to_matrix(train, test, NUMERIC_FEATURES)
    XB = to_matrix(train, test, APPLICANT_FEATURES)
    rows = []
    for seed in SEEDS:
        xgb = lambda: XGBClassifier(**{**XGB_PARAMS, "random_state": seed})
        for name, (Xtr, Xte) in [("A. Full model", XA), ("B. No neighbourhood", XB)]:
            m = xgb().fit(Xtr, y_tr)
            cut = np.quantile(m.predict_proba(Xtr)[:, 1], 1 - flag_share)
            s = m.predict_proba(Xte)[:, 1]
            rows.append({"model": name, "seed": seed, "PR-AUC": average_precision_score(y_te, s),
                         "ratio": false_alarm_ratio(test, (s >= cut).astype(int))})
        # C: same recipe as fix.py, different seed for XGBoost AND for the 60k training sample.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train))
        tr = train.iloc[perm[: int(0.8 * len(train))]].reset_index(drop=True)
        Xtr, Xte = to_matrix(tr, test, APPLICANT_FEATURES)
        sample = rng.choice(len(tr), 60_000, replace=False)
        fair = ExponentiatedGradient(xgb(), constraints=FalsePositiveRateParity(ratio_bound=0.9), max_iter=40)
        fair.fit(Xtr[sample], (1 - tr.approved.to_numpy())[sample], sensitive_features=tr.race_group.to_numpy()[sample])
        keep = [(w, p) for w, p in zip(fair.weights_, fair.predictors_) if w > 0]
        score = lambda Z: sum(w * p.predict_proba(Z)[:, 1] for w, p in keep)
        cut = np.quantile(score(Xtr), 1 - flag_share)
        s = score(Xte)
        rows.append({"model": "C. Fairness-trained (ratio ≥ 0.9)", "seed": seed, "PR-AUC": average_precision_score(y_te, s),
                     "ratio": false_alarm_ratio(test, (s >= cut).astype(int))})
        print(pd.DataFrame(rows).tail(3).round(3).to_string(index=False), flush=True)

    r = pd.DataFrame(rows)
    g = r.groupby("model", sort=False).agg(pr_mean=("PR-AUC", "mean"), pr_sd=("PR-AUC", "std"),
                                          ra_mean=("ratio", "mean"), ra_sd=("ratio", "std"),
                                          ra_min=("ratio", "min"), ra_max=("ratio", "max")).reset_index()
    out = pd.DataFrame({
        "model": g.model,
        "PR-AUC (mean ± sd)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(g.pr_mean, g.pr_sd)],
        "Black vs. white false alarms (mean ± sd)": [f"{a:.2f}× ± {b:.2f}" for a, b in zip(g.ra_mean, g.ra_sd)],
        "lowest–highest of 5 runs": [f"{a:.2f}–{b:.2f}×" for a, b in zip(g.ra_min, g.ra_max)],
    })
    print(out.to_string(index=False))
    (ROOT / "results_seeds.md").write_text(
        "# Training randomness: 5 seeds per model (2023 test data)\n\n" + out.to_markdown(index=False) +
        "\n\nEach model was retrained 5 times (seeds 0–4) with everything else identical to fix.py; model C also "
        "draws a different 60,000-row training sample each time. This is the spread our test-set confidence "
        "intervals leave out.\n\n## Every run\n\n" + r.round(3).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
