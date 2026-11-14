"""Model C retrained on ALL of 2022 (146,793 rows) instead of a 60,000-row sample.

Run:  .venv/bin/python src/robustness_c.py      (slow: ~40 XGBoost fits on the full data)
Writes results_robustness_c.md.

Plain-English summary
---------------------
fix.py trained the fairness model C on a random 60,000-row sample because the fairness
method (Exponentiated Gradient) is slow. Judges pointed out that this mixes two things:
is C less accurate because of the fairness rule, or just because it saw less data?
Here we train C exactly the same way (same rule, ratio >= 0.9, picked on 2022 validation
data) but on the full 2022 data, and compare.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.reductions import FalsePositiveRateParity
from sklearn.metrics import average_precision_score

from fix import APPLICANT_FEATURES, DATA, false_alarm_ratio, fit_fair, to_matrix

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_parquet(DATA)
    train, test = df[df.year == 2022].reset_index(drop=True), df[df.year == 2023].reset_index(drop=True)
    y = 1 - train.approved.to_numpy()
    Xtr, Xte = to_matrix(train, test, APPLICANT_FEATURES)
    keep, score = fit_fair(Xtr, y, train.race_group.to_numpy(), FalsePositiveRateParity(ratio_bound=0.9))
    cut = np.quantile(score(Xtr), 1 - y.mean())
    s = score(Xte)
    f = (s >= cut).astype(int)
    t = test.assign(f=f)
    ok = t[t.approved == 1]
    fa = lambda g: ok[ok.race_group == g].f.mean()
    row = (f"| C retrained on all 146,793 rows | {average_precision_score(1 - test.approved, s):.3f} | "
           f"{false_alarm_ratio(test, f):.2f}× | {f.mean():.1%} | {fa('White'):.1%} / {fa('Black'):.1%} |")
    text = ("# Model C on the full 2022 data\n\n"
            "| model | PR-AUC | Black vs white false alarms | share flagged | false alarms white / Black |\n|---|---|---|---|---|\n"
            "| A. Full model (published) | 0.516 | 3.01× | 10.3% | 4.0% / 12.0% |\n"
            "| C on a 60,000-row sample (published) | 0.501 | 1.21× | 9.5% | 5.6% / 6.8% |\n" + row + "\n")
    (ROOT / "results_robustness_c.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
