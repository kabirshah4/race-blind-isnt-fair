"""Second-state check: does the Georgia finding hold in North Carolina?

Run:  .venv/bin/python src/other_state.py      (after fix.py; needs data/raw/nc_2022.csv and nc_2023.csv,
      which src/download.py can fetch with STATES = ["GA", "NC"])
Writes results_nc.md.

Plain-English summary
---------------------
Test 1, TRANSFER: take the models trained on Georgia 2022, change nothing, and run them on
        North Carolina 2023. Does the Georgia model still work, and is it still unfair there?
Test 2, FRESH:    repeat our whole recipe in NC: train A (all inputs), B (no neighbourhood)
        and C (B + the ratio fairness rule we chose on Georgia validation data) on NC 2022,
        test on NC 2023. Does the same pattern (big gap, shrinking with each fix) appear?
Same measures as Georgia: PR-AUC, and the Black vs. white false-alarm ratio among
applicants the bank approved, with a 95% bootstrap range.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fairlearn.reductions import FalsePositiveRateParity
from sklearn.metrics import average_precision_score

from fix import APPLICANT_FEATURES, bootstrap_ci, false_alarm_ratio, fit_fair, group_costs, to_matrix, xgb
from prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_year

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)


def bundle_score(bundle, d):
    X = bundle["prep"].transform(d[bundle["features"] + CATEGORICAL_FEATURES])
    return sum(w * m.predict_proba(X)[:, 1] for w, m in zip(bundle["weights"], bundle["models"]))


def row(name, test, score, flagged):
    lo, hi = bootstrap_ci(test, flagged)
    c = group_costs(test, flagged)
    return {"model": name, "PR-AUC": round(average_precision_score(1 - test.approved, score), 3),
            "Black vs white false alarms": round(false_alarm_ratio(test, flagged), 2), "95% range": f"{lo:.2f}–{hi:.2f}",
            "white false alarm %": c["White false alarm %"], "Black false alarm %": c["Black false alarm %"]}


def main():
    nc22 = load_year(2022, "nc").reset_index(drop=True)
    nc23 = load_year(2023, "nc").reset_index(drop=True)
    print(f"NC denial rate by group (2023): " +
          ", ".join(f"{g} {1 - nc23[nc23.race_group == g].approved.mean():.1%}" for g in ["White", "Black", "Hispanic", "Asian"]))

    # ---- Test 1: Georgia-trained models, unchanged, on NC 2023 -----------------------------
    rows1 = []
    for name, file in [("A. Georgia model, all inputs", "model_A_full.joblib"), ("C. Georgia fair model", "model_C_fair.joblib")]:
        b = joblib.load(ROOT / "models" / file)
        s = bundle_score(b, nc23)
        rows1.append(row(name, nc23, s, (s >= b["cutoff"]).astype(int)))
    t1 = pd.DataFrame(rows1)

    # ---- Test 2: same recipe, trained fresh on NC 2022, tested on NC 2023 -------------------
    y = (1 - nc22.approved).to_numpy()
    share = y.mean()
    rows2 = []
    for name, feats in [("A. All inputs", NUMERIC_FEATURES), ("B. No neighbourhood", APPLICANT_FEATURES)]:
        Xtr, Xte = to_matrix(nc22, nc23, feats)
        m = xgb().fit(Xtr, y)
        cut = np.quantile(m.predict_proba(Xtr)[:, 1], 1 - share)
        s = m.predict_proba(Xte)[:, 1]
        rows2.append(row(name, nc23, s, (s >= cut).astype(int)))
    Xtr, Xte = to_matrix(nc22, nc23, APPLICANT_FEATURES)
    sample = rng.choice(len(nc22), 60_000, replace=False)
    keep, score = fit_fair(Xtr[sample], y[sample], nc22.race_group.to_numpy()[sample], FalsePositiveRateParity(ratio_bound=0.9))
    cut = np.quantile(score(Xtr), 1 - share)
    s = score(Xte)
    rows2.append(row("C. B + ratio fairness rule", nc23, s, (s >= cut).astype(int)))
    t2 = pd.DataFrame(rows2)

    print("\nTest 1 (Georgia models on NC 2023):\n", t1.to_string(index=False))
    print("\nTest 2 (trained on NC 2022, tested on NC 2023):\n", t2.to_string(index=False))
    (ROOT / "results_nc.md").write_text(
        f"# Second-state check: North Carolina\n\nNC applications: {len(nc22):,} (2022), {len(nc23):,} (2023).\n\n"
        "## Test 1: Georgia-trained models, unchanged, on NC 2023\n\n" + t1.to_markdown(index=False) +
        "\n\n## Test 2: same recipe trained on NC 2022, tested on NC 2023\n\n" + t2.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
