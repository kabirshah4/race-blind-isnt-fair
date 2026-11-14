"""The "Why we care" numbers: Asian Indian vs. white applicants in Georgia, 2023.

Run:  .venv/bin/python src/asian_indian.py      (after download.py and fix.py)
Writes results_asian_indian.md.

Our main race groups come from HMDA's `derived_race`, which lumps all Asian applicants
together. HMDA also records a detailed race for each applicant (`applicant_race-1`), where
code 21 = Asian Indian. Here we use that code to check two things:
  1. Raw denial rates (before accounting for finances): Asian Indian vs. white.
  2. Model A's false-alarm rate (bank-approved applicants it wrongly flags), the same
     measure as the rest of the project, with a 95% bootstrap range.
Same cleaning as every other script (prep.load_year); non-Hispanic applicants only, so the
white group matches the project's "White" group.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prep import CATEGORICAL_FEATURES, RAW, load_year

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)


def main():
    test = load_year(2023)  # index = row number in the raw file, so we can join extra raw columns
    raw = pd.read_csv(RAW / "ga_2023.csv", usecols=["applicant_race-1", "derived_ethnicity"], dtype=str)
    test = test.join(raw)
    not_hispanic = test.derived_ethnicity != "Hispanic or Latino"
    test["group"] = np.select(
        [not_hispanic & (test["applicant_race-1"] == "21"), test.race_group == "White"],
        ["Asian Indian", "White"], "other")
    test = test[test.group != "other"].reset_index(drop=True)

    bundle = joblib.load(ROOT / "models" / "model_A_full.joblib")
    X = bundle["prep"].transform(test[bundle["features"] + CATEGORICAL_FEATURES])
    flagged = bundle["models"][0].predict_proba(X)[:, 1] >= bundle["cutoff"]

    ok = (test.approved == 1).to_numpy()
    ai, wh = (test.group == "Asian Indian").to_numpy(), (test.group == "White").to_numpy()

    def ratio(idx):
        f, o, a, w = flagged[idx], ok[idx], ai[idx], wh[idx]
        return f[a & o].mean() / f[w & o].mean()

    boots = [ratio(rng.integers(0, len(test), len(test))) for _ in range(500)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    rows = []
    for g, m in [("Asian Indian", ai), ("White", wh)]:
        rows.append({"group": g, "applications": int(m.sum()),
                     "bank denied %": round(100 * (1 - test.approved[m].mean()), 1),
                     "model A false alarms % (qualified, wrongly flagged)": round(100 * flagged[m & ok].mean(), 1)})
    t = pd.DataFrame(rows)
    fa = ratio(np.arange(len(test)))
    summary = (f"Georgia 2023: banks denied {t.iloc[0]['bank denied %']}% of Asian Indian applicants vs. "
               f"{t.iloc[1]['bank denied %']}% of white applicants (before accounting for finances). "
               f"Model A's false-alarm ratio, Asian Indian vs. white: {fa:.2f}× (95% range {lo:.2f}–{hi:.2f}).")
    print(summary)
    print(t.to_string(index=False))
    (ROOT / "results_asian_indian.md").write_text(
        "# Asian Indian vs. white applicants, Georgia 2023\n\n" + summary + "\n\n" + t.to_markdown(index=False) +
        "\n\nAsian Indian = HMDA `applicant_race-1` code 21, non-Hispanic. White = the project's White group. "
        "A ratio near 1 means model A does not over-flag qualified Asian Indian applicants. Small group, so the range is wide.\n")


if __name__ == "__main__":
    main()
