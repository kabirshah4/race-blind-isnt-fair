""""Same person, different address": what happens if ONLY the neighbourhood changes?

Run:  .venv/bin/python src/counterfactual.py      (after fix.py)
Writes results_counterfactual.md.

Plain-English summary
---------------------
We take qualified applicants (the bank approved them) whom the full model A wrongly
flags as "likely denied". We keep everything about them the same (income, debts,
loan, home price) and replace ONLY their 7 neighbourhood numbers with those of a
typical mostly-white Georgia neighbourhood (the median of 2023 tracts under 20%
minority residents). Then we ask model A again.

We report it for two groups of people, so the number isn't cherry-picked:
  * everyone model A wrongly flags, and
  * only the people model A flags but the fair model C does not,
and for white applicants too, as a comparison.
"""
from pathlib import Path

import joblib
import pandas as pd

from prep import CATEGORICAL_FEATURES, TRACT_FEATURES

ROOT = Path(__file__).resolve().parents[1]


def risk(bundle, d):
    X = bundle["prep"].transform(d[bundle["features"] + CATEGORICAL_FEATURES])
    return sum(w * m.predict_proba(X)[:, 1] for w, m in zip(bundle["weights"], bundle["models"]))


def main():
    A = joblib.load(ROOT / "models" / "model_A_full.joblib")
    C = joblib.load(ROOT / "models" / "model_C_fair.joblib")
    t = pd.read_parquet(ROOT / "data" / "processed" / "hmda_ga.parquet")
    t = t[t.year == 2023].reset_index(drop=True)
    t["flag_A"] = risk(A, t) >= A["cutoff"]
    t["flag_C"] = risk(C, t) >= C["cutoff"]
    white_tract = t[t.tract_minority_population_percent < 20][TRACT_FEATURES].median()

    rows = []
    qualified = t[t.approved == 1]
    for g in ["Black", "White"]:
        for label, subset in [("all wrongly flagged by A", qualified[(qualified.race_group == g) & qualified.flag_A]),
                              ("flagged by A but not by C", qualified[(qualified.race_group == g) & qualified.flag_A & ~qualified.flag_C])]:
            moved = subset.copy()
            moved[TRACT_FEATURES] = white_tract.values
            flipped = (risk(A, moved) < A["cutoff"]).mean()
            rows.append({"group": f"Qualified {g} applicants", "who": label, "people": len(subset),
                         "share of that group's qualified applicants": f"{len(subset) / (qualified.race_group == g).sum():.1%}",
                         "approved by A after moving ONLY the address": f"{flipped:.0%}"})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    (ROOT / "results_counterfactual.md").write_text(
        "## Same person, different address (2023 test data, model A)\n\n" + table.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
