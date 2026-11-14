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

    t["risk_A"] = risk(A, t)
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

    # Two checks a fairness researcher would ask for:
    # (1) Which part of the address matters? Swap ONLY "% minority residents" vs. ONLY the other 6.
    # (2) Is 53% vs 22% just because flagged Black applicants sit closer to the cut-off?
    #     Compare people at the SAME distance from it (risk between 1.0x and 1.5x the cut-off).
    rows2 = []
    for g in ["Black", "White"]:
        flagged = qualified[(qualified.race_group == g) & qualified.flag_A]
        for label, cols in [("only % minority residents", ["tract_minority_population_percent"]),
                            ("only the other 6 neighbourhood numbers", [c for c in TRACT_FEATURES if c != "tract_minority_population_percent"])]:
            moved = flagged.copy()
            moved[cols] = white_tract[cols].values
            rows2.append({"group": g, "swap": label, "people": len(flagged),
                          "approved by A after the swap": f"{(risk(A, moved) < A['cutoff']).mean():.0%}"})
        near = flagged[flagged.risk_A < 1.5 * A["cutoff"]]
        moved = near.copy()
        moved[TRACT_FEATURES] = white_tract.values
        rows2.append({"group": g, "swap": "all 7, only people within 1.5x of the cut-off", "people": len(near),
                      "approved by A after the swap": f"{(risk(A, moved) < A['cutoff']).mean():.0%}"})
    table2 = pd.DataFrame(rows2)
    print(table.to_string(index=False), "\n\n", table2.to_string(index=False))
    (ROOT / "results_counterfactual.md").write_text(
        "## Same person, different address (2023 test data, model A)\n\n" + table.to_markdown(index=False) +
        "\n\n## Which part of the address, and a like-for-like check\n\n" + table2.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
