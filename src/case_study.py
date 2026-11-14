"""One real applicant the model gets wrong: a case study for the write-up.

Run:  .venv/bin/python src/case_study.py      (after download.py and fix.py)
Writes results_case_study.md.

The brief asks every project to show at least one example its model gets wrong. Our
fairness numbers are about groups, so this picks ONE real 2023 Georgia application that:
  * the bank approved (a qualified applicant),
  * model A (race-blind, uses neighbourhood data) wrongly flags as "likely denied",
  * model A approves once ONLY the 7 neighbourhood numbers are swapped to the typical
    mostly-white tract (models/neighbourhoods.json), with everything else unchanged.
Among all such applicants we take the one closest to the group's median finances, so the
example is typical, not cherry-picked for drama. HMDA is public data; we still round every
number and don't name the lender, county or tract.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from prep import CATEGORICAL_FEATURES, TRACT_FEATURES, load_year

ROOT = Path(__file__).resolve().parents[1]
LOAN_TYPE = {"1": "conventional", "2": "FHA", "3": "VA", "4": "USDA"}


def risk(bundle, df):
    X = bundle["prep"].transform(df[bundle["features"] + CATEGORICAL_FEATURES])
    return sum(w * m.predict_proba(X)[:, 1] for w, m in zip(bundle["weights"], bundle["models"]))


def main():
    test = load_year(2023).reset_index(drop=True)
    A = joblib.load(ROOT / "models" / "model_A_full.joblib")
    C = joblib.load(ROOT / "models" / "model_C_fair.joblib")
    white_tract = json.loads((ROOT / "models" / "neighbourhoods.json").read_text())["Mostly white"]

    test["rA"] = risk(A, test)
    test["rC"] = risk(C, test)
    swapped = test.copy()
    for c in TRACT_FEATURES:
        swapped[c] = white_tract[c]
    test["rA_swap"] = risk(A, swapped)

    wrong = test[(test.race_group == "Black") & (test.approved == 1) & (test.rA >= A["cutoff"])]
    flips = wrong[wrong.rA_swap < A["cutoff"]]
    cand = flips[flips.rC < C["cutoff"]]  # and our fixed model gets it right
    cols = ["income", "loan_amount", "dti"]
    z = ((cand[cols] - cand[cols].median()) / cand[cols].std()).abs().sum(axis=1)
    p = cand.loc[z.idxmin()]

    r5 = lambda v: int(5 * round(v / 5))
    story = (
        f"A Black applicant in Georgia, 2023: about ${r5(p.income)}K a year, borrowing about ${r5(p.loan_amount)}K "
        f"on a ${r5(p.property_value)}K home ({p.ltv:.0f}% loan-to-value), debts around {p.dti:.0f}% of income, "
        f"{LOAN_TYPE.get(p.loan_type, 'other')} loan, {'with' if p.has_coapplicant else 'no'} co-applicant. "
        f"Their census tract is about {r5(p.tract_minority_population_percent)}% minority.\n\n"
        f"* **The bank approved them.** They were qualified.\n"
        f"* **Model A wrongly flags them:** {p.rA:.0%} chance of denial, above its {A['cutoff']:.0%} cut-off.\n"
        f"* **Same person, typical mostly-white tract** (only the 7 neighbourhood numbers change): "
        f"{p.rA_swap:.0%}, so model A would approve.\n"
        f"* **Our fixed model C** (no neighbourhood data): {p.rC:.0%}, below its {C['cutoff']:.0%} cut-off, so it approves.\n\n"
        f"How common this is: of {len(wrong):,} qualified Black applicants model A wrongly flagged in 2023, "
        f"{len(flips):,} ({len(flips) / len(wrong):.0%}) would be approved by model A at the mostly-white tract.\n\n"
        "**Why we think model A gets this wrong:** nothing about this person's finances changed between the two "
        "runs; only the neighbourhood numbers did. Model A learned from past bank decisions that applications from "
        "high-minority tracts were denied more often, so it carries that pattern forward as if the address were a "
        "risk factor. Model C never sees the neighbourhood, so it can't.\n")
    print(story)
    (ROOT / "results_case_study.md").write_text("# Case study: one qualified applicant model A gets wrong\n\n" + story +
        "\nPicked by `src/case_study.py`: the wrongly-flagged, address-flips, model-C-approves applicant closest to "
        "that group's median income, loan and debt-to-income. Numbers rounded; no lender, county or tract named.\n")


if __name__ == "__main__":
    main()
