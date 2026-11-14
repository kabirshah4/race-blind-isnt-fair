"""Pre-compute the demo's answers so it can run as a free static web page (GitHub Pages).

Run:  .venv/bin/python src/export_demo.py      (after fix.py)
Writes docs/demo_data.json, which docs/index.html reads.

Plain-English summary
---------------------
Free hosting can't run Python, so instead of running the models live, we run them ahead of
time for every combination of the demo's inputs and save the answers in a file:

  9 incomes × 7 loan sizes × 5 down payments × 19 debt-to-income values × 4 loan types
  × with/without co-applicant = 47,880 applicants, each judged by
  model A (in all 3 neighbourhoods) and model C (which ignores neighbourhood).

Debt-to-income uses exactly the 19 values in HMDA (it is reported in ranges), so the page
loses nothing there. The page then just looks answers up, so it is instant.
Answers are stored as risk × 1000 (whole numbers) to keep the file small.
"""
import itertools
import json
from pathlib import Path

import joblib
import pandas as pd

from prep import CATEGORICAL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
INCOMES = [30, 45, 60, 75, 90, 110, 130, 160, 200]          # $ thousands per year
LOANS = [75, 100, 150, 200, 250, 300, 400]                   # $ thousands
DOWN = [3.5, 5, 10, 20, 30]                                  # % down payment
DTI = [15, 25, 33] + list(range(36, 50)) + [55, 65]          # the 19 values HMDA uses
DTI_LABEL = {15: "under 20%", 25: "20–30%", 33: "30–36%", 55: "50–60%", 65: "over 60%"}
LOAN_TYPES = {"Conventional": "1", "FHA (government-backed)": "2", "VA (veterans)": "3", "USDA (rural)": "4"}
COAPP = [0, 1]


def risk(bundle, d):
    X = bundle["prep"].transform(d[bundle["features"] + CATEGORICAL_FEATURES])
    return sum(w * m.predict_proba(X)[:, 1] for w, m in zip(bundle["weights"], bundle["models"]))


def main():
    A = joblib.load(ROOT / "models" / "model_A_full.joblib")
    C = joblib.load(ROOT / "models" / "model_C_fair.joblib")
    nbs = json.loads((ROOT / "models" / "neighbourhoods.json").read_text())

    rows = []
    for inc, loan, down, dti, lt, co in itertools.product(INCOMES, LOANS, DOWN, DTI, LOAN_TYPES.values(), COAPP):
        price = loan / (1 - down / 100)
        rows.append({"income": inc, "loan_amount": loan, "property_value": price, "ltv": 100 - down, "dti": dti,
                     "loan_term_360": 1, "has_coapplicant": co, "loan_type": lt,
                     "conforming_loan_limit": "C" if loan <= 766 else "NC", "submission_of_application": "1"})
    grid = pd.DataFrame(rows)  # row order = itertools order, which the web page re-creates

    out = {"axes": {"income": INCOMES, "loan": LOANS, "down": DOWN, "dti": DTI, "loan_type": list(LOAN_TYPES),
                    "coapp": COAPP},
           "dti_label": {str(k): v for k, v in DTI_LABEL.items()},
           "cutoff": {"A": round(1000 * float(A["cutoff"])), "C": round(1000 * float(C["cutoff"]))},
           "neighbourhoods": list(nbs), "A": {}, "C": None}
    for name, profile in nbs.items():
        g = grid.assign(**profile)
        out["A"][name] = [round(1000 * float(x)) for x in risk(A, g)]
    out["C"] = [round(1000 * float(x)) for x in risk(C, grid.assign(**nbs["Mostly white"]))]  # C ignores neighbourhood

    (ROOT / "docs").mkdir(exist_ok=True)
    path = ROOT / "docs" / "demo_data.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(grid):,} applicants × (3 neighbourhoods for A + C) -> {path.relative_to(ROOT)} "
          f"({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
