"""FairLens demo: the same mortgage applicant, judged by two race-blind models.

Run:  .venv/bin/python app/app.py      then open the link it prints.
Needs models/model_A_full.joblib and models/model_C_fair.joblib (made by src/fix.py).

What the demo shows
-------------------
* Left:  the STANDARD model (A). It uses neighbourhood data.
* Right: the FAIRNESS-CHECKED model (C). No neighbourhood data, plus
         fairness-constrained training.
* Neither model ever sees race.
* The key moment: change ONLY the neighbourhood slider. The standard model's
  risk moves; the fairness-checked model's doesn't, because it ignores location.
* "What would change the decision?" tries small, realistic changes (lower
  debt-to-income, smaller loan, adding a co-applicant) and reports the first
  one that flips a denial to an approval.
"""
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_A = joblib.load(ROOT / "models" / "model_A_full.joblib")
MODEL_C = joblib.load(ROOT / "models" / "model_C_fair.joblib")

# Three REAL neighbourhood types, built from Georgia's 2023 census-tract data.
# Each profile = the median of all 7 tract columns across tracts in that range.
TRACT_COLS = ["tract_minority_population_percent", "tract_to_msa_income_percentage",
              "ffiec_msa_md_median_family_income", "tract_population", "tract_owner_occupied_units",
              "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
_d = pd.read_parquet(ROOT / "data" / "processed" / "hmda_ga.parquet")
_d = _d[_d.year == 2023]
_m = _d.tract_minority_population_percent
NEIGHBOURHOODS = {
    "Mostly white (under 20% minority residents)": _d[_m < 20][TRACT_COLS].median(),
    "Mixed (40–60% minority residents)": _d[(_m >= 40) & (_m <= 60)][TRACT_COLS].median(),
    "Mostly minority (over 90% minority residents)": _d[_m > 90][TRACT_COLS].median(),
}

LOAN_TYPES = {"Conventional": "1", "FHA (government-backed)": "2", "VA (veterans)": "3", "USDA (rural)": "4"}


def build_applicant(income, loan_amount, property_value, dti, loan_type, thirty_year,
                    coapplicant, preapproval, neighbourhood):
    """Turn the form inputs into one row with the same columns the models were trained on."""
    row = {
        "income": income, "loan_amount": loan_amount, "property_value": property_value,
        "ltv": min(loan_amount / property_value * 100, 150), "dti": dti,
        "loan_term_360": int(thirty_year), "has_coapplicant": int(coapplicant),
        "loan_type": LOAN_TYPES[loan_type], "preapproval": "1" if preapproval else "2",
        "conforming_loan_limit": "C" if loan_amount <= 766 else "NC",
        "submission_of_application": "1", "initially_payable_to_institution": "1",
        **NEIGHBOURHOODS[neighbourhood].to_dict(),
    }
    return pd.DataFrame([row])


def denial_risk(bundle, applicant):
    """Probability of denial. Model C is a weighted mix of models, so we average them."""
    from prep import CATEGORICAL_FEATURES
    X = bundle["prep"].transform(applicant[bundle["features"] + CATEGORICAL_FEATURES])
    return float(sum(w * m.predict_proba(X)[0, 1] for w, m in zip(bundle["weights"], bundle["models"])))


def what_would_change(bundle, inputs):
    """Try simple, realistic changes one at a time; return the first that flips to approval."""
    tries = []
    for new_dti in [40, 36, 33, 30, 25]:
        if new_dti < inputs["dti"]:
            tries.append((f"debt-to-income lowered to {new_dti}% (e.g. pay off a car loan)", {"dti": new_dti}))
    for cut in [0.95, 0.9, 0.85]:
        tries.append((f"a {int((1 - cut) * 100)}% smaller loan (bigger down payment)",
                      {"loan_amount": round(inputs["loan_amount"] * cut)}))
    if not inputs["coapplicant"]:
        tries.append(("adding a co-applicant", {"coapplicant": True}))
    for label, change in tries:
        trial = {**inputs, **change}
        if denial_risk(bundle, build_applicant(**trial)) < bundle["cutoff"]:
            return label
    return "no single small change was enough. This applicant would need bigger changes."


def card(title, subtitle, risk, cutoff, advice):
    flagged = risk >= cutoff
    verdict = "⚠️ Likely DENIED" if flagged else "✅ Likely APPROVED"
    color = "#b45309" if flagged else "#15803d"
    extra = f"<p style='margin:8px 0 0'><b>What would change it:</b> {advice}</p>" if flagged else ""
    return (f"<div style='border:2px solid {color};border-radius:10px;padding:14px'>"
            f"<div style='font-size:13px;opacity:.75'>{subtitle}</div>"
            f"<div style='font-size:17px;font-weight:600'>{title}</div>"
            f"<div style='font-size:22px;color:{color};font-weight:700;margin-top:6px'>{verdict}</div>"
            f"<div>Denial risk: <b>{risk:.0%}</b> (flags above {cutoff:.0%})</div>{extra}</div>")


def judge(income, loan_amount, property_value, dti, loan_type, thirty_year, coapplicant,
          preapproval, neighbourhood):
    inputs = dict(income=income, loan_amount=loan_amount, property_value=property_value, dti=dti,
                  loan_type=loan_type, thirty_year=thirty_year, coapplicant=coapplicant,
                  preapproval=preapproval, neighbourhood=neighbourhood)
    applicant = build_applicant(**inputs)
    out = []
    for bundle, title, sub in [(MODEL_A, "Standard model", "Uses neighbourhood data · never sees race"),
                               (MODEL_C, "Fairness-checked model", "No neighbourhood data + fairness training · never sees race")]:
        risk = denial_risk(bundle, applicant)
        advice = what_would_change(bundle, inputs) if risk >= bundle["cutoff"] else ""
        out.append(card(title, sub, risk, bundle["cutoff"], advice))
    return out


with gr.Blocks(title="FairLens") as demo:
    gr.Markdown(
        "# FairLens: would a race-blind mortgage model still treat you differently?\n"
        "Two models trained on 264,000 real Georgia mortgage applications (HMDA 2022–2023). "
        "**Neither model ever sees race.** Try moving *only* the neighbourhood slider.\n\n"
        "*Prototype for a high-school hackathon. This is not financial advice.*")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### The applicant")
            # Default applicant: a borderline case found by searching, where the
            # neighbourhood alone flips the standard model's decision.
            income = gr.Number(100, label="Yearly income ($ thousands)")
            loan_amount = gr.Number(100, label="Loan amount ($ thousands)")
            property_value = gr.Number(125, label="Home price ($ thousands)")
            dti = gr.Slider(10, 65, 46, step=1, label="Debt-to-income ratio (%)")
            loan_type = gr.Dropdown(list(LOAN_TYPES), value="Conventional", label="Loan type")
            thirty_year = gr.Checkbox(True, label="30-year mortgage")
            coapplicant = gr.Checkbox(False, label="Has a co-applicant")
            preapproval = gr.Checkbox(False, label="Asked for pre-approval")
            gr.Markdown("### The neighbourhood")
            neighbourhood = gr.Radio(list(NEIGHBOURHOODS), value=list(NEIGHBOURHOODS)[2],
                                     label="Where the home is (real Georgia neighbourhood profiles)")
            go = gr.Button("Check application", variant="primary")
        with gr.Column():
            gr.Markdown("### Decisions")
            out_a, out_c = gr.HTML(), gr.HTML()
            gr.Markdown(
                "**Why this matters:** on 2023 data, the standard model wrongly flagged qualified Black "
                "applicants **2.5×** as often as qualified white applicants. The fairness-checked model "
                "brought that to **1.05×** (no measurable gap) while keeping 97% of the accuracy.\n\n"
                "**Try it:** keep everything the same and switch only *Where the home is*.")
    inputs = [income, loan_amount, property_value, dti, loan_type, thirty_year, coapplicant,
              preapproval, neighbourhood]
    go.click(judge, inputs, [out_a, out_c])
    for comp in inputs:  # update live as sliders move
        comp.change(judge, inputs, [out_a, out_c])
    demo.load(judge, inputs, [out_a, out_c])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))  # so `from prep import ...` works
    demo.launch()
