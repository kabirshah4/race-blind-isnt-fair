"""FairLens demo: the same mortgage applicant, judged by two race-blind models.

Run:  .venv/bin/python app/app.py      then open http://127.0.0.1:7860
Needs models/model_A_full.joblib and models/model_C_fair.joblib (made by src/fix.py)
and data/processed/hmda_ga.parquet (made by src/prep.py).

What a judge should get in five seconds
---------------------------------------
* The headline says it: same applicant, only the address changes.
* The neighbourhood switch sits right above the two decisions it changes.
* Two decision cards: the STANDARD model (A, uses neighbourhood data) and the
  FAIRNESS-CHECKED model (C, no neighbourhood data + fairness training).
  Neither model ever sees race.
* "Every neighbourhood at once" shows both models across all three
  neighbourhoods, so the flip is visible without clicking anything.
* If a model flags the applicant, "What would change it" tries small, realistic
  changes and reports the first one that flips the decision.
"""
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prep import CATEGORICAL_FEATURES  # noqa: E402

MODEL_A = joblib.load(ROOT / "models" / "model_A_full.joblib")
MODEL_C = joblib.load(ROOT / "models" / "model_C_fair.joblib")
MODELS = [
    ("standard", "Standard model", "Uses neighborhood data", MODEL_A),
    ("fair", "Fairness-checked model", "No neighborhood data · fairness-trained", MODEL_C),
]

# Three REAL neighbourhood types from Georgia's 2023 census-tract data.
# Each profile = the median of all 7 tract columns across tracts in that range.
TRACT_COLS = ["tract_minority_population_percent", "tract_to_msa_income_percentage",
              "ffiec_msa_md_median_family_income", "tract_population", "tract_owner_occupied_units",
              "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
_d = pd.read_parquet(ROOT / "data" / "processed" / "hmda_ga.parquet")
_d = _d[_d.year == 2023]
_m = _d.tract_minority_population_percent
NEIGHBOURHOODS = {
    "Mostly white": _d[_m < 20][TRACT_COLS].median(),
    "Mixed": _d[(_m >= 40) & (_m <= 60)][TRACT_COLS].median(),
    "Mostly minority": _d[_m > 90][TRACT_COLS].median(),
}
NB_NOTE = {"Mostly white": "under 20% minority", "Mixed": "40–60% minority", "Mostly minority": "over 90% minority"}
LOAN_TYPES = {"Conventional": "1", "FHA (government-backed)": "2", "VA (veterans)": "3", "USDA (rural)": "4"}

# The borderline example: a real case found by search, where only the address flips model A.
MAYA = dict(income=100, loan_amount=100, property_value=125, dti=46, loan_type="Conventional",
            thirty_year=True, coapplicant=False, preapproval=False)


# ----------------------------------------------------------------------------- model logic
def build_applicant(income, loan_amount, property_value, dti, loan_type, thirty_year,
                    coapplicant, preapproval, neighbourhood):
    """Turn the form inputs into one row with the same columns the models were trained on."""
    row = {
        "income": income, "loan_amount": loan_amount, "property_value": property_value,
        "ltv": min(loan_amount / max(property_value, 1) * 100, 150), "dti": dti,
        "loan_term_360": int(thirty_year), "has_coapplicant": int(coapplicant),
        "loan_type": LOAN_TYPES[loan_type], "preapproval": "1" if preapproval else "2",
        "conforming_loan_limit": "C" if loan_amount <= 766 else "NC",
        "submission_of_application": "1", "initially_payable_to_institution": "1",
        **NEIGHBOURHOODS[neighbourhood].to_dict(),
    }
    return pd.DataFrame([row])


def denial_risk(bundle, applicant):
    """Probability of denial. Model C is a weighted mix of models, so we average them."""
    X = bundle["prep"].transform(applicant[bundle["features"] + CATEGORICAL_FEATURES])
    return float(sum(w * m.predict_proba(X)[0, 1] for w, m in zip(bundle["weights"], bundle["models"])))


def what_would_change(bundle, inputs):
    """Try simple, realistic changes one at a time; return the first that flips to approval."""
    tries = [(f"Lower debt-to-income to {d}%", {"dti": d}) for d in (40, 36, 33, 30, 25) if d < inputs["dti"]]
    tries += [(f"Borrow {int((1 - c) * 100)}% less (bigger down payment)",
               {"loan_amount": round(inputs["loan_amount"] * c)}) for c in (0.95, 0.9, 0.85)]
    if not inputs["coapplicant"]:
        tries.append(("Add a co-applicant", {"coapplicant": True}))
    for label, change in tries:
        if denial_risk(bundle, build_applicant(**{**inputs, **change})) < bundle["cutoff"]:
            return label
    # Nothing about her finances flips it. Check whether moving does: that's the point of the demo.
    for n in NEIGHBOURHOODS:
        if n != inputs["neighbourhood"] and denial_risk(bundle, build_applicant(**{**inputs, "neighbourhood": n})) < bundle["cutoff"]:
            return (f"Not lower debts, a smaller loan or a co-applicant. "
                    f"<b>Only moving to a “{n.lower()}” neighborhood</b> flips it.")
    return "None of the small changes we tried flips it"


# ----------------------------------------------------------------------------- rendering
ICON_OK = ('<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="currentColor"/>'
           '<path d="M6 10.3l2.6 2.6L14.2 7" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ICON_NO = ('<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="currentColor"/>'
           '<path d="M7 7l6 6M13 7l-6 6" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/></svg>')


def card(key, title, subtitle, risk, cutoff, advice):
    flagged = risk >= cutoff
    state = "no" if flagged else "ok"
    verdict = "Likely denied" if flagged else "Likely approved"
    scale = max(0.4, risk * 1.15)  # meter runs 0 .. scale
    fill, tick = min(risk / scale, 1) * 100, cutoff / scale * 100
    fix = (f'<div class="fl-fix"><span class="fl-fixk">What would change it</span>'
           f'<span class="fl-fixv">{advice}</span></div>') if flagged else ""
    return f"""
    <article class="fl-card fl-{state}" aria-label="{title}: {verdict}">
      <header><div class="fl-ct">{title}</div><div class="fl-cs">{subtitle}</div></header>
      <div class="fl-verdict">{ICON_NO if flagged else ICON_OK}<span>{verdict}</span></div>
      <div class="fl-meter" role="img" aria-label="Denial risk {risk:.0%}, flags at {cutoff:.0%}">
        <div class="fl-fill" style="width:{fill:.1f}%"></div>
        <div class="fl-tick" style="left:{tick:.1f}%"><span>flags at {cutoff:.0%}</span></div>
      </div>
      <div class="fl-risk"><b>{risk:.0%}</b> denial risk</div>
      {fix}
    </article>"""


def strip(inputs):
    """Both models across all three neighbourhoods for the same applicant."""
    head = "".join(f'<th scope="col">{n}<small>{NB_NOTE[n]}</small></th>' for n in NEIGHBOURHOODS)
    rows = ""
    for key, title, _, bundle in MODELS:
        cells, verdicts = "", []
        for n in NEIGHBOURHOODS:
            r = denial_risk(bundle, build_applicant(**inputs, neighbourhood=n))
            bad = r >= bundle["cutoff"]
            verdicts.append(bad)
            cells += (f'<td><span class="fl-chip fl-{"no" if bad else "ok"}">{ICON_NO if bad else ICON_OK}'
                      f'{"Denied" if bad else "Approved"} · {r:.0%}</span></td>')
        tag = '<span class="fl-tag fl-tag-flip">Changes with address</span>' if len(set(verdicts)) > 1 \
            else '<span class="fl-tag">Same everywhere</span>'
        rows += f'<tr><th scope="row">{title}{tag}</th>{cells}</tr>'
    return (f'<section class="fl-strip"><h3>Same applicant in every neighborhood</h3>'
            f'<div class="fl-scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></div></section>')


def judge(income, loan_amount, property_value, dti, loan_type, thirty_year, coapplicant,
          preapproval, neighbourhood):
    inputs = dict(income=income or 0, loan_amount=loan_amount or 0, property_value=property_value or 1,
                  dti=dti, loan_type=loan_type, thirty_year=thirty_year, coapplicant=coapplicant,
                  preapproval=preapproval)
    applicant = build_applicant(**inputs, neighbourhood=neighbourhood)
    cards = []
    for key, title, sub, bundle in MODELS:
        risk = denial_risk(bundle, applicant)
        advice = what_would_change(bundle, {**inputs, "neighbourhood": neighbourhood}) if risk >= bundle["cutoff"] else ""
        cards.append(card(key, title, sub, risk, bundle["cutoff"], advice))
    return f'<div class="fl-cards">{"".join(cards)}</div>', strip(inputs)


CSS = """
:root { --ink:#10223A; --muted:#55657A; --line:#D5DEE7; --paper:#FFFFFF; --ground:#EEF2F6;
        --ok:#1F7A4C; --ok-bg:#E7F4EC; --no:#B42F25; --no-bg:#FBEAE8; --focus:#2563EB; }
.gradio-container { background: var(--ground) !important; max-width: 1240px !important; margin: 0 auto !important;
  font-family: "IBM Plex Sans", system-ui, sans-serif !important; color: var(--ink); }
.fl-hero { padding: 28px 4px 8px; }
.fl-eyebrow { font-size: 14px; font-weight: 600; color: var(--muted); letter-spacing: .01em; }
.fl-hero h1 { font-size: clamp(30px, 4.2vw, 46px); line-height: 1.06; letter-spacing: -0.02em; font-weight: 700; margin: 8px 0 10px; color: var(--ink); }
.fl-hero p { font-size: 18px; line-height: 1.5; color: var(--muted); max-width: 760px; margin: 0; }
.fl-hero p b { color: var(--ink); }
.fl-panel { background: var(--paper) !important; border: 1px solid var(--line) !important; border-radius: 14px !important; padding: 18px !important; }
.fl-label { font-size: 13px; font-weight: 700; color: var(--muted); text-transform: none; margin: 2px 0 6px; }
/* neighbourhood segmented control */
#fl-nb .wrap { display: grid !important; grid-template-columns: repeat(3, 1fr); gap: 6px; background: #E2E8EF; padding: 5px; border-radius: 12px; }
#fl-nb label { justify-content: center; border: 0 !important; border-radius: 9px !important; padding: 12px 10px !important; min-height: 44px;
  font-size: 16px; font-weight: 600; color: var(--ink) !important; background: transparent !important; cursor: pointer; transition: background-color 150ms ease, transform 100ms ease-out; }
#fl-nb label:hover { background: rgba(255,255,255,.6) !important; }
#fl-nb label:active { transform: scale(0.97); }
#fl-nb label.selected { background: var(--paper) !important; box-shadow: 0 1px 3px rgba(16,34,58,.18); }
#fl-nb input { position: absolute; opacity: 0; }
#fl-nb label:has(input:focus-visible) { outline: 3px solid var(--focus); outline-offset: 2px; }
/* decision cards */
.fl-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 4px; }
.fl-card { background: var(--paper); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; border-top: 5px solid var(--state);
  transition: border-color 200ms ease; }
.fl-ok { --state: var(--ok); --state-bg: var(--ok-bg); }
.fl-no { --state: var(--no); --state-bg: var(--no-bg); }
.fl-ct { font-size: 18px; font-weight: 700; }
.fl-cs { font-size: 14px; color: var(--muted); margin-top: 2px; }
.fl-verdict { display: inline-flex; align-items: center; gap: 10px; margin: 16px 0 14px; padding: 8px 14px 8px 10px; border-radius: 999px;
  background: var(--state-bg); color: var(--state); font-size: 22px; font-weight: 700; }
.fl-verdict svg, .fl-chip svg { width: 24px; height: 24px; flex: none; }
.fl-meter { position: relative; height: 12px; background: #E6ECF2; border-radius: 6px; margin: 26px 0 8px; }
.fl-fill { position: absolute; inset: 0 auto 0 0; background: var(--state); border-radius: 6px; transition: width 250ms ease-out; }
.fl-tick { position: absolute; top: -8px; bottom: -8px; width: 2px; background: var(--ink); }
.fl-tick span { position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); white-space: nowrap; font-size: 12px; font-weight: 600; color: var(--muted); padding-bottom: 2px; }
.fl-risk { font-size: 15px; color: var(--muted); font-variant-numeric: tabular-nums; }
.fl-risk b { font-size: 26px; color: var(--ink); font-weight: 700; margin-right: 4px; }
.fl-fix { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); display: flex; flex-direction: column; gap: 2px; }
.fl-fixk { font-size: 13px; font-weight: 700; color: var(--muted); }
.fl-fixv { font-size: 16px; font-weight: 600; }
/* every-neighbourhood strip */
.fl-strip { background: var(--paper); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; margin-top: 14px; }
.fl-strip h3 { font-size: 16px; font-weight: 700; margin: 0 0 10px; color: var(--ink); }
.fl-scroll { overflow-x: auto; }
.fl-strip table { width: 100%; border-collapse: collapse; border: 0 !important; font-variant-numeric: tabular-nums; table-layout: fixed; }
.fl-strip th, .fl-strip td { text-align: left; padding: 10px 6px !important; border: 0 !important; border-top: 1px solid var(--line) !important;
  vertical-align: middle; background: transparent !important; }
.fl-strip th:first-child { width: 28%; }
.fl-strip thead, .fl-strip tbody, .fl-strip tr { border: 0 !important; }
.fl-strip thead th { border-top: 0 !important; }
.fl-strip thead th { border-top: 0; font-size: 14px; font-weight: 700; color: var(--ink); }
.fl-strip thead small { display: block; font-weight: 500; color: var(--muted); font-size: 12px; }
.fl-strip tbody th { font-size: 15px; font-weight: 700; color: var(--ink); white-space: nowrap; }
.fl-chip { display: inline-flex; align-items: center; gap: 5px; padding: 5px 9px 5px 5px; border-radius: 999px; font-size: 13px; font-weight: 700;
  background: var(--state-bg); color: var(--state); white-space: nowrap; }
.fl-chip svg { width: 18px; height: 18px; }
.fl-tag { display: block; width: fit-content; margin-top: 4px; font-size: 12px; font-weight: 600; color: var(--muted); }
.fl-tag-flip { color: var(--no); }
/* applicant form */
#fl-maya { min-height: 44px; font-weight: 600; transition: transform 100ms ease-out; }
#fl-maya:active { transform: scale(0.97); }
.fl-foot { font-size: 14px; color: var(--muted); line-height: 1.55; padding: 6px 4px 24px; }
.fl-foot b { color: var(--ink); }
@media (max-width: 760px) { .fl-cards { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

HEAD = (# The design is light-only (bank-paper look): force Gradio's light theme even on dark-mode computers.
        '<script>if(!/__theme=light/.test(location.search)){const u=new URL(location);u.searchParams.set("__theme","light");location.replace(u)}</script>'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">')

with gr.Blocks(title="FairLens") as demo:
    gr.HTML("""<div class="fl-hero">
      <div class="fl-eyebrow">FairLens · 264,340 real Georgia mortgage applications (HMDA 2022–2023)</div>
      <h1>Same applicant. Only the address changes.</h1>
      <p>Two mortgage models judge the same person. <b>Neither one ever sees race.</b>
         Switch the neighborhood and watch which model changes its mind.</p></div>""")
    with gr.Row(equal_height=False):
        with gr.Column(scale=7):
            neighbourhood = gr.Radio(list(NEIGHBOURHOODS), value="Mostly white", show_label=False, elem_id="fl-nb",
                                     container=False)
            out_cards = gr.HTML()
            out_strip = gr.HTML()
        with gr.Column(scale=4, elem_classes="fl-panel"):
            gr.HTML('<div class="fl-label">The applicant</div>')
            maya = gr.Button("Load the borderline example", elem_id="fl-maya", variant="secondary")
            income = gr.Number(MAYA["income"], label="Yearly income ($ thousands)", minimum=1)
            with gr.Row():
                loan_amount = gr.Number(MAYA["loan_amount"], label="Loan ($K)", minimum=1)
                property_value = gr.Number(MAYA["property_value"], label="Home price ($K)", minimum=1)
            dti = gr.Slider(10, 65, MAYA["dti"], step=1, label="Monthly debts as % of income (DTI)")
            loan_type = gr.Dropdown(list(LOAN_TYPES), value=MAYA["loan_type"], label="Loan type")
            thirty_year = gr.Checkbox(MAYA["thirty_year"], label="30-year mortgage")
            coapplicant = gr.Checkbox(MAYA["coapplicant"], label="Has a co-applicant")
            preapproval = gr.Checkbox(MAYA["preapproval"], label="Asked for pre-approval")
    gr.HTML("""<div class="fl-foot">On 2023 test data the standard model wrongly flagged qualified Black applicants
      <b>2.5×</b> as often as qualified white applicants, mostly through neighborhood data. The fairness-checked model brought
      that to <b>1.05×</b>, with no measurable gap, while keeping 97% of the accuracy.<br>
      Prototype built for the Veritas AI High School Hackathon. Not financial advice.</div>""")

    fields = [income, loan_amount, property_value, dti, loan_type, thirty_year, coapplicant, preapproval, neighbourhood]
    for comp in fields:
        comp.change(judge, fields, [out_cards, out_strip])
    maya.click(lambda: [MAYA[k] for k in ("income", "loan_amount", "property_value", "dti", "loan_type",
                                          "thirty_year", "coapplicant", "preapproval")],
               None, fields[:-1])
    demo.load(judge, fields, [out_cards, out_strip])

if __name__ == "__main__":
    demo.launch(css=CSS, head=HEAD, theme=gr.themes.Base(primary_hue="slate", radius_size="lg"))
