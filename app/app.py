"""FairLens demo: the same mortgage applicant, judged by two race-blind models.

Run:  .venv/bin/python app/app.py      then open http://127.0.0.1:7860
Needs models/model_A_full.joblib and models/model_C_fair.joblib (made by src/fix.py)
and data/processed/hmda_ga.parquet (made by src/prep.py).

What a judge should get in five seconds
---------------------------------------
* The headline says it: same applicant, only the address changes.
* The neighbourhood switch sits right above the two decisions it changes.
* Two decision cards: the "typical bank AI" (model A, uses neighbourhood data) and
  "our fixed AI" (model C, no neighbourhood data + fairness training).
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
    ("standard", "Typical bank AI", "Uses the neighborhood · never sees race", MODEL_A),
    ("fair", "Our fixed AI", "Ignores the neighborhood · fairness-trained", MODEL_C),
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

# The example applicant is CONSTRUCTED (not a real person): one of 174 combinations we
# tested where only the address flips model A. Chosen for the widest margin on both sides.
MAYA = dict(income=60, loan_amount=100, property_value=125, dti=42, loan_type="Conventional",
            thirty_year=True, coapplicant=False)


# ----------------------------------------------------------------------------- model logic
def build_applicant(income, loan_amount, property_value, dti, loan_type, thirty_year,
                    coapplicant, neighbourhood):
    """Turn the form inputs into one row with the same columns the models were trained on."""
    row = {
        "income": income, "loan_amount": loan_amount, "property_value": property_value,
        "ltv": min(loan_amount / max(property_value, 1) * 100, 150), "dti": dti,
        "loan_term_360": int(thirty_year), "has_coapplicant": int(coapplicant),
        "loan_type": LOAN_TYPES[loan_type], "conforming_loan_limit": "C" if loan_amount <= 766 else "NC",
        "submission_of_application": "1",
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
           '<path d="M6 10.3l2.6 2.6L14.2 7" fill="none" stroke="#06110C" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ICON_NO = ('<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="currentColor"/>'
           '<path d="M7 7l6 6M13 7l-6 6" fill="none" stroke="#1A0707" stroke-width="2.2" stroke-linecap="round"/></svg>')
_UNUSED_ICON = ('<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/>'
             '<path d="M10 9v5M10 6.2v.2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>')


def step(n, text):
    return f'<div class="fl-step"><span>{n}</span>{text}</div>'


def card(title, subtitle, risk, cutoff, advice):
    flagged = risk >= cutoff
    state = "no" if flagged else "ok"
    verdict = "Would deny" if flagged else "Would approve"
    scale = max(0.4, risk * 1.15)  # bar runs 0 .. scale
    fill, tick = min(risk / scale, 1) * 100, cutoff / scale * 100
    fix = (f'<div class="fl-fix"><span class="fl-fixk">What would change its mind?</span>'
           f'<span class="fl-fixv">{advice}</span></div>') if flagged else ""
    return f"""
    <article class="fl-card glass fl-{state}" aria-label="{title}: {verdict}">
      <div class="fl-ct">{title}</div><div class="fl-cs">{subtitle}</div>
      <div class="fl-verdict"><i></i>{verdict}</div>
      <div class="fl-mk">Chance this AI says deny: <b>{risk:.0%}</b></div>
      <div class="fl-meter" role="img" aria-label="Chance of denial {risk:.0%}; this AI denies above {cutoff:.0%}">
        <div class="fl-fill" style="width:{fill:.1f}%"></div>
        <div class="fl-tick" style="left:{tick:.1f}%"></div>
      </div>
      <div class="fl-scale"><span>0%</span><span class="fl-tlab" style="left:{tick:.1f}%">denies above {cutoff:.0%}</span></div>
      {fix}
    </article>"""


def everywhere(inputs):
    """Verdict of each model in each neighbourhood, for the same applicant."""
    out = {}
    for key, title, _, bundle in MODELS:
        out[key] = [(n, denial_risk(bundle, build_applicant(**inputs, neighbourhood=n)), bundle["cutoff"]) for n in NEIGHBOURHOODS]
    return out


def strip(grid):
    head = "".join(f'<th scope="col">{n}<small>{NB_NOTE[n]}</small></th>' for n in NEIGHBOURHOODS)
    rows = ""
    for key, title, _, _ in MODELS:
        cells = "".join(
            f'<td><span class="fl-chip fl-{"no" if r >= c else "ok"}"><i></i>'
            f'{"Deny" if r >= c else "Approve"}<em>{r:.0%}</em></span></td>' for n, r, c in grid[key])
        flips = len({r >= c for _, r, c in grid[key]}) > 1
        tag = '<span class="fl-tag flip">Answer changes with the address</span>' if flips else '<span class="fl-tag">Same answer everywhere</span>'
        rows += f'<tr><th scope="row">{title}{tag}</th>{cells}</tr>'
    return (f'<section class="fl-strip glass">{step(3, "See this person at every address at once")}'
            f'<div class="fl-scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></div></section>')


def banner(grid, neighbourhood):
    """One plain sentence explaining what the screen shows right now."""
    a_flips = len({r >= c for _, r, c in grid["standard"]}) > 1
    here = {k: dict((n, r >= c) for n, r, c in grid[k]) for k in grid}
    if a_flips:
        now = ""
        if here["standard"][neighbourhood] and not here["fair"][neighbourhood]:
            now = f' At a <b>{neighbourhood.lower()}</b> address, the typical AI says <span class="no">deny</span> and our fixed AI says <span class="ok">approve</span>.'
        msg = (f'<b>The typical bank AI changes its answer just because of the address.</b> '
               f'Our fixed AI gives this person the same answer everywhere.{now}')
    else:
        msg = ("For this applicant, neither AI changes its answer with the address. "
               "Press <b>Reset to the example applicant</b> to see a case where it does.")
    return f'<div class="fl-banner" role="status"><span class="fl-bk">What you’re seeing</span><p>{msg}</p></div>'


def judge(income, loan_amount, property_value, dti, loan_type, thirty_year, coapplicant, neighbourhood):
    inputs = dict(income=income or 0, loan_amount=loan_amount or 0, property_value=property_value or 1,
                  dti=dti, loan_type=loan_type, thirty_year=thirty_year, coapplicant=coapplicant)
    applicant = build_applicant(**inputs, neighbourhood=neighbourhood)
    cards = []
    for key, title, sub, bundle in MODELS:
        risk = denial_risk(bundle, applicant)
        advice = what_would_change(bundle, {**inputs, "neighbourhood": neighbourhood}) if risk >= bundle["cutoff"] else ""
        cards.append(card(title, sub, risk, bundle["cutoff"], advice))
    grid = everywhere(inputs)
    return (banner(grid, neighbourhood),
            f'{step(2, "Compare the two AIs. Neither one ever sees race.")}<div class="fl-cards">{"".join(cards)}</div>',
            strip(grid))


CSS = """
/* Quiet dark. No blur, no glows: flat surfaces, hairline borders, small status dots.
   Everything here is cheap to paint, so scrolling and updates stay smooth. */
:root, .dark {
  --bg:#0B0E13; --surface:#11151C; --ink:#E8EDF3; --muted:#8B97A8; --line:#1F2630; --line2:#2A323E;
  --ok:#55C28E; --no:#EC7A6F; --accent:#8AB4F8; --focus:#8AB4F8;
  --body-background-fill:var(--bg); --background-fill-primary:var(--surface); --background-fill-secondary:var(--surface);
  --block-background-fill:transparent; --block-border-color:transparent; --block-border-width:0px; --block-label-text-color:var(--muted);
  --block-title-text-color:var(--muted); --body-text-color:var(--ink); --body-text-color-subdued:var(--muted);
  --input-background-fill:#0E1218; --input-border-color:var(--line2); --input-border-color-focus:var(--focus); --input-shadow:none;
  --border-color-primary:var(--line); --checkbox-background-color:#0E1218; --checkbox-border-color:#3A4452;
  --checkbox-background-color-selected:var(--accent); --slider-color:var(--accent); --neutral-200:var(--line2);
  --button-secondary-background-fill:transparent; --button-secondary-background-fill-hover:#171C24;
  --button-secondary-text-color:var(--ink); --button-secondary-border-color:var(--line2); --shadow-drop:none; --shadow-drop-lg:none;
}
body, gradio-app, .gradio-container { background: var(--bg) !important; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; font-family: "IBM Plex Sans", system-ui, sans-serif !important;
  color: var(--ink); -webkit-font-smoothing: antialiased; }
.glass { background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; box-shadow: none !important; }
.fl-hero { padding: 34px 2px 4px; }
.fl-eyebrow { font-size: 13px; font-weight: 500; color: var(--muted); }
.fl-hero h1 { font-size: clamp(28px, 3.4vw, 40px); line-height: 1.12; letter-spacing: -0.02em; font-weight: 600; margin: 10px 0 10px; color: #fff; }
.fl-hero p { font-size: 17px; line-height: 1.55; color: var(--muted); max-width: 720px; margin: 0; font-weight: 400; }
.fl-hero p b { color: var(--ink); font-weight: 500; }
.fl-banner { padding: 14px 2px 6px; border-top: 1px solid var(--line); margin-top: 18px; }
.fl-banner .fl-bk { font-size: 12px; font-weight: 500; color: var(--muted); letter-spacing: .02em; }
.fl-banner p { margin: 4px 0 0; font-size: 16px; line-height: 1.55; color: var(--ink); font-weight: 400; max-width: 900px; }
.fl-banner b { font-weight: 600; color: #fff; }
.fl-banner .ok { color: var(--ok); font-weight: 600; } .fl-banner .no { color: var(--no); font-weight: 600; }
.fl-step { display: flex; align-items: baseline; gap: 8px; font-size: 14px; font-weight: 500; color: var(--ink); margin: 14px 0 8px; }
.fl-step span { color: var(--muted); font-variant-numeric: tabular-nums; }
.fl-step span::after { content: "."; }
/* neighbourhood switch: a thin segmented control */
#fl-nb .wrap { display: grid !important; grid-template-columns: repeat(3, 1fr); gap: 2px; padding: 3px; border-radius: 10px;
  background: var(--surface); border: 1px solid var(--line); }
#fl-nb label { justify-content: center; border: 0 !important; border-radius: 7px !important; padding: 10px 8px !important; min-height: 44px;
  font-size: 15px; font-weight: 500; color: var(--muted) !important; background: transparent !important; cursor: pointer; box-shadow: none !important;
  transition: background-color 120ms ease, color 120ms ease; }
#fl-nb label:hover { color: var(--ink) !important; }
#fl-nb label:active { transform: scale(0.98); }
#fl-nb label.selected { color: #fff !important; background: #1E2530 !important; }
#fl-nb input { position: absolute; opacity: 0; }
#fl-nb label:has(input:focus-visible) { outline: 2px solid var(--focus); outline-offset: 2px; }
/* the two decisions */
.fl-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.fl-card { padding: 18px 20px !important; }
.fl-ok { --state: var(--ok); } .fl-no { --state: var(--no); }
.fl-ct { font-size: 16px; font-weight: 600; color: #fff; }
.fl-cs { font-size: 13px; color: var(--muted); margin-top: 2px; }
.fl-verdict { display: flex; align-items: center; gap: 9px; margin: 18px 0 14px; font-size: 22px; font-weight: 600; color: var(--state); letter-spacing: -0.01em; }
.fl-verdict i, .fl-chip i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--state); flex: none; }
.fl-mk { font-size: 14px; color: var(--muted); font-variant-numeric: tabular-nums; }
.fl-mk b { color: var(--ink); font-weight: 600; }
.fl-meter { position: relative; height: 4px; background: var(--line2); border-radius: 2px; margin: 10px 0 6px; }
.fl-fill { position: absolute; inset: 0 auto 0 0; background: var(--state); border-radius: 2px; }
.fl-tick { position: absolute; top: -5px; bottom: -5px; width: 1.5px; background: var(--ink); }
.fl-scale { position: relative; height: 16px; font-size: 12px; color: var(--muted); }
.fl-tlab { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.fl-fix { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line); }
.fl-fixk { display: block; font-size: 12px; font-weight: 500; color: var(--muted); }
.fl-fixv { display: block; font-size: 15px; line-height: 1.45; color: var(--ink); margin-top: 2px; }
.fl-fixv b { color: #fff; font-weight: 600; }
/* every address at once */
.fl-strip { padding: 4px 18px 8px !important; margin-top: 12px; }
.fl-scroll { overflow-x: auto; }
.fl-strip table { width: 100%; border-collapse: collapse; border: 0 !important; font-variant-numeric: tabular-nums; table-layout: fixed; background: transparent !important; }
.fl-strip thead, .fl-strip tbody, .fl-strip tr { border: 0 !important; background: transparent !important; }
.fl-strip th, .fl-strip td { text-align: left; padding: 11px 6px !important; border: 0 !important; border-top: 1px solid var(--line) !important;
  vertical-align: middle; background: transparent !important; color: var(--ink); }
.fl-strip thead th { border-top: 0 !important; font-size: 13px; font-weight: 500; }
.fl-strip thead small { display: block; font-weight: 400; color: var(--muted); font-size: 12px; }
.fl-strip th:first-child { width: 30%; }
.fl-strip tbody th { font-size: 14px; font-weight: 600; color: #fff; }
.fl-chip { display: inline-flex; align-items: center; gap: 7px; font-size: 14px; font-weight: 500; color: var(--state); white-space: nowrap; }
.fl-chip em { font-style: normal; color: var(--muted); font-weight: 400; margin-left: 2px; }
.fl-tag { display: block; margin-top: 3px; font-size: 12px; font-weight: 400; color: var(--muted); }
.fl-tag.flip { color: var(--no); }
/* applicant panel */
.fl-panel { padding: 18px !important; }
.fl-panel .block, .fl-panel .form { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.fl-ptitle { font-size: 15px; font-weight: 600; color: #fff; }
.fl-psub { font-size: 13px; color: var(--muted); margin: 3px 0 6px; line-height: 1.5; }
#fl-maya { min-height: 44px; font-weight: 500; border-radius: 9px !important; box-shadow: none !important; border: 1px solid var(--line2) !important; background: transparent !important; }
#fl-maya:hover { background: #171C24 !important; }
#fl-maya:active { transform: scale(0.98); }
.fl-foot { font-size: 13px; color: var(--muted); line-height: 1.6; padding: 18px 2px 28px; border-top: 1px solid var(--line); margin-top: 18px; }
.fl-foot b { color: var(--ink); font-weight: 600; }
footer { opacity: .5; }
@media (max-width: 760px) { .fl-cards { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

HEAD = (# The design is dark-only (liquid glass): force Gradio's dark theme so its own widgets match.
        '<script>if(!/__theme=dark/.test(location.search)){const u=new URL(location);u.searchParams.set("__theme","dark");location.replace(u)}</script>'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">')

with gr.Blocks(title="FairLens") as demo:
    gr.HTML("""<div class="fl-hero">
      <div class="fl-eyebrow">FairLens · trained on 264,340 real Georgia mortgage applications</div>
      <h1>Same person, different address.<br>Does the AI change its answer?</h1>
      <p>Banks use AI to approve mortgages. These two AIs are <b>never told anyone's race</b>,
         but one of them is told the neighborhood. Watch what that does.</p></div>""")
    out_banner = gr.HTML()
    with gr.Row(equal_height=False):
        with gr.Column(scale=7):
            gr.HTML(step(1, "Pick where the home is"))
            neighbourhood = gr.Radio(list(NEIGHBOURHOODS), value="Mostly minority", show_label=False, elem_id="fl-nb",
                                     container=False)
            out_cards = gr.HTML()
            out_strip = gr.HTML()
        with gr.Column(scale=4, elem_classes=["fl-panel", "glass"]):
            gr.HTML('<div class="fl-ptitle">The applicant</div><div class="fl-psub">Filled in with a constructed borderline example (not a real person). '
                    'Change anything and both AIs update instantly.</div>')
            maya = gr.Button("Reset to the example applicant", elem_id="fl-maya", variant="secondary")
            income = gr.Number(MAYA["income"], label="Yearly income ($ thousands)", minimum=1)
            with gr.Row():
                loan_amount = gr.Number(MAYA["loan_amount"], label="Loan ($K)", minimum=1)
                property_value = gr.Number(MAYA["property_value"], label="Home price ($K)", minimum=1)
            dti = gr.Slider(10, 65, MAYA["dti"], step=1, label="Monthly debts as % of income")
            loan_type = gr.Dropdown(list(LOAN_TYPES), value=MAYA["loan_type"], label="Loan type")
            thirty_year = gr.Checkbox(MAYA["thirty_year"], label="30-year mortgage")
            coapplicant = gr.Checkbox(MAYA["coapplicant"], label="Has a co-applicant")
    gr.HTML("""<div class="fl-foot">Our fixed AI can’t see the neighborhood at all, so here its answer can’t change with the address by design.
      What matters is the result across all 117,547 real 2023 applications: the typical AI wrongly flagged qualified Black applicants
      <b>3.0×</b> as often as qualified white applicants; our fixed AI brought that to <b>1.2×</b> while keeping 97% of the PR-AUC score.
      The trade-off: qualified white applicants are flagged a little more often (4.0% → 5.6%).<br>
      Prototype built for the Veritas AI High School Hackathon. Not financial advice.</div>""")

    fields = [income, loan_amount, property_value, dti, loan_type, thirty_year, coapplicant, neighbourhood]
    outs = [out_banner, out_cards, out_strip]
    # Speed: sliders update on release; every event drops stale calls and skips the loading overlay.
    for comp in fields:
        if comp is dti:
            comp.release(judge, fields, outs, show_progress="hidden", trigger_mode="always_last")
        else:
            comp.change(judge, fields, outs, show_progress="hidden", trigger_mode="always_last")
    maya.click(lambda: [MAYA[k] for k in ("income", "loan_amount", "property_value", "dti", "loan_type",
                                          "thirty_year", "coapplicant")],
               None, fields[:-1])
    demo.load(judge, fields, outs, show_progress="hidden")

if __name__ == "__main__":
    demo.launch(css=CSS, head=HEAD, theme=gr.themes.Base(primary_hue="sky", neutral_hue="slate", radius_size="lg"))
