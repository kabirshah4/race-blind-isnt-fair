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
           '<path d="M6 10.3l2.6 2.6L14.2 7" fill="none" stroke="#06110C" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ICON_NO = ('<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="currentColor"/>'
           '<path d="M7 7l6 6M13 7l-6 6" fill="none" stroke="#1A0707" stroke-width="2.2" stroke-linecap="round"/></svg>')
ICON_INFO = ('<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/>'
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
      <div class="fl-verdict">{ICON_NO if flagged else ICON_OK}<span>{verdict}</span></div>
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
            f'<td><span class="fl-chip fl-{"no" if r >= c else "ok"}">{ICON_NO if r >= c else ICON_OK}'
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
               "Press <b>Load the borderline example</b> to see a case where it does.")
    return f'<div class="fl-banner glass" role="status">{ICON_INFO}<p><span class="fl-bk">What you’re seeing</span>{msg}</p></div>'


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
        cards.append(card(title, sub, risk, bundle["cutoff"], advice))
    grid = everywhere(inputs)
    return (banner(grid, neighbourhood),
            f'{step(2, "Compare the two AIs. Neither one ever sees race.")}<div class="fl-cards">{"".join(cards)}</div>',
            strip(grid))


CSS = """
/* Liquid glass, dark. Glass = translucent fill + backdrop blur + bright top edge + deep shadow,
   floating over soft colour glows so the blur has something to refract. */
:root, .dark {
  --ink:#EEF3F9; --muted:#A3B1C4; --line:rgba(255,255,255,.12);
  --ok:#3EE59A; --ok-bg:rgba(62,229,154,.14); --no:#FF7A70; --no-bg:rgba(255,122,112,.15); --focus:#7CC4FF;
  --body-background-fill:#070B14; --background-fill-primary:transparent; --background-fill-secondary:rgba(255,255,255,.05);
  --block-background-fill:transparent; --block-border-color:transparent; --block-border-width:0px; --block-label-text-color:var(--muted);
  --block-title-text-color:var(--muted); --body-text-color:var(--ink); --body-text-color-subdued:var(--muted);
  --input-background-fill:rgba(255,255,255,.07); --input-border-color:rgba(255,255,255,.14); --input-border-color-focus:var(--focus);
  --border-color-primary:rgba(255,255,255,.12); --checkbox-background-color:rgba(255,255,255,.08); --checkbox-border-color:rgba(255,255,255,.3);
  --checkbox-background-color-selected:#7CC4FF; --slider-color:#7CC4FF; --neutral-200:rgba(255,255,255,.18);
  --button-secondary-background-fill:rgba(255,255,255,.12); --button-secondary-background-fill-hover:rgba(255,255,255,.18);
  --button-secondary-text-color:var(--ink); --button-secondary-border-color:rgba(255,255,255,.2);
}
body, .gradio-container, gradio-app {
  background: radial-gradient(900px 560px at 12% 6%, rgba(56,152,248,.28), transparent 62%),
              radial-gradient(760px 520px at 88% 22%, rgba(150,110,255,.24), transparent 62%),
              radial-gradient(900px 620px at 45% 104%, rgba(40,200,150,.18), transparent 62%), #070B14 !important;
  background-attachment: fixed !important; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; font-family: "IBM Plex Sans", system-ui, sans-serif !important; color: var(--ink); }
.glass { background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.04)) !important;
  -webkit-backdrop-filter: blur(26px) saturate(170%); backdrop-filter: blur(26px) saturate(170%);
  border: 1px solid rgba(255,255,255,.13) !important; border-radius: 22px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.22), 0 24px 60px rgba(0,0,0,.40) !important; }
.fl-hero { padding: 30px 6px 6px; }
.fl-eyebrow { font-size: 14px; font-weight: 600; color: var(--muted); }
.fl-hero h1 { font-size: clamp(32px, 4.4vw, 50px); line-height: 1.04; letter-spacing: -0.025em; font-weight: 700; margin: 8px 0 10px; color: #fff; }
.fl-hero p { font-size: 18px; line-height: 1.5; color: var(--muted); max-width: 780px; margin: 0; }
.fl-hero p b { color: #fff; }
.fl-banner { display: flex; gap: 14px; align-items: flex-start; padding: 16px 20px; margin: 6px 0 4px; color: #BFE3FF; }
.fl-banner svg { width: 24px; height: 24px; flex: none; margin-top: 2px; }
.fl-banner p { margin: 0; font-size: 17px; line-height: 1.5; color: var(--ink); }
.fl-banner .fl-bk { display: block; font-size: 13px; font-weight: 700; color: #9CD2FF; margin-bottom: 2px; }
.fl-banner .ok { color: var(--ok); font-weight: 700; } .fl-banner .no { color: var(--no); font-weight: 700; }
.fl-step { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 700; color: var(--ink); margin: 6px 2px 10px; }
.fl-step span { display: grid; place-items: center; width: 26px; height: 26px; border-radius: 50%; background: rgba(124,196,255,.18);
  color: #BFE3FF; font-size: 14px; border: 1px solid rgba(124,196,255,.35); }
/* step 1: neighbourhood segmented control (a glass track with a brighter glass thumb) */
#fl-nb .wrap { display: grid !important; grid-template-columns: repeat(3, 1fr); gap: 6px; padding: 6px; border-radius: 18px;
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); box-shadow: inset 0 1px 0 rgba(255,255,255,.12); }
#fl-nb label { justify-content: center; border: 0 !important; border-radius: 13px !important; padding: 13px 10px !important; min-height: 48px;
  font-size: 16px; font-weight: 600; color: var(--muted) !important; background: transparent !important; cursor: pointer;
  transition: background-color 160ms ease, color 160ms ease, transform 100ms ease-out; }
#fl-nb label:hover { color: var(--ink) !important; background: rgba(255,255,255,.06) !important; }
#fl-nb label:active { transform: scale(0.97); }
#fl-nb label.selected { color: #fff !important; background: rgba(255,255,255,.18) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.35), 0 6px 18px rgba(0,0,0,.30); }
#fl-nb input { position: absolute; opacity: 0; }
#fl-nb label:has(input:focus-visible) { outline: 3px solid var(--focus); outline-offset: 2px; }
/* step 2: the two decisions */
.fl-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.fl-card { padding: 20px 22px !important; position: relative; overflow: hidden; }
.fl-card::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: var(--state); opacity: .9; }
.fl-ok { --state: var(--ok); --state-bg: var(--ok-bg); }
.fl-no { --state: var(--no); --state-bg: var(--no-bg); }
.fl-ct { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.fl-cs { font-size: 14px; color: var(--muted); margin-top: 2px; }
.fl-verdict { display: inline-flex; align-items: center; gap: 10px; margin: 16px 0 14px; padding: 9px 16px 9px 10px; border-radius: 999px;
  background: var(--state-bg); color: var(--state); font-size: 24px; font-weight: 700; border: 1px solid color-mix(in srgb, var(--state) 35%, transparent); }
.fl-verdict svg, .fl-chip svg { width: 24px; height: 24px; flex: none; }
.fl-mk { font-size: 15px; color: var(--muted); font-variant-numeric: tabular-nums; }
.fl-mk b { color: #fff; font-size: 20px; }
.fl-meter { position: relative; height: 12px; background: rgba(255,255,255,.10); border-radius: 6px; margin: 10px 0 6px; }
.fl-fill { position: absolute; inset: 0 auto 0 0; background: var(--state); border-radius: 6px; box-shadow: 0 0 0 1px rgba(0,0,0,.2);
  transition: width 260ms cubic-bezier(.2,.8,.2,1); }
.fl-tick { position: absolute; top: -6px; bottom: -6px; width: 2px; background: #fff; border-radius: 1px; }
.fl-scale { position: relative; height: 18px; font-size: 12px; color: var(--muted); }
.fl-tlab { position: absolute; transform: translateX(-50%); white-space: nowrap; font-weight: 600; color: var(--ink); }
.fl-fix { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line); display: flex; flex-direction: column; gap: 3px; }
.fl-fixk { font-size: 13px; font-weight: 700; color: var(--muted); }
.fl-fixv { font-size: 16px; font-weight: 500; line-height: 1.4; color: var(--ink); }
.fl-fixv b { color: #fff; }
/* step 3: every address at once */
.fl-strip { padding: 16px 18px !important; margin-top: 16px; }
.fl-scroll { overflow-x: auto; }
.fl-strip table { width: 100%; border-collapse: collapse; border: 0 !important; font-variant-numeric: tabular-nums; table-layout: fixed; background: transparent !important; }
.fl-strip thead, .fl-strip tbody, .fl-strip tr { border: 0 !important; background: transparent !important; }
.fl-strip th, .fl-strip td { text-align: left; padding: 11px 6px !important; border: 0 !important; border-top: 1px solid var(--line) !important;
  vertical-align: middle; background: transparent !important; color: var(--ink); }
.fl-strip thead th { border-top: 0 !important; font-size: 14px; font-weight: 700; }
.fl-strip thead small { display: block; font-weight: 500; color: var(--muted); font-size: 12px; }
.fl-strip th:first-child { width: 30%; }
.fl-strip tbody th { font-size: 15px; font-weight: 700; color: #fff; }
.fl-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px 5px 5px; border-radius: 999px; font-size: 14px; font-weight: 700;
  background: var(--state-bg); color: var(--state); border: 1px solid color-mix(in srgb, var(--state) 30%, transparent); white-space: nowrap; }
.fl-chip svg { width: 18px; height: 18px; }
.fl-chip em { font-style: normal; font-weight: 500; opacity: .85; }
.fl-tag { display: block; width: fit-content; margin-top: 4px; font-size: 12px; font-weight: 600; color: var(--muted); }
.fl-tag.flip { color: var(--no); }
/* applicant panel */
.fl-panel { padding: 18px !important; }
.fl-panel .block, .fl-panel .form { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.fl-ptitle { font-size: 17px; font-weight: 700; color: #fff; }
.fl-psub { font-size: 14px; color: var(--muted); margin: 2px 0 6px; line-height: 1.45; }
#fl-maya { min-height: 46px; font-weight: 600; border-radius: 14px !important; transition: transform 100ms ease-out, background-color 150ms ease; }
#fl-maya:active { transform: scale(0.97); }
.fl-foot { font-size: 14px; color: var(--muted); line-height: 1.55; padding: 10px 6px 28px; }
.fl-foot b { color: #fff; }
footer { opacity: .6; }
@media (max-width: 760px) { .fl-cards { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (prefers-reduced-transparency: reduce) {
  .glass, #fl-nb .wrap { background: #121A2A !important; -webkit-backdrop-filter: none; backdrop-filter: none; } }
"""

HEAD = (# The design is dark-only (liquid glass): force Gradio's dark theme so its own widgets match.
        '<script>if(!/__theme=dark/.test(location.search)){const u=new URL(location);u.searchParams.set("__theme","dark");location.replace(u)}</script>'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">')

with gr.Blocks(title="FairLens") as demo:
    gr.HTML("""<div class="fl-hero">
      <div class="fl-eyebrow">FairLens · trained on 264,340 real Georgia mortgage applications</div>
      <h1>Same person. Different address.<br>Does the AI change its answer?</h1>
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
            gr.HTML('<div class="fl-ptitle">The applicant</div><div class="fl-psub">Already filled in with a real borderline case. '
                    'Change anything and both AIs update instantly.</div>')
            maya = gr.Button("Load the borderline example", elem_id="fl-maya", variant="secondary")
            income = gr.Number(MAYA["income"], label="Yearly income ($ thousands)", minimum=1)
            with gr.Row():
                loan_amount = gr.Number(MAYA["loan_amount"], label="Loan ($K)", minimum=1)
                property_value = gr.Number(MAYA["property_value"], label="Home price ($K)", minimum=1)
            dti = gr.Slider(10, 65, MAYA["dti"], step=1, label="Monthly debts as % of income")
            loan_type = gr.Dropdown(list(LOAN_TYPES), value=MAYA["loan_type"], label="Loan type")
            thirty_year = gr.Checkbox(MAYA["thirty_year"], label="30-year mortgage")
            coapplicant = gr.Checkbox(MAYA["coapplicant"], label="Has a co-applicant")
            preapproval = gr.Checkbox(MAYA["preapproval"], label="Asked for pre-approval")
    gr.HTML("""<div class="fl-foot">Tested on 117,547 real 2023 applications: the typical AI wrongly flagged qualified Black applicants
      <b>2.5×</b> as often as qualified white applicants, mostly through neighborhood data. Our fixed AI brought that to
      <b>1.05×</b> (no measurable gap) while keeping 97% of the accuracy.<br>
      Prototype built for the Veritas AI High School Hackathon. Not financial advice.</div>""")

    fields = [income, loan_amount, property_value, dti, loan_type, thirty_year, coapplicant, preapproval, neighbourhood]
    outs = [out_banner, out_cards, out_strip]
    for comp in fields:
        comp.change(judge, fields, outs)
    maya.click(lambda: [MAYA[k] for k in ("income", "loan_amount", "property_value", "dti", "loan_type",
                                          "thirty_year", "coapplicant", "preapproval")],
               None, fields[:-1])
    demo.load(judge, fields, outs)

if __name__ == "__main__":
    demo.launch(css=CSS, head=HEAD, theme=gr.themes.Base(primary_hue="sky", neutral_hue="slate", radius_size="lg"))
