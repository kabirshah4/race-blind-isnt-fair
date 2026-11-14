"""FairView demo: the same mortgage applicant, judged by two race-blind models.

Run:  .venv/bin/python app/app.py      then open http://127.0.0.1:7860
Needs models/model_A_full.joblib and models/model_C_fair.joblib (made by src/fix.py)
and models/neighbourhoods.json (all made by src/fix.py and included in the repo).

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
import json
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

# Three typical neighbourhood profiles from Georgia's 2023 data (medians of the 7 tract
# numbers in each band). src/fix.py writes this file, so the app needs no data download.
NEIGHBOURHOODS = {k: pd.Series(v) for k, v in
                  json.loads((ROOT / "models" / "neighbourhoods.json").read_text()).items()}
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
/* Sky: a navy sky that fades into white. Navy is the only colour on actions; serif for big moments. */
:root, .dark {
  --serif: "New York", ui-serif, "Iowan Old Style", Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
  --bg:#FBFBFD; --surface:#FFFFFF; --ink:#0B1A33; --muted:#5A6780; --line:#E3E7EF; --line2:#D5DBE6;
  --navy:#0B2A5B; --ok:#127A5B; --no:#B4233C; --focus:#2C5AA0;
  --shadow: 0 1px 2px rgba(11,26,51,.04), 0 12px 32px -12px rgba(11,26,51,.12);
  --body-background-fill:var(--bg); --background-fill-primary:var(--surface); --background-fill-secondary:#F5F7FB;
  --block-background-fill:transparent; --block-border-color:transparent; --block-border-width:0px; --block-label-text-color:var(--muted);
  --block-title-text-color:var(--muted); --block-label-background-fill:transparent; --body-text-color:var(--ink); --body-text-color-subdued:var(--muted);
  --input-background-fill:#F5F7FB; --input-background-fill-focus:#FFFFFF; --input-border-color:var(--line); --input-border-color-focus:var(--focus); --input-shadow:none;
  --input-radius:12px; --border-color-primary:var(--line); --checkbox-background-color:#FFFFFF; --checkbox-border-color:#B9C2D3;
  --checkbox-background-color-selected:var(--navy); --checkbox-border-color-selected:var(--navy); --slider-color:var(--navy); --neutral-200:var(--line2);
  --button-secondary-background-fill:transparent; --button-secondary-background-fill-hover:#F5F7FB;
  --button-secondary-text-color:var(--ink); --button-secondary-border-color:var(--line); --shadow-drop:none; --shadow-drop-lg:none;
  color-scheme: light;
}
body, gradio-app, .gradio-container { background: var(--bg) !important; }
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; font-family: var(--sans) !important;
  color: var(--ink); -webkit-font-smoothing: antialiased; padding-top: 0 !important; width: 100% !important; min-width: 0 !important; }
gradio-app .main, gradio-app .wrap, gradio-app main.contain { min-width: 0 !important; max-width: 100% !important; }
@media (max-width: 760px) { gradio-app .main { padding-left: 14px !important; padding-right: 14px !important; } }
body { overflow-x: hidden; }
.glass { background: var(--surface) !important; border: 0 !important; border-radius: 20px !important; box-shadow: var(--shadow) !important; }

/* hero: full-bleed sky with drifting light, grain, and a live card that flips */
.fl-hero { position: relative; margin: 12px 0 0; border-radius: 28px; padding: 96px 24px 112px; text-align: center; color: #FBFBFD; overflow: hidden;
  background: linear-gradient(180deg,#0A1F45 0%,#123668 30%,#3E6AA8 58%,#A9C1E3 82%,#FBFBFD 100%); }
.fl-haze { position: absolute; inset: -20%; pointer-events: none;
  background: radial-gradient(40% 30% at 20% 30%,rgba(169,193,227,.35),transparent 70%),radial-gradient(35% 25% at 80% 20%,rgba(255,255,255,.18),transparent 70%),radial-gradient(50% 30% at 60% 70%,rgba(120,160,215,.30),transparent 70%);
  animation: fl-drift 28s ease-in-out infinite alternate; }
@keyframes fl-drift { to { transform: translate3d(4%,3%,0) scale(1.08); } }
.fl-grain { position: absolute; inset: 0; pointer-events: none; opacity: .07; mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
.fl-hero > *:not(.fl-haze):not(.fl-grain) { position: relative; z-index: 1; }
.fl-brand { font: 600 15px var(--sans); color: rgba(251,251,253,.85) !important; margin-bottom: 48px; }
.fl-hero h1 { font-family: var(--serif); font-weight: 400; font-size: clamp(44px,6.5vw,84px); line-height: 1.04; letter-spacing: -0.02em;
  margin: 0 auto; max-width: 900px; color: #FBFBFD; text-wrap: balance; }
.fl-hero p.fl-lede { margin: 24px auto 0; font-size: 19px; line-height: 1.55; max-width: 600px; color: rgba(251,251,253,.86); }
.fl-rise { opacity: 0; transform: translateY(18px); filter: blur(6px); animation: fl-rise 1s cubic-bezier(.2,.8,.2,1) forwards; }
.fl-d1 { animation-delay: .12s; } .fl-d2 { animation-delay: .24s; } .fl-d3 { animation-delay: .4s; }
@keyframes fl-rise { to { opacity: 1; transform: none; filter: none; } }
.fl-live { max-width: 560px; margin: 52px auto 0; padding: 22px 24px; border-radius: 24px; background: #FBFBFD; color: var(--ink); text-align: left;
  box-shadow: 0 1px 2px rgba(11,26,51,.06), 0 40px 80px -30px rgba(5,15,40,.55); }
.fl-who { display: flex; justify-content: space-between; gap: 12px; font-size: 13px; color: var(--muted); }
.fl-addr { margin: 14px 0 18px; font-family: var(--serif); font-size: 26px; line-height: 1.15; }
.fl-addr span { display: inline-block; transition: opacity .35s, transform .35s, filter .35s; }
.fl-addr span.out { opacity: 0; transform: translateY(-8px); filter: blur(4px); }
.fl-r { display: flex; justify-content: space-between; align-items: center; padding-top: 10px; margin-top: 10px; border-top: 1px solid var(--line); font-size: 15px; }
.fl-v { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; }
.fl-v i { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.fl-v.ok { color: var(--ok); } .fl-v.no { color: var(--no); }
.fl-r.flash { animation: fl-flash .9s ease; } @keyframes fl-flash { 30% { background: rgba(180,35,60,.07); } }
.fl-live small { display: block; margin-top: 14px; font-size: 12px; color: var(--muted); }
.fl-stats { display: grid; grid-template-columns: repeat(3,1fr); gap: 32px; text-align: center; margin: 8px 0 0; }
.fl-stats strong { display: block; font-family: var(--serif); font-weight: 400; font-size: clamp(40px,5vw,60px); line-height: 1; letter-spacing: -0.02em; color: var(--ink); }
.fl-stats span { display: block; margin: 10px auto 0; font-size: 14px; color: var(--muted); max-width: 240px; }

.fl-banner { margin: 40px 0 4px; padding: 20px 2px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.fl-banner .fl-bk { display: none; }
.fl-banner p { margin: 0; font-size: 18px; line-height: 1.55; color: var(--ink); max-width: 820px; }
.fl-banner b { font-weight: 600; }
.fl-banner .ok { color: var(--ok); font-weight: 600; } .fl-banner .no { color: var(--no); font-weight: 600; }
.fl-step { display: flex; align-items: baseline; gap: 12px; font-size: 15px; font-weight: 500; color: var(--ink); margin: 32px 0 14px; }
.fl-step span { font-family: var(--serif); font-size: 20px; color: var(--muted); }

#fl-nb .wrap { display: grid !important; grid-template-columns: repeat(3,1fr); gap: 4px; padding: 4px; border-radius: 999px; background: #EEF1F7; border: 0; }
#fl-nb label { justify-content: center; border: 0 !important; border-radius: 999px !important; padding: 10px 8px !important; min-height: 44px;
  font-size: 15px; font-weight: 500; color: var(--muted) !important; background: transparent !important; cursor: pointer; box-shadow: none !important;
  transition: background-color .2s ease, color .2s ease; }
#fl-nb label:hover { color: var(--ink) !important; }
#fl-nb label:active { transform: scale(0.97); }
#fl-nb label.selected { color: #FBFBFD !important; background: var(--navy) !important; }
#fl-nb input { position: absolute; opacity: 0; }
#fl-nb label:has(input:focus-visible) { outline: 2px solid var(--focus); outline-offset: 3px; }

.fl-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.fl-card { padding: 24px !important; }
.fl-ok { --state: var(--ok); } .fl-no { --state: var(--no); }
.fl-ct { font-size: 16px; font-weight: 600; color: var(--ink); }
.fl-cs { font-size: 13px; color: var(--muted); margin-top: 2px; }
.fl-verdict { display: flex; align-items: center; gap: 10px; margin: 22px 0 6px; font-family: var(--serif); font-size: 34px; line-height: 1.1; color: var(--state); }
.fl-verdict i, .fl-chip i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--state); flex: none; }
.fl-mk { font-size: 14px; color: var(--muted); font-variant-numeric: tabular-nums; }
.fl-mk b { color: var(--ink); font-weight: 600; }
.fl-meter { position: relative; height: 6px; background: #E8ECF3; border-radius: 9px; margin: 14px 0 6px; }
.fl-fill { position: absolute; inset: 0 auto 0 0; background: var(--state); border-radius: 9px; }
.fl-tick { position: absolute; top: -5px; height: 16px; width: 2px; border-radius: 2px; background: var(--ink); }
.fl-scale { position: relative; height: 18px; font-size: 12px; color: var(--muted); }
.fl-tlab { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.fl-fix { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }
.fl-fixk { display: block; font-size: 13px; color: var(--muted); }
.fl-fixv { display: block; font-size: 15px; line-height: 1.45; color: var(--ink); margin-top: 2px; }
.fl-fixv b { font-weight: 600; }

.fl-strip { padding: 0 !important; margin-top: 8px; background: transparent !important; box-shadow: none !important; }
.fl-scroll { overflow-x: auto; }
.fl-strip table { width: 100%; border-collapse: collapse; border: 0 !important; font-variant-numeric: tabular-nums; background: transparent !important; }
.fl-strip thead, .fl-strip tbody, .fl-strip tr { border: 0 !important; background: transparent !important; }
.fl-strip th, .fl-strip td { text-align: left; padding: 14px 10px !important; border: 0 !important; border-bottom: 1px solid var(--line) !important;
  vertical-align: top; background: transparent !important; color: var(--ink); }
.fl-strip thead th { font-size: 14px; font-weight: 500; color: var(--muted); }
.fl-strip thead small { display: block; font-weight: 400; color: var(--muted); font-size: 12px; }
.fl-strip tbody th { font-size: 14px; font-weight: 600; min-width: 150px; }
.fl-chip { display: inline-flex; align-items: center; gap: 7px; font-size: 14px; white-space: nowrap; }
.fl-chip em { font-style: normal; color: var(--muted); margin-left: 4px; }
.fl-tag { display: block; margin-top: 3px; font-size: 12px; font-weight: 400; color: var(--muted); }
.fl-tag.flip { color: var(--no); }

.fl-panel { padding: 26px !important; margin-top: 36px; gap: 14px !important; }
.fl-panel .block, .fl-panel .form { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.fl-panel input[type=number] { font-size: 16px !important; font-weight: 500; }
.fl-ptitle { font-family: var(--serif); font-size: 24px; color: var(--ink); }
.fl-psub { font-size: 13px; color: var(--muted); margin: 6px 0 4px; line-height: 1.5; }
#fl-maya { min-height: 44px; font-weight: 500; border-radius: 999px !important; box-shadow: none !important; border: 1px solid var(--line) !important; background: transparent !important; color: var(--ink) !important; }
#fl-maya:hover { border-color: var(--ink) !important; }
#fl-maya:active { transform: scale(0.98); }
.fl-foot { font-size: 14px; color: var(--muted); line-height: 1.6; padding: 24px 2px 32px; border-top: 1px solid var(--line); margin-top: 48px; max-width: 820px; }
.fl-foot b { color: var(--ink); font-weight: 600; }
footer { opacity: .5; }
@media (max-width: 760px) { .fl-cards, .fl-stats { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } .fl-rise { opacity: 1; transform: none; filter: none; } }
"""

HEAD = (# Light-only design: force Gradio's light theme so its own widgets match.
        '<script>if(!/__theme=light/.test(location.search)){const u=new URL(location);u.searchParams.set("__theme","light");location.replace(u)}</script>'
        # The hero card cycles the same applicant through the three neighbourhoods (real model A answers for the example).
        '<script>(function(){var s=[["mostly white",0],["mixed",0],["mostly minority",1]],i=0;'
        'if(matchMedia("(prefers-reduced-motion: reduce)").matches)return;'
        'setInterval(function(){var a=document.getElementById("hAddr"),v=document.getElementById("hA");if(!a||!v)return;'
        'i=(i+1)%3;var t=s[i];a.classList.add("out");setTimeout(function(){a.textContent=t[0];a.classList.remove("out");'
        'v.className="fl-v "+(t[1]?"no":"ok");v.innerHTML="<i></i>"+(t[1]?"Deny":"Approve");'
        'if(t[1]){var r=document.getElementById("hRowA");r.classList.remove("flash");void r.offsetWidth;r.classList.add("flash");}},350);},2600);})();</script>')

with gr.Blocks(title="FairView") as demo:
    gr.HTML("""<div class="fl-hero"><div class="fl-haze"></div><div class="fl-grain"></div>
      <div class="fl-brand fl-rise">FairView</div>
      <h1 class="fl-rise fl-d1">Same person.<br>Different address.</h1>
      <p class="fl-lede fl-rise fl-d2">Banks use AI to approve mortgages. These two AIs are never told anyone's race,
         but one of them is told the neighborhood. Watch what that does.</p>
      <div class="fl-live fl-rise fl-d3">
        <div class="fl-who"><span>$60K a year · $100K loan · 20% down</span><span>Same applicant</span></div>
        <div class="fl-addr">Home in a <span id="hAddr">mostly white</span> neighborhood</div>
        <div class="fl-r" id="hRowA"><span>Typical bank AI</span><span class="fl-v ok" id="hA"><i></i>Approve</span></div>
        <div class="fl-r"><span>Our fixed AI</span><span class="fl-v ok"><i></i>Approve</span></div>
        <small>Real model answers for one constructed example applicant.</small>
      </div></div>
      <div class="fl-stats">
        <div><strong>3.0×</strong><span>how often a race-blind AI wrongly flagged bank-approved Black applicants vs. white</span></div>
        <div><strong>1.2×</strong><span>after our fix, keeping 97% of the model's accuracy score</span></div>
        <div><strong>264,340</strong><span>real Georgia mortgage applications, 2022–2023</span></div>
      </div>""")
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
      What matters is the result across all 117,547 real 2023 applications: the typical AI wrongly flagged bank-approved Black applicants
      <b>3.0×</b> as often as white ones; our fixed AI brought that to <b>1.2×</b> while keeping 97% of the PR-AUC score.
      The trade-off: white applicants the bank approved are flagged a little more often (4.0% → 5.6%).<br>
      Data: public HMDA records from the CFPB / FFIEC (ffiec.cfpb.gov). Built with pandas, scikit-learn, XGBoost, fairlearn and Gradio; Claude Code helped write code.<br>
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
    demo.launch(css=CSS, head=HEAD, favicon_path=str(Path(__file__).resolve().parents[1] / "docs" / "favicon.svg"), theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate", radius_size="lg"))
