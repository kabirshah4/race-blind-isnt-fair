"""Lender audit tool: is a lender's racial gap bigger than its applicants' finances explain?

Run:
  .venv/bin/python src/audit_lender.py                     # all Georgia 2023 lenders, anonymised summary
  .venv/bin/python src/audit_lender.py --state nc          # same for North Carolina
  .venv/bin/python src/audit_lender.py --lei <LEI>         # one-page report for one lender
Writes results_lenders.md (summary) or reports/lender_<LEI>.md (one lender).

Plain-English summary
---------------------
Our user is a fair-lending analyst at a housing nonprofit or regulator. They have a lender's
public HMDA file and need to decide: does this lender deserve a closer look?

1. We train a "finances-only" model on the previous year: it sees ONLY the applicant's
   income, loan, home value, loan-to-value, debt-to-income, loan term/type and co-applicant.
   No race, and no neighbourhood data (so it can't pick up the neighbourhood proxy).
2. For every application in the audit year, it gives an EXPECTED chance of denial based on
   finances alone.
3. For each lender and each group we compare:
       observed denial rate  vs.  expected denial rate (from finances)
   "Excess" = observed / expected. 1.0 means the lender denied exactly as often as the
   applicants' finances predict.
4. Finance-adjusted gap = Black excess / white excess. Above 1 means Black applicants were
   denied more than their finances explain, relative to white applicants at the same lender.
   We show a 95% bootstrap range, and only audit lenders with at least 50 Black and 50 white
   applications and at least 5 denials in each group.

What this is NOT: proof of discrimination. HMDA has no credit scores, so a gap can come from
information we can't see. It is a screening tool: "this lender's gap is larger than finances
explain; look closer."
Published summaries are anonymised on purpose: naming lenders from data without credit scores
would be unfair to them. Use --lei to look at one lender yourself.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fix import APPLICANT_FEATURES, to_matrix, xgb
from prep import load_year

ROOT = Path(__file__).resolve().parents[1]
MIN_PER_GROUP = 50
MIN_DENIALS = 5  # each group also needs at least 5 denials, or the ratio is meaningless
rng = np.random.default_rng(11)


def expected_denials(state, year):
    """Train the finances-only model on year-1, return the audit year with an `expected` column."""
    train, audit = load_year(year - 1, state).reset_index(drop=True), load_year(year, state).reset_index(drop=True)
    Xtr, Xau = to_matrix(train, audit, APPLICANT_FEATURES)
    model = xgb().fit(Xtr, 1 - train.approved)
    audit["expected"] = model.predict_proba(Xau)[:, 1]
    audit["denied"] = 1 - audit.approved
    return audit


def gap(d):
    """Finance-adjusted gap for one lender's applications (Black excess / white excess)."""
    b, w = d[d.race_group == "Black"], d[d.race_group == "White"]
    with np.errstate(divide="ignore", invalid="ignore"):  # a resample with 0 white denials gives inf/nan; skipped below
        return (b.denied.mean() / b.expected.mean()) / (w.denied.mean() / w.expected.mean())


def auditable(d):
    b, w = d[d.race_group == "Black"], d[d.race_group == "White"]
    return (len(b) >= MIN_PER_GROUP and len(w) >= MIN_PER_GROUP
            and b.denied.sum() >= MIN_DENIALS and w.denied.sum() >= MIN_DENIALS)


def lender_row(d):
    b, w = d[d.race_group == "Black"], d[d.race_group == "White"]
    boots = []
    for _ in range(200):
        boots.append(gap(pd.concat([b.sample(len(b), replace=True, random_state=rng.integers(1e9)),
                                    w.sample(len(w), replace=True, random_state=rng.integers(1e9))])))
    boots = np.array(boots)
    lo, hi = np.percentile(boots[np.isfinite(boots)], [2.5, 97.5])
    return {"applications": len(d), "Black apps": len(b), "white apps": len(w),
            "Black denied %": round(100 * b.denied.mean(), 1), "white denied %": round(100 * w.denied.mean(), 1),
            "raw gap (x)": round(b.denied.mean() / max(w.denied.mean(), 1e-9), 2),
            "Black expected %": round(float(100 * b.expected.mean()), 1), "white expected %": round(float(100 * w.expected.mean()), 1),
            "finance-adjusted gap (x)": round(gap(d), 2), "95% range": f"{lo:.2f}–{hi:.2f}", "_lo": lo, "_hi": hi}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", default="ga")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--lei", help="audit one lender by its Legal Entity Identifier")
    args = ap.parse_args()
    audit = expected_denials(args.state, args.year)

    if args.lei:
        d = audit[audit.lender == args.lei]
        if not auditable(d):
            raise SystemExit(f"Too few applications or denials to audit ({len(d)} applications); need "
                             f"{MIN_PER_GROUP}+ Black and white applications and {MIN_DENIALS}+ denials in each group.")
        r = lender_row(d)
        verdict = ("worth a closer look" if r["_lo"] > 1 else "no gap beyond what finances explain")
        out = ROOT / "reports" / f"lender_{args.lei}.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(
            f"# Lender audit: {args.lei} ({args.state.upper()} {args.year})\n\n"
            f"| | Black | White |\n|---|---|---|\n| Applications | {r['Black apps']:,} | {r['white apps']:,} |\n"
            f"| Actually denied | {r['Black denied %']}% | {r['white denied %']}% |\n"
            f"| Expected from finances alone | {r['Black expected %']}% | {r['white expected %']}% |\n\n"
            f"**Raw gap:** {r['raw gap (x)']}× · **Finance-adjusted gap:** {r['finance-adjusted gap (x)']}× "
            f"(95% range {r['95% range']})\n\n**Screening result:** {verdict}.\n\n"
            "*Prototype, not legal or financial advice. HMDA has no credit scores, so a gap is a reason to look "
            "closer, not proof of discrimination.*\n")
        print(out.read_text())
        return

    rows = []
    for lei, d in audit.groupby("lender"):
        if auditable(d):
            rows.append({"lei": lei, **lender_row(d)})
    t = pd.DataFrame(rows).sort_values("applications", ascending=False).reset_index(drop=True)
    t.insert(0, "lender", [f"Lender {i + 1}" for i in range(len(t))])  # anonymised, ranked by size
    flagged = (t._lo > 1).sum()
    summary = (f"{len(t)} lenders had at least {MIN_PER_GROUP} Black and {MIN_PER_GROUP} white applications "
               f"and {MIN_DENIALS}+ denials in each group. "
               f"Median raw gap {t['raw gap (x)'].median():.2f}x; median finance-adjusted gap "
               f"{t['finance-adjusted gap (x)'].median():.2f}x. {flagged} lenders ({flagged / len(t):.0%}) have a "
               f"finance-adjusted gap whose whole 95% range is above 1 (worth a closer look).")
    print(summary)
    cols = ["lender", "applications", "Black denied %", "white denied %", "raw gap (x)", "finance-adjusted gap (x)", "95% range"]
    print(t[cols].head(15).to_string(index=False))
    (ROOT / f"results_lenders{'' if args.state == 'ga' else '_' + args.state}.md").write_text(
        f"# Lender audit, {args.state.upper()} {args.year} (anonymised)\n\n{summary}\n\n"
        "Finance-adjusted gap = (Black observed/expected denial) ÷ (white observed/expected denial), where "
        "'expected' comes from a finances-only model trained on the previous year (no race, no neighbourhood).\n\n"
        + t[cols].to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
