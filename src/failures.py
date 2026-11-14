"""Where it fails: concrete failure cases from the 2023 Georgia test data.

Run:  .venv/bin/python src/failures.py      (after fix.py)
Writes results_failures.md.

Plain-English summary
---------------------
A model's average score hides who it gets wrong. This script looks at the actual mistakes:
  1. Misses: applicants the bank DENIED but our model said were fine. HMDA records the bank's
     reason for each denial, so we can check how many were denied for credit history,
     which our model can't see (public data has no credit scores).
  2. False alarms: applicants the bank APPROVED but model A flagged, with one real example
     whose flag disappears when only the neighbourhood is swapped.
  3. Where the fix backfires: who model C flags that model A didn't, and how it does for
     other groups (not just Black vs white).
Example applicants are rounded (income and loan to the nearest $5K) so they read as cases,
not records.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prep import CATEGORICAL_FEATURES, load_year

ROOT = Path(__file__).resolve().parents[1]
HIDDEN = ["credit history", "application incomplete", "information couldn't be verified", "other"]
REASON = {"1": "debt-to-income", "2": "employment history", "3": "credit history", "4": "collateral (home value)",
          "5": "not enough cash", "6": "information couldn't be verified", "7": "application incomplete",
          "8": "mortgage insurance denied", "9": "other"}


def score(bundle, df):
    X = bundle["prep"].transform(df[bundle["features"] + CATEGORICAL_FEATURES])
    return sum(w * m.predict_proba(X)[:, 1] for w, m in zip(bundle["weights"], bundle["models"]))


def r5(v):
    return int(round(v / 5) * 5)


def main():
    test = load_year(2023)
    raw = pd.read_csv(ROOT / "data" / "raw" / "ga_2023.csv", low_memory=False, dtype=str, usecols=["denial_reason-1"])
    test["reason"] = raw.loc[test.index, "denial_reason-1"].map(REASON).fillna("not given")
    A = joblib.load(ROOT / "models" / "model_A_full.joblib")
    C = joblib.load(ROOT / "models" / "model_C_fair.joblib")
    nbs = json.loads((ROOT / "models" / "neighbourhoods.json").read_text())
    test["a"], test["c"] = score(A, test), score(C, test)
    test["fa"], test["fc"] = test.a >= A["cutoff"], test.c >= C["cutoff"]
    denied, approved = test[test.approved == 0], test[test.approved == 1]
    out = ["# Where it fails (Georgia 2023 test data)", ""]

    # 1. Misses, by the bank's own denial reason
    miss, caught = denied[~denied.fa], denied[denied.fa]
    m_share = miss.reason.value_counts(normalize=True)
    c_share = caught.reason.value_counts(normalize=True)
    out += ["## 1. Misses: the bank denied them, model A didn't flag them", "",
            f"Model A misses **{len(miss):,} of {len(denied):,}** denials ({len(miss) / len(denied):.0%}).",
            f"**{m_share.reindex(HIDDEN).fillna(0).sum():.0%}** of the misses were denied for reasons public data can't show "
            f"(credit history, incomplete application, unverifiable information, other), vs "
            f"{c_share.reindex(HIDDEN).fillna(0).sum():.0%} of the denials it caught.",
            f"**{m_share.get('credit history', 0):.0%}** of the misses were denied for **credit history**, which public data "
            f"doesn't include (vs {c_share.get('credit history', 0):.0%} of the denials it caught). The model mostly catches "
            "debt-to-income denials, which it *can* see.", "",
            "| bank's main denial reason | share of misses | share of caught denials |", "|---|---|---|"]
    for r in m_share.index[:6]:
        out.append(f"| {r} | {m_share[r]:.0%} | {c_share.get(r, 0):.0%} |")
    ex = miss[(miss.reason == "credit history") & (miss.dti <= 36) & (miss.ltv <= 85)].sort_values("income")
    if len(ex):
        e = ex.iloc[len(ex) // 2]
        out += ["", f"*Example:* income ${r5(e.income)}K, loan ${r5(e.loan_amount)}K, {100 - e.ltv:.0f}% down, debts {e.dti:.0f}% "
                f"of income. Model A: {e.a:.0%} chance of denial (flags above {A['cutoff']:.0%}). The bank denied it for credit "
                "history. On paper this applicant looks safe; the reason is invisible to us."]

    # 2. A false alarm that the address alone causes
    fa_black = approved[(approved.race_group == "Black") & approved.fa]
    white_profile = nbs["Mostly white"]
    swapped = score(A, fa_black.assign(**white_profile))
    flips = fa_black[swapped < A["cutoff"]]
    out += ["", "## 2. False alarms: the bank approved them, model A flagged them", "",
            f"Model A flags **{len(fa_black):,}** Black applicants the bank approved. Moving them, on paper, to a typical mostly "
            f"white neighbourhood (nothing else changed) clears **{len(flips):,} ({len(flips) / len(fa_black):.0%})** of those flags."]
    good = flips[(flips.dti <= 43) & (flips.ltv.between(75, 97)) & (flips.c < C["cutoff"])]
    if len(good):
        e = good.loc[(good.a - A["cutoff"]).abs().sort_values().index[len(good) // 2]]
        out += ["", f"*Example:* income ${r5(e.income)}K, loan ${r5(e.loan_amount)}K, {100 - e.ltv:.0f}% down, debts {e.dti:.0f}% of "
                f"income, in a neighbourhood that is {e.tract_minority_population_percent:.0f}% minority. The bank approved it. "
                f"Model A: {e.a:.0%} (flagged). Same file in a mostly white neighbourhood: "
                f"{float(score(A, pd.DataFrame([e]).assign(**white_profile))[0]):.0%} (not flagged). Model C: {e.c:.0%}."]

    # 3. Where the fix backfires
    w = approved[approved.race_group == "White"]
    new_w = w[w.fc & ~w.fa]
    rec_a, rec_c = denied.fa.mean(), denied.fc.mean()
    out += ["", "## 3. Where the fix backfires", "",
            f"- Model C newly flags **{len(new_w):,}** white applicants the bank approved and stops flagging "
            f"{(w.fa & ~w.fc).sum():,} others: a net **{w.fc.sum() - w.fa.sum():+,}** (white false alarms {w.fa.mean():.1%} → {w.fc.mean():.1%}).",
            f"- Model C catches **{rec_c:.0%}** of real denials vs **{rec_a:.0%}** for model A.", "",
            "False-alarm rate (bank-approved applicants flagged), by group:", "",
            "| group | applicants | model A | model C | vs white (A → C) |", "|---|---|---|---|---|"]
    wa, wc = w.fa.mean(), w.fc.mean()
    for g in ["White", "Black", "Hispanic", "Asian", "Not reported"]:
        s = approved[approved.race_group == g]
        if len(s) < 200:
            continue
        out.append(f"| {g} | {len(s):,} | {s.fa.mean():.1%} | {s.fc.mean():.1%} | {s.fa.mean() / wa:.2f}× → {s.fc.mean() / wc:.2f}× |")
    out += ["", "C was tuned only for Black vs white, so check the other rows before trusting it for everyone.", ""]

    path = ROOT / "results_failures.md"
    path.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
