"""Robustness checks the blind judges asked for. Does the main finding survive?

Run:  .venv/bin/python src/robustness.py      (after fix.py; needs data/external/urban_county_overall_2024.csv)
Writes results_robustness.md.

Plain-English summary
---------------------
Public mortgage data has no credit scores, so a skeptic can say "the 3x gap is just credit
scores you can't see." We test that four ways, plus three other weak spots judges named:

  1. Credit stand-in. The Urban Institute's "Debt in America" data (Aug 2023 credit-bureau
     records) gives, for every county, the share of people with debt in collections and the
     credit-card delinquency rate. We add both to model A, to the "no neighbourhood" model B,
     and to the lender audit's finances-only model, then re-measure.
  2. Banks' own denial reasons. We drop every application the bank denied for credit history
     (HMDA's denial-reason field), retrain, and re-measure. If credit history drove the gap,
     it should shrink a lot.
  3. Fair comparison of the fix. Model C flags slightly fewer people than A (9.5% vs 10.3%),
     which by itself lowers false alarms. Here both flag exactly the same share.
  4. Stronger baselines: a tuned logistic regression and scikit-learn's own boosted trees.
  5. Wider, more honest uncertainty: confidence ranges that resample whole census tracts
     (applications in the same neighbourhood aren't independent), not single rows.
The full-data retrain of model C is in src/robustness_c.py (it is slow).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from audit_lender import MIN_DENIALS, MIN_PER_GROUP, gap
from fix import APPLICANT_FEATURES, false_alarm_ratio, to_matrix, xgb
from prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_year

ROOT = Path(__file__).resolve().parents[1]
CREDIT = ["county_in_collections", "county_card_delinquency"]
rng = np.random.default_rng(7)


def add_extras(df, year):
    """County credit stand-ins (Urban Institute) + bank denial reasons + census tract, joined by row."""
    raw = pd.read_csv(ROOT / "data" / "raw" / f"ga_{year}.csv", dtype=str, low_memory=False,
                      usecols=["census_tract"] + [f"denial_reason-{i}" for i in range(1, 5)])
    raw = raw.loc[df.index]
    df = df.copy()
    df["tract"] = raw.census_tract.to_numpy()
    df["credit_history_denial"] = raw[[f"denial_reason-{i}" for i in range(1, 5)]].eq("3").any(axis=1).to_numpy()
    u = pd.read_csv(ROOT / "data" / "external" / "urban_county_overall_2024.csv", dtype=str)
    u = u.assign(county_in_collections=pd.to_numeric(u.totcoll, errors="coerce"),
                 county_card_delinquency=pd.to_numeric(u.carddelrate, errors="coerce"))
    u = u.set_index("id")[CREDIT]
    df = df.join(u, on="county")
    return df


def fit_flag(train, test, feats):
    """Same recipe as fix.py: XGBoost, cut-off flags the 2022 denial share on 2022 data."""
    Xtr, Xte = to_matrix(train, test, feats)
    y = 1 - train.approved.to_numpy()
    m = xgb().fit(Xtr, y)
    cut = np.quantile(m.predict_proba(Xtr)[:, 1], 1 - y.mean())
    s = m.predict_proba(Xte)[:, 1]
    return s, (s >= cut).astype(int)


def tract_ci(test, flagged, n=300):
    """95% range resampling whole census tracts (clusters), not single applications."""
    t = test.assign(f=flagged)
    t = t[t.approved == 1]
    by = {k: g for k, g in t.groupby("tract")}
    keys = np.array(list(by))
    out = []
    for _ in range(n):
        s = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        b, w = s[s.race_group == "Black"], s[s.race_group == "White"]
        out.append(b.f.mean() / w.f.mean())
    return np.percentile(out, [2.5, 97.5])


def lender_summary(audit_df, label):
    rows = []
    for _, d in audit_df.groupby("lender"):
        b, w = d[d.race_group == "Black"], d[d.race_group == "White"]
        if not (len(b) >= MIN_PER_GROUP and len(w) >= MIN_PER_GROUP and b.denied.sum() >= MIN_DENIALS and w.denied.sum() >= MIN_DENIALS):
            continue
        boots = [gap(pd.concat([b.sample(len(b), replace=True, random_state=rng.integers(1e9)),
                                w.sample(len(w), replace=True, random_state=rng.integers(1e9))])) for _ in range(1000)]
        boots = np.array(boots)
        rows.append((gap(d), np.percentile(boots[np.isfinite(boots)], 2.5)))
    g = np.array(rows)
    return {"lender audit": label, "lenders": len(g), "median finance-adjusted gap": f"{np.median(g[:, 0]):.2f}×",
            "lenders whose 95% range is above 1": f"{(g[:, 1] > 1).sum()} ({(g[:, 1] > 1).mean():.0%})"}


def main():
    train = add_extras(load_year(2022), 2022).reset_index(drop=True)
    test = add_extras(load_year(2023), 2023).reset_index(drop=True)
    y_test = 1 - test.approved.to_numpy()
    print("county credit data matched for", f"{test.county_in_collections.notna().mean():.1%}", "of 2023 applications")
    out = ["# Robustness checks (Georgia, trained 2022, tested 2023)", ""]

    # ---- 1 & 2: does the Black/white false-alarm gap survive credit stand-ins? ----
    rows = []
    def row(name, s, f, t=test, y=y_test):
        lo, hi = tract_ci(t, f)
        rows.append({"model": name, "PR-AUC": round(average_precision_score(y, s), 3),
                     "Black vs white false alarms": f"{false_alarm_ratio(t, f):.2f}×", "95% range (by tract)": f"{lo:.2f}–{hi:.2f}"})
        print(rows[-1])
    sA, fA = fit_flag(train, test, NUMERIC_FEATURES); row("A. Full model (as published)", sA, fA)
    s, f = fit_flag(train, test, NUMERIC_FEATURES + CREDIT); row("A + county credit (collections, card delinquency)", s, f)
    sB, fB = fit_flag(train, test, APPLICANT_FEATURES); row("B. No neighbourhood (as published)", sB, fB)
    s, f = fit_flag(train, test, APPLICANT_FEATURES + CREDIT); row("B + county credit", s, f)
    tr2, te2 = train[~train.credit_history_denial].reset_index(drop=True), test[~test.credit_history_denial].reset_index(drop=True)
    s, f = fit_flag(tr2, te2, NUMERIC_FEATURES); row("A, credit-history denials removed (train and test)", s, f, te2, 1 - te2.approved.to_numpy())
    C = joblib.load(ROOT / "models" / "model_C_fair.joblib")
    Xc = C["prep"].transform(test[C["features"] + CATEGORICAL_FEATURES])
    sC = sum(w * m.predict_proba(Xc)[:, 1] for w, m in zip(C["weights"], C["models"]))
    fC = (sC >= C["cutoff"]).astype(int); row("C. Fairness-trained (as published)", sC, fC)
    out += ["## 1. Credit stand-ins and the model's race gap", "",
            pd.DataFrame(rows).to_markdown(index=False), "",
            f"County credit data from the Urban Institute's Debt in America (Aug 2023 credit-bureau records) matched "
            f"{test.county_in_collections.notna().mean():.0%} of applications. Ranges resample whole census tracts "
            f"({test.tract.nunique():,} tracts), so they're wider than the row-level ranges in results_fix.md.", ""]

    # ---- 3: same flag rate for A and C ----
    share = fA.mean()
    fC_matched = (sC >= np.quantile(sC, 1 - share)).astype(int)
    fA_matched = (sA >= np.quantile(sA, 1 - fC.mean())).astype(int)
    out += ["## 2. Comparing the fix at the same flag rate", "",
            "| comparison | share flagged | A: Black vs white | C: Black vs white |", "|---|---|---|---|",
            f"| as published | A {fA.mean():.1%}, C {fC.mean():.1%} | {false_alarm_ratio(test, fA):.2f}× | {false_alarm_ratio(test, fC):.2f}× |",
            f"| both flag {share:.1%} | {share:.1%} | {false_alarm_ratio(test, fA):.2f}× | {false_alarm_ratio(test, fC_matched):.2f}× |",
            f"| both flag {fC.mean():.1%} | {fC.mean():.1%} | {false_alarm_ratio(test, fA_matched):.2f}× | {false_alarm_ratio(test, fC):.2f}× |",
            "", "Cut-offs here are set on 2023 data only to equalise flag rates for this comparison.", ""]

    # ---- 4: stronger baselines ----
    num = NUMERIC_FEATURES
    logs = ["income", "loan_amount", "property_value"]
    prep = ColumnTransformer([("log", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p), StandardScaler()), logs),
                              ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), [c for c in num if c not in logs]),
                              ("dti", OneHotEncoder(handle_unknown="ignore"), ["dti"]),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    y_tr = 1 - train.approved.to_numpy()
    base = []
    for name, model in [("Logistic regression, tuned (log money, DTI as categories, scaled)", make_pipeline(prep, LogisticRegression(C=1.0, max_iter=2000))),
                        ("scikit-learn boosted trees (HistGradientBoosting)", make_pipeline(ColumnTransformer([("num", "passthrough", num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)]), HistGradientBoostingClassifier(random_state=0)))]:
        model.fit(train[num + CATEGORICAL_FEATURES], y_tr)
        s = model.predict_proba(test[num + CATEGORICAL_FEATURES])[:, 1]
        f = (s >= np.quantile(model.predict_proba(train[num + CATEGORICAL_FEATURES])[:, 1], 1 - y_tr.mean())).astype(int)
        base.append({"model": name, "PR-AUC": round(average_precision_score(y_test, s), 3), "Black vs white false alarms": f"{false_alarm_ratio(test, f):.2f}×"})
        print(base[-1])
    base.append({"model": "Our XGBoost (model A)", "PR-AUC": round(average_precision_score(y_test, sA), 3), "Black vs white false alarms": f"{false_alarm_ratio(test, fA):.2f}×"})
    base.append({"model": "Classic rule: deny if debts > 43% of income", "PR-AUC": 0.342, "Black vs white false alarms": "n/a"})
    out += ["## 3. Stronger baselines (same 17 inputs)", "", pd.DataFrame(base).to_markdown(index=False), ""]

    # ---- lender audit with credit stand-ins ----
    lrows = []
    for label, feats, tr, te in [("finances only (as published)", APPLICANT_FEATURES, train, test),
                                 ("finances + county credit", APPLICANT_FEATURES + CREDIT, train, test),
                                 ("finances only, credit-history denials removed", APPLICANT_FEATURES, tr2, te2)]:
        Xtr, Xte = to_matrix(tr, te, feats)
        m = xgb().fit(Xtr, 1 - tr.approved)
        a = te.assign(expected=m.predict_proba(Xte)[:, 1], denied=1 - te.approved)
        lrows.append(lender_summary(a, label))
        print(lrows[-1])
    out += ["## 4. Lender audit with credit stand-ins", "", pd.DataFrame(lrows).to_markdown(index=False), "",
            "Simple 95% ranges (1,000 draws), no multiple-testing correction, so compare rows with each other; "
            "the published headline (30 of 60) uses the stricter correction in results_lenders_strict.md.", ""]

    (ROOT / "results_robustness.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
