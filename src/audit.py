"""Fairness audit: does a model that never sees race still treat groups differently?

Run:  .venv/bin/python src/audit.py      (after prep.py and train.py)

The four questions, in plain English
------------------------------------
Q1  GAP:      Using the 2023 test data, what share of each group does the model
              flag as "likely denied"? Compare with White applicants.
Q2  ERRORS:   Among people the bank actually APPROVED, how often does the model
              wrongly flag them (false alarm rate)? If that's higher for one
              group, the model is harsher on qualified people from that group.
Q3  PROXIES:  Train a separate "detective" model to guess whether an applicant
              is Black or White using ONLY the features our lending model sees.
              If the detective does well, those features carry race information,
              even though race was removed.
Q4  NO NEIGHBOURHOOD: Retrain the lending model WITHOUT the neighbourhood
              (census tract) features and see what happens to the gaps and to
              accuracy. This is our first attempt at a fix, and it measures the cost.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TRACT_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "hmda_ga.parquet"
GROUPS = ["White", "Black", "Hispanic", "Asian"]  # big enough to measure reliably
REFERENCE = "White"


def group_table(test, flagged):
    """Per group: real denial rate, model flag rate, and false-alarm rate."""
    t = test.assign(flagged=flagged, denied=1 - test.approved)
    rows = []
    for g in GROUPS:
        s = t[t.race_group == g]
        approved_by_bank = s[s.denied == 0]
        rows.append({
            "group": g,
            "applicants": len(s),
            "real denial %": round(100 * s.denied.mean(), 1),
            "model flags %": round(100 * s.flagged.mean(), 1),
            "false alarm %": round(100 * approved_by_bank.flagged.mean(), 1),
        })
    out = pd.DataFrame(rows).set_index("group")
    ref = out.loc[REFERENCE]
    out["flags vs White (x)"] = (out["model flags %"] / ref["model flags %"]).round(2)
    out["false alarms vs White (x)"] = (out["false alarm %"] / ref["false alarm %"]).round(2)
    return out


def fit_and_flag(train, test, numeric_features):
    """Train XGBoost on the given features and flag ~the 2022 denial share."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier

    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric_features),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    pipe = Pipeline([("prep", prep), ("model", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, random_state=42))])
    cols = numeric_features + CATEGORICAL_FEATURES
    y = 1 - train.approved
    pipe.fit(train[cols], y)
    cutoff = np.quantile(pipe.predict_proba(train[cols])[:, 1], 1 - y.mean())
    score = pipe.predict_proba(test[cols])[:, 1]
    return pipe, score, (score >= cutoff).astype(int)


def main():
    df = pd.read_parquet(DATA)
    train, test = df[df.year == 2022], df[df.year == 2023]
    y_test = 1 - test.approved
    report = []

    # ---- Q1 + Q2: the full model (includes neighbourhood features) ----------
    saved = joblib.load(ROOT / "models" / "xgboost.joblib")
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    score_full = saved["pipeline"].predict_proba(test[cols])[:, 1]
    flags_full = (score_full >= saved["cutoff"]).astype(int)
    full = group_table(test, flags_full)
    print("Q1+Q2  Full model (race never used):\n", full.to_string(), "\n")
    report.append(("Full model (race never used)", full))

    # ---- Q3: can the 'neutral' features predict race? ----------------------
    bw_train = train[train.race_group.isin(["Black", "White"])]
    bw_test = test[test.race_group.isin(["Black", "White"])]
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    proxy_rows = []
    for label, feats in [("All model features", NUMERIC_FEATURES),
                         ("Neighbourhood features only", TRACT_FEATURES),
                         ("Applicant/loan features only", [f for f in NUMERIC_FEATURES if f not in TRACT_FEATURES])]:
        # The "detective": same kind of model, but it tries to guess race instead of denial.
        pipe = Pipeline([("prep", ColumnTransformer([("num", SimpleImputer(strategy="median"), feats)])),
                         ("model", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 n_jobs=-1, random_state=42))])
        pipe.fit(bw_train, (bw_train.race_group == "Black").astype(int))
        auc = roc_auc_score((bw_test.race_group == "Black").astype(int), pipe.predict_proba(bw_test)[:, 1])
        proxy_rows.append({"features the detective sees": label, "ROC-AUC guessing Black vs White": round(auc, 3)})
    proxies = pd.DataFrame(proxy_rows)
    print("Q3  Proxy detective (0.5 = can't tell, 1.0 = perfect):\n", proxies.to_string(index=False), "\n")

    # Which single feature carries the most race information?
    single = []
    for f in NUMERIC_FEATURES:
        x = bw_test[f].fillna(bw_test[f].median())
        auc = roc_auc_score((bw_test.race_group == "Black").astype(int), x)
        single.append({"feature": f, "race signal (AUC, 0.5 = none)": round(max(auc, 1 - auc), 3)})
    single = pd.DataFrame(single).sort_values("race signal (AUC, 0.5 = none)", ascending=False)
    print("Q3b Race signal of each single feature:\n", single.head(8).to_string(index=False), "\n")

    # ---- Q4: retrain without neighbourhood features ------------------------
    applicant_only = [f for f in NUMERIC_FEATURES if f not in TRACT_FEATURES]
    _, score_nt, flags_nt = fit_and_flag(train, test, applicant_only)
    no_tract = group_table(test, flags_nt)
    print("Q4  Model WITHOUT neighbourhood features:\n", no_tract.to_string(), "\n")
    report.append(("Model without neighbourhood features", no_tract))

    cost = pd.DataFrame([
        {"model": "Full", "PR-AUC": round(average_precision_score(y_test, score_full), 3),
         "Black flags vs White (x)": full.loc["Black", "flags vs White (x)"],
         "Black false alarms vs White (x)": full.loc["Black", "false alarms vs White (x)"]},
        {"model": "No neighbourhood", "PR-AUC": round(average_precision_score(y_test, score_nt), 3),
         "Black flags vs White (x)": no_tract.loc["Black", "flags vs White (x)"],
         "Black false alarms vs White (x)": no_tract.loc["Black", "false alarms vs White (x)"]},
    ])
    print("Q4b The trade-off:\n", cost.to_string(index=False))

    # Save everything for the write-up.
    with open(ROOT / "results_audit.md", "w") as f:
        for title, t in report:
            f.write(f"## {title}\n\n{t.to_markdown()}\n\n")
        f.write(f"## Proxy detective\n\n{proxies.to_markdown(index=False)}\n\n")
        f.write(f"## Race signal of single features\n\n{single.to_markdown(index=False)}\n\n")
        f.write(f"## Trade-off\n\n{cost.to_markdown(index=False)}\n")


if __name__ == "__main__":
    main()
