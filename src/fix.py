"""Compare fixes for the false-alarm gap, with confidence intervals.

Run:  .venv/bin/python src/fix.py      (after prep.py and train.py)

Plain-English summary
---------------------
We compare three lending models on the 2023 test data:
  A. Full model: uses neighbourhood features (our original).
  B. No neighbourhood: same model, neighbourhood features removed.
  C. Fairness-trained: model B, trained with Microsoft's fairlearn
     "Exponentiated Gradient" method. During TRAINING it looks at race and
     keeps re-weighting applicants until the false-alarm rates of the groups
     are close. When it makes DECISIONS it still never sees race, which matters
     because US lending law (ECOA) bans using race in credit decisions.

For each model we report:
  * PR-AUC: how good it is at spotting denials (higher = better)
  * false-alarm ratio, Black vs White: how much more often qualified Black
    applicants get wrongly flagged (1.00 = equal)
  * a 95% confidence interval for that ratio. We re-draw the test set 200 times
    at random (a "bootstrap") and see how much the ratio moves. A narrow
    interval that stays above 1 means the gap is real, not luck.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, FalsePositiveRateParity
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TRACT_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "hmda_ga.parquet"
APPLICANT_FEATURES = [f for f in NUMERIC_FEATURES if f not in TRACT_FEATURES]
rng = np.random.default_rng(42)


def xgb():
    return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1, random_state=42)


def to_matrix(train, test, numeric_features):
    """Turn tables into number matrices (blanks filled, categories one-hot)."""
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric_features),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                               CATEGORICAL_FEATURES)])
    cols = numeric_features + CATEGORICAL_FEATURES
    Xtr = prep.fit_transform(train[cols])
    to_matrix.last_prep = prep  # remembered so main() can save it with the model for the demo app
    return Xtr, prep.transform(test[cols])


def false_alarm_ratio(test, flagged, group="Black", idx=None):
    """(false-alarm rate for `group`) / (false-alarm rate for White), among bank-approved people."""
    t = test if idx is None else test.iloc[idx]
    f = flagged if idx is None else flagged[idx]
    ok = (t.approved == 1).to_numpy()
    g = (t.race_group == group).to_numpy() & ok
    w = (t.race_group == "White").to_numpy() & ok
    return f[g].mean() / f[w].mean()


def bootstrap_ci(test, flagged, group="Black", n=200):
    ratios = [false_alarm_ratio(test, flagged, group, rng.integers(0, len(test), len(test)))
              for _ in range(n)]
    return np.percentile(ratios, [2.5, 97.5])


def main():
    df = pd.read_parquet(DATA)
    train, test = df[df.year == 2022], df[df.year == 2023].reset_index(drop=True)
    y_train, y_test = (1 - train.approved).to_numpy(), (1 - test.approved).to_numpy()
    flag_share = y_train.mean()  # flag the same share as 2022's denial rate
    results = {}

    # A and B: ordinary XGBoost, with and without neighbourhood features.
    for name, feats in [("A. Full model", NUMERIC_FEATURES), ("B. No neighbourhood", APPLICANT_FEATURES)]:
        Xtr, Xte = to_matrix(train, test, feats)
        m = xgb().fit(Xtr, y_train)
        cutoff = np.quantile(m.predict_proba(Xtr)[:, 1], 1 - flag_share)
        score = m.predict_proba(Xte)[:, 1]
        results[name] = (score, (score >= cutoff).astype(int))
        if name.startswith("A"):  # save model A for the demo app
            joblib.dump({"prep": to_matrix.last_prep, "models": [m], "weights": [1.0],
                         "cutoff": cutoff, "features": feats}, ROOT / "models" / "model_A_full.joblib")

    # C: fairness-trained. Race is passed as `sensitive_features` for TRAINING only.
    # ExponentiatedGradient is slow, so it trains on a random 60k sample of 2022.
    Xtr, Xte = to_matrix(train, test, APPLICANT_FEATURES)
    sample = rng.choice(len(train), 60_000, replace=False)
    # Why ratio_bound and not the default difference_bound: false-alarm rates are
    # small (2-5%), so a "1 percentage point" gap already passes a difference
    # check while still meaning ~1.45x more false alarms. A ratio check catches it:
    # each group's false-alarm rate must be within ~10% of the others in RELATIVE terms.
    fair = ExponentiatedGradient(xgb(), constraints=FalsePositiveRateParity(ratio_bound=0.9),
                                 max_iter=40)
    fair.fit(Xtr[sample], y_train[sample], sensitive_features=train.race_group.to_numpy()[sample])
    # The fair model is a weighted mix of several XGBoost models. Its score for an
    # applicant = the weighted average of each sub-model's "denied" probability.
    # (No race is needed to compute it.)
    def fair_score(X):
        return sum(w * p.predict_proba(X)[:, 1] for w, p in zip(fair.weights_, fair.predictors_) if w > 0)
    score = fair_score(Xte)
    cutoff = np.quantile(fair_score(Xtr), 1 - flag_share)
    results["C. Fairness-trained"] = (score, (score >= cutoff).astype(int))
    keep = [(w, p) for w, p in zip(fair.weights_, fair.predictors_) if w > 0]
    joblib.dump({"prep": to_matrix.last_prep, "models": [p for _, p in keep],
                 "weights": [w for w, _ in keep], "cutoff": cutoff, "features": APPLICANT_FEATURES},
                ROOT / "models" / "model_C_fair.joblib")

    rows = []
    for name, (score, flagged) in results.items():
        lo, hi = bootstrap_ci(test, flagged)
        rows.append({
            "model": name,
            "PR-AUC": round(average_precision_score(y_test, score), 3),
            "Black false alarms vs White": round(false_alarm_ratio(test, flagged), 2),
            "95% range": f"{lo:.2f}–{hi:.2f}",
            "Hispanic vs White": round(false_alarm_ratio(test, flagged, "Hispanic"), 2),
            "flagged %": round(100 * flagged.mean(), 1),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(ROOT / "results_fix.csv", index=False)  # read by charts.py
    with open(ROOT / "results_fix.md", "w") as f:
        f.write(table.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
