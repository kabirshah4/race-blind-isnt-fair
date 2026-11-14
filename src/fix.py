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


def group_costs(test, flagged):
    """Per-group error rates, so the cost of any fix is visible (not just the ratio)."""
    t = test.assign(f=flagged, d=1 - test.approved)
    out = {}
    for g in ["White", "Black"]:
        s = t[t.race_group == g]
        out[f"{g} false alarm %"] = round(100 * s[s.d == 0].f.mean(), 1)           # qualified, wrongly flagged
        out[f"{g} recall %"] = round(100 * s[s.d == 1].f.mean(), 1)                # real denials caught
        out[f"{g} precision %"] = round(100 * s[s.f == 1].d.mean(), 1)             # flags that were right
    return out


def fit_fair(X, y, groups, constraint):
    """Exponentiated Gradient: race is passed as `sensitive_features` for TRAINING only."""
    fair = ExponentiatedGradient(xgb(), constraints=constraint, max_iter=40)
    fair.fit(X, y, sensitive_features=groups)
    keep = [(w, p) for w, p in zip(fair.weights_, fair.predictors_) if w > 0]
    # How we turn it into ONE decision: average the sub-models' "denied" probabilities
    # (weighted), then apply a cut-off. This is not fairlearn's randomized classifier,
    # so fairness at our cut-off is MEASURED on held-out data, not guaranteed.
    score = lambda Z: sum(w * p.predict_proba(Z)[:, 1] for w, p in keep)
    return keep, score


def main():
    df = pd.read_parquet(DATA)
    train, test = df[df.year == 2022].reset_index(drop=True), df[df.year == 2023].reset_index(drop=True)
    y_train, y_test = (1 - train.approved).to_numpy(), (1 - test.approved).to_numpy()
    flag_share = y_train.mean()  # flag the same share as 2022's denial rate
    results = {}

    # A, B, B2: ordinary XGBoost with different inputs.
    no_minority = [f for f in NUMERIC_FEATURES if f != "tract_minority_population_percent"]
    for name, feats in [("A. Full model", NUMERIC_FEATURES),
                        ("B2. Drop only % minority", no_minority),
                        ("B. No neighbourhood", APPLICANT_FEATURES)]:
        Xtr, Xte = to_matrix(train, test, feats)
        m = xgb().fit(Xtr, y_train)
        cutoff = np.quantile(m.predict_proba(Xtr)[:, 1], 1 - flag_share)
        score = m.predict_proba(Xte)[:, 1]
        results[name] = (score, (score >= cutoff).astype(int))
        if name.startswith("A"):  # save model A for the demo app
            joblib.dump({"prep": to_matrix.last_prep, "models": [m], "weights": [1.0],
                         "cutoff": cutoff, "features": feats}, ROOT / "models" / "model_A_full.joblib")

    # C: fairness-trained. The CHOICE of fairness rule is made on 2022 data only:
    # we hold out 20% of 2022 as a validation set and never look at 2023 to choose.
    perm = rng.permutation(len(train))
    tr_i, va_i = perm[: int(0.8 * len(train))], perm[int(0.8 * len(train)):]
    tr, va = train.iloc[tr_i].reset_index(drop=True), train.iloc[va_i].reset_index(drop=True)
    Xtr, Xva = to_matrix(tr, va, APPLICANT_FEATURES)
    sample = rng.choice(len(tr), 60_000, replace=False)  # ExponentiatedGradient is slow
    candidates = {
        # default check: absolute gap in false-alarm rates of at most 1 percentage point
        "difference ≤ 1 point": FalsePositiveRateParity(difference_bound=0.01),
        # ratio check: each group's false-alarm rate within ~10% of the others, relatively
        "ratio ≥ 0.9": FalsePositiveRateParity(ratio_bound=0.9),
    }
    val_rows, fitted = [], {}
    for label, cons in candidates.items():
        keep, score = fit_fair(Xtr[sample], 1 - tr.approved.to_numpy()[sample], tr.race_group.to_numpy()[sample], cons)
        cut = np.quantile(score(Xtr), 1 - flag_share)
        vf = (score(Xva) >= cut).astype(int)
        val_rows.append({"fairness rule": label, "validation PR-AUC": round(average_precision_score(1 - va.approved, score(Xva)), 3),
                         "validation Black/White false alarms": round(false_alarm_ratio(va, vf), 2)})
        fitted[label] = (keep, score)
    val = pd.DataFrame(val_rows)
    print("Choosing the fairness rule on 2022 VALIDATION data:\n", val.to_string(index=False), "\n")
    chosen = val.iloc[(val["validation Black/White false alarms"] - 1).abs().argmin()]["fairness rule"]
    keep, score = fitted[chosen]
    prep_c = to_matrix.last_prep
    Xte_c = prep_c.transform(test[APPLICANT_FEATURES + CATEGORICAL_FEATURES])
    cutoff = np.quantile(score(Xtr), 1 - flag_share)
    results[f"C. Fairness-trained ({chosen})"] = (score(Xte_c), (score(Xte_c) >= cutoff).astype(int))
    joblib.dump({"prep": prep_c, "models": [p for _, p in keep], "weights": [w for w, _ in keep],
                 "cutoff": cutoff, "features": APPLICANT_FEATURES}, ROOT / "models" / "model_C_fair.joblib")

    rows, cost_rows = [], []
    for name, (sc, flagged) in results.items():
        lo, hi = bootstrap_ci(test, flagged)
        rows.append({
            "model": name,
            "PR-AUC": round(average_precision_score(y_test, sc), 3),
            "Black false alarms vs White": round(false_alarm_ratio(test, flagged), 2),
            "95% range": f"{lo:.2f}–{hi:.2f}",
            "Hispanic vs White": round(false_alarm_ratio(test, flagged, "Hispanic"), 2),
            "flagged %": round(100 * flagged.mean(), 1),
        })
        cost_rows.append({"model": name, **group_costs(test, flagged)})
    table, costs = pd.DataFrame(rows), pd.DataFrame(cost_rows)
    print(table.to_string(index=False), "\n\n", costs.to_string(index=False))
    table.to_csv(ROOT / "results_fix.csv", index=False)  # read by charts.py
    with open(ROOT / "results_fix.md", "w") as f:
        f.write("## Models on 2023 test data\n\n" + table.to_markdown(index=False) + "\n\n")
        f.write("## What each fix costs, by group (2023 test data)\n\n" + costs.to_markdown(index=False) + "\n\n")
        f.write("## How the fairness rule was chosen (2022 validation data only)\n\n" + val.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
