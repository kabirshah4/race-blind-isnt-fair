"""Train baselines and models to predict mortgage DENIAL, then compare them.

Run:  .venv/bin/python src/train.py

Plain-English summary
---------------------
* We train on 2022 applications and test on 2023 applications, so the model
  never sees the "future" (the hackathon brief requires this for dated data).
* We predict DENIAL (the rare outcome, about 10%). Accuracy is misleading here:
  a "model" that approves everyone is 90% accurate but catches zero denials.
  So we report:
    - precision: of the people we flag as "likely denied", how many really were
    - recall:    of the people who really were denied, how many we flagged
    - PR-AUC:    one number for precision and recall across every threshold
                 (higher is better; random guessing scores about 0.10 here)
* Baselines (simple rules the AI has to beat):
    1. "Approve everyone"
    2. "Deny if debt-to-income is over 43%" (the classic rule lenders use)
* Models: logistic regression (simple, explainable) and XGBoost (stronger).
* Race, sex and age are NEVER given to the models.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "hmda_ga.parquet"
MODELS = ROOT / "models"


def make_preprocessor():
    """Turn the raw table into numbers a model can use.
    Numbers: fill blanks with the median, then scale to similar ranges.
    Categories (like loan type): one column per option (one-hot encoding)."""
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])


def make_models():
    return {
        "Logistic regression": Pipeline([
            ("prep", make_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "XGBoost": Pipeline([
            ("prep", make_preprocessor()),
            ("model", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="aucpr", n_jobs=-1, random_state=42)),
        ]),
    }


def evaluate(name, y_true_denied, denial_score, flagged):
    """y_true_denied: 1 if really denied. denial_score: higher = more likely denied.
    flagged: 1 if this method predicts 'denied'."""
    return {
        "method": name,
        "flagged %": round(100 * flagged.mean(), 1),
        "precision": round(precision_score(y_true_denied, flagged, zero_division=0), 3),
        "recall": round(recall_score(y_true_denied, flagged, zero_division=0), 3),
        "PR-AUC": round(average_precision_score(y_true_denied, denial_score), 3),
        "ROC-AUC": round(roc_auc_score(y_true_denied, denial_score), 3)
        if len(np.unique(denial_score)) > 1 else 0.5,
    }


def main():
    df = pd.read_parquet(DATA)
    train, test = df[df.year == 2022], df[df.year == 2023]
    X_train, X_test = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train_denied = 1 - train.approved
    y_test_denied = (1 - test.approved).to_numpy()
    print(f"train 2022: {len(train):,} rows ({y_train_denied.mean():.1%} denied) | "
          f"test 2023: {len(test):,} rows ({y_test_denied.mean():.1%} denied)\n")

    results = []
    # Baseline 1: approve everyone -> flags nobody.
    zeros = np.zeros(len(test))
    results.append(evaluate("Baseline: approve everyone", y_test_denied, zeros, zeros))
    # Baseline 2: the classic lending rule.
    dti_rule = (test.dti > 43).astype(int).to_numpy()
    results.append(evaluate("Baseline: deny if DTI > 43%", y_test_denied, test.dti.to_numpy(), dti_rule))

    # To compare fairly with the rule, each model flags the SAME share of 2023
    # applicants that we expect to be denied (the 2022 denial rate). We pick that
    # cut-off using 2022 data only, so nothing from the test year leaks in.
    flag_share = y_train_denied.mean()
    MODELS.mkdir(exist_ok=True)
    for name, pipe in make_models().items():
        pipe.fit(X_train, y_train_denied)
        cutoff = np.quantile(pipe.predict_proba(X_train)[:, 1], 1 - flag_share)
        score = pipe.predict_proba(X_test)[:, 1]
        results.append(evaluate(name, y_test_denied, score, (score >= cutoff).astype(int)))
        joblib.dump({"pipeline": pipe, "cutoff": cutoff}, MODELS / f"{name.split()[0].lower()}.joblib")

    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv(ROOT / "results_models.csv", index=False)


if __name__ == "__main__":
    main()
