"""Export the real trained models so the web demo can run them live in the browser.

Run:  .venv/bin/python src/export_models.py      (after fix.py)
Writes docs/models.json, which docs/index.html reads.

Plain-English summary
---------------------
An XGBoost model is just a few hundred small yes/no decision trees ("is debt-to-income
under 43%? -> is the loan-to-value over 95%? -> ..."). Each tree ends in a number; add
up the numbers from every tree, squash the total into 0-100%, and that's the model's
answer. So instead of running Python on a server, we write every tree out to a file and
the web page walks through the same trees in JavaScript. Same trees, same answers, for
any input you type.

At the end we test it: we score 2,000 random applicants with Python and with a copy of
the page's tree-walking logic, and check the answers match.
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prep import CATEGORICAL_FEATURES

ROOT = Path(__file__).resolve().parents[1]


def flatten(tree):
    """Turn one nested JSON tree into a flat list. Node i is either
    [feature, threshold, yes_index, no_index] (go to yes if value < threshold) or [leaf_value]."""
    nodes = {}

    def walk(n):
        if "leaf" in n:
            nodes[n["nodeid"]] = [n["leaf"]]
        else:
            nodes[n["nodeid"]] = [int(n["split"][1:]), n["split_condition"], n["yes"], n["no"]]
            for c in n["children"]:
                walk(c)

    walk(tree)
    return [nodes[i] for i in range(len(nodes))]


def export(bundle):
    num, cat = bundle["prep"].transformers_[0], bundle["prep"].transformers_[1]
    subs = []
    for model in bundle["models"]:
        booster = model.get_booster()
        base = float(json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
        trees = [flatten(json.loads(t)) for t in booster.get_dump(dump_format="json")]
        subs.append({"base": math.log(base / (1 - base)), "trees": trees})
    return {"numeric": list(num[2]), "medians": [float(v) for v in num[1].statistics_],
            "categorical": list(cat[2]), "categories": [list(map(str, c)) for c in cat[1].categories_],
            "weights": [float(w) for w in bundle["weights"]], "cutoff": float(bundle["cutoff"]), "models": subs}


# ---- the page's logic, copied into Python, used only to check the export ----
def js_style_risk(m, row):
    f32 = lambda v: float(np.float32(v))
    x = [f32(row[c]) for c in m["numeric"]]
    for col, cats in zip(m["categorical"], m["categories"]):
        x += [1.0 if str(row[col]) == c else 0.0 for c in cats]
    total = 0.0
    for w, sub in zip(m["weights"], m["models"]):
        margin = sub["base"]
        for t in sub["trees"]:
            i = 0
            while len(t[i]) == 4:
                i = t[i][2] if x[t[i][0]] < f32(t[i][1]) else t[i][3]
            margin += t[i][0]
        total += w / (1 + math.exp(-margin))
    return total


def main():
    nbs = json.loads((ROOT / "models" / "neighbourhoods.json").read_text())
    out = {"A": export(joblib.load(ROOT / "models" / "model_A_full.joblib")),
           "C": export(joblib.load(ROOT / "models" / "model_C_fair.joblib")),
           "neighbourhoods": nbs}

    # Check: random applicants, Python vs the page's logic.
    rng = np.random.default_rng(0)
    n = 2000
    loan = rng.uniform(30, 900, n)
    price = loan / rng.uniform(0.5, 1.05, n)
    df = pd.DataFrame({"income": rng.uniform(15, 400, n).round(), "loan_amount": loan.round(), "property_value": price.round()})
    df["ltv"] = df.loan_amount / df.property_value * 100
    df["dti"] = rng.integers(10, 66, n)
    df["loan_term_360"] = rng.integers(0, 2, n)
    df["has_coapplicant"] = rng.integers(0, 2, n)
    df["loan_type"] = rng.choice(["1", "2", "3", "4"], n)
    df["conforming_loan_limit"] = np.where(df.loan_amount <= 766, "C", "NC")
    df["submission_of_application"] = "1"
    df = df.assign(**list(nbs.values())[rng.integers(0, 3)])
    worst = 0.0
    for key in ["A", "C"]:
        b = joblib.load(ROOT / "models" / f"model_{'A_full' if key == 'A' else 'C_fair'}.joblib")
        X = b["prep"].transform(df[b["features"] + CATEGORICAL_FEATURES])
        py = sum(w * m.predict_proba(X)[:, 1] for w, m in zip(b["weights"], b["models"]))
        js = np.array([js_style_risk(out[key], r) for r in df.to_dict("records")])
        worst = max(worst, float(np.abs(py - js).max()))
        flips = int(((py >= b["cutoff"]) != (js >= b["cutoff"])).sum())
        print(f"model {key}: max difference {np.abs(py - js).max():.2e}, decisions that differ: {flips} of {n}")
    assert worst < 1e-4, "export doesn't match the Python models"

    path = ROOT / "docs" / "models.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"-> {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
