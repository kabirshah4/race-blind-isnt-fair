"""Export the anonymised lender-audit results for the web page docs/lenders.html.

Run:  .venv/bin/python src/export_lenders.py      (after audit_lender.py and lender_chart.py)
Writes docs/lenders.json.

Plain-English summary
---------------------
The lender audit already exists as tables (results_lenders*.md). The web page needs the
same numbers as data, so this script reads those tables and saves them as JSON. Lenders
stay anonymised ("Lender 1" = the largest by number of applications); no names, no IDs.
Georgia uses the stricter test (Benjamini-Hochberg correction for testing 60 lenders at
once); North Carolina uses the simple 95% range.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(path):
    lines = [l for l in Path(path).read_text().splitlines() if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(head, [c.strip() for c in l.strip("|").split("|")])) for l in lines[2:]]


def rng(s):
    lo, hi = re.split(r"[–-]", s)
    return float(lo), float(hi)


def main():
    out = {}
    simple = {r["lender"]: r for r in table(ROOT / "results_lenders.md")}
    strict = {r["lender"]: r for r in table(ROOT / "results_lenders_strict.md")}
    ga = []
    for name, r in simple.items():
        s = strict[name]
        lo, hi = rng(s["95% range"])
        ga.append({"name": name, "apps": int(r["applications"]), "black_denied": float(r["Black denied %"]),
                   "white_denied": float(r["white denied %"]), "raw": float(r["raw gap (x)"]),
                   "adjusted": float(s["finance-adjusted gap (x)"]), "lo": lo, "hi": hi,
                   "flag": s["passes BH (5% FDR)"] == "True"})
    out["ga"] = {"label": "Georgia 2023", "test": "95% range from 1,000 draws, plus a correction for testing 60 lenders at once",
                 "lenders": ga}
    nc = []
    for r in table(ROOT / "results_lenders_nc.md"):
        lo, hi = rng(r["95% range"])
        nc.append({"name": r["lender"], "apps": int(r["applications"]), "black_denied": float(r["Black denied %"]),
                   "white_denied": float(r["white denied %"]), "raw": float(r["raw gap (x)"]),
                   "adjusted": float(r["finance-adjusted gap (x)"]), "lo": lo, "hi": hi, "flag": lo > 1})
    out["nc"] = {"label": "North Carolina 2023", "test": "95% range from 1,000 draws (no extra correction)", "lenders": nc}
    p = ROOT / "docs" / "lenders.json"
    p.write_text(json.dumps(out, separators=(",", ":")))
    for k, v in out.items():
        print(k, len(v["lenders"]), "lenders,", sum(l["flag"] for l in v["lenders"]), "flagged")


if __name__ == "__main__":
    main()
