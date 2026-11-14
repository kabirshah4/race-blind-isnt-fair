"""Lender audit, stricter: a forest plot of every lender plus a multiple-testing correction.

Run:  .venv/bin/python src/lender_chart.py [--state nc]     (after download.py)
Writes figures/4_lenders[_nc].png and results_lenders_strict[_nc].md.

Why: audit_lender.py flags a lender when its whole 95% range sits above 1. Checking 60 lenders
that way, a few could cross the line by chance alone. Here we:
  1. use 1,000 bootstrap draws per lender (not 200),
  2. turn each lender's draws into a one-sided p-value: the share of draws with gap <= 1,
  3. apply the Benjamini-Hochberg correction across all lenders (false discovery rate 5%),
     so that among the lenders we flag, we expect at most ~5% to be false alarms.
Lenders stay anonymised (ranked by size), same as results_lenders.md.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_lender import auditable, expected_denials, gap

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(11)
DRAWS = 1000


def lender_stats(d):
    b, w = d[d.race_group == "Black"], d[d.race_group == "White"]
    bd, be, wd, we = (b.denied.to_numpy(), b.expected.to_numpy(), w.denied.to_numpy(), w.expected.to_numpy())
    bi = rng.integers(0, len(b), (DRAWS, len(b)))
    wi = rng.integers(0, len(w), (DRAWS, len(w)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = (bd[bi].mean(1) / be[bi].mean(1)) / (wd[wi].mean(1) / we[wi].mean(1))
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = ((boots <= 1).sum() + 1) / (len(boots) + 1)
    return {"applications": len(d), "gap": gap(d), "lo": lo, "hi": hi, "p": p}


def benjamini_hochberg(p, q=0.05):
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="ga")
    args = ap.parse_args()
    audit = expected_denials(args.state, 2023)
    rows = [{"lei": lei, **lender_stats(d)} for lei, d in audit.groupby("lender") if auditable(d)]
    t = pd.DataFrame(rows).sort_values("applications", ascending=False).reset_index(drop=True)
    t.insert(0, "lender", [f"Lender {i + 1}" for i in range(len(t))])
    t["q"] = benjamini_hochberg(t.p.to_numpy())
    t["range above 1"] = t.lo > 1
    t["passes BH (5% FDR)"] = t.q < 0.05
    n, n_ci, n_bh = len(t), int(t["range above 1"].sum()), int(t["passes BH (5% FDR)"].sum())
    summary = (f"{args.state.upper()} 2023: {n} auditable lenders. {n_ci} have a 95% range entirely above 1 "
               f"(1,000 draws); {n_bh} ({n_bh / n:.0%}) still stand out after the Benjamini-Hochberg correction "
               f"for testing {n} lenders at once (false discovery rate 5%).")
    print(summary)

    s = t.sort_values("gap").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(6, 0.17 * n)), dpi=160)
    navy, grey = "#0B2A5B", "#A7B1C2"
    for i, r in s.iterrows():
        c = navy if r["passes BH (5% FDR)"] else grey
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(r.gap, i, "o", color=c, ms=4)
    ax.axvline(1, color="#B4233C", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks([])
    ax.set_xlabel("Finance-adjusted gap, Black vs. white denials (log scale; 1 = no gap beyond finances)")
    ax.set_title(f"{n} {args.state.upper()} lenders, 2023 (anonymised). Navy = stands out after correction ({n_bh})",
                 fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    suffix = "" if args.state == "ga" else f"_{args.state}"
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / f"4_lenders{suffix}.png")

    show = t.assign(**{"finance-adjusted gap (x)": t.gap.round(2), "95% range": [f"{a:.2f}–{b:.2f}" for a, b in zip(t.lo, t.hi)],
                       "BH q": t.q.round(3)})
    cols = ["lender", "applications", "finance-adjusted gap (x)", "95% range", "BH q", "passes BH (5% FDR)"]
    (ROOT / f"results_lenders_strict{suffix}.md").write_text(
        f"# Lender audit with a multiple-testing correction, {args.state.upper()} 2023 (anonymised)\n\n{summary}\n\n"
        f"![Lender forest plot](figures/4_lenders{suffix}.png)\n\n" + show[cols].to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
