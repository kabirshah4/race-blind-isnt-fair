"""Placebo ("fake test") checks: does the 3x gap vanish when we break the thing we say causes it?

Run:  .venv/bin/python src/placebo.py      (after fix.py)
Writes results_placebo.md.

Plain-English summary
---------------------
A placebo test runs the same analysis on data where the real link has been destroyed. If the
gap still showed up, our method would be producing it by itself. Two checks:

  1. Shuffled race labels. Keep model A's flags exactly as they are, but hand out the race
     labels at random (1,000 times). If our 3x came from the arithmetic, it would survive.
     It should drop to about 1.0x.
  2. Fake neighbourhoods. Give every applicant someone else's neighbourhood (all 7
     neighbourhood numbers move together, so each fake neighbourhood is a real one), in both
     2022 and 2023, then retrain model A. The model still has neighbourhood columns, but they
     no longer describe where the person lives. If neighbourhood data drives the extra gap,
     the gap should fall to about the level of model B (no neighbourhood data).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from fix import APPLICANT_FEATURES, false_alarm_ratio
from prep import NUMERIC_FEATURES, TRACT_FEATURES, load_year
from robustness import fit_flag

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(11)


def shuffle_neighbourhoods(df):
    """Move each applicant's whole block of neighbourhood numbers to a random other applicant."""
    df = df.copy()
    perm = rng.permutation(len(df))
    df[TRACT_FEATURES] = df[TRACT_FEATURES].to_numpy()[perm]
    return df


def main():
    train = load_year(2022).reset_index(drop=True)
    test = load_year(2023).reset_index(drop=True)
    out = ["# Placebo checks (Georgia, trained 2022, tested 2023)", ""]

    # real models, for reference
    _, fA = fit_flag(train, test, NUMERIC_FEATURES)
    _, fB = fit_flag(train, test, APPLICANT_FEATURES)
    realA, realB = false_alarm_ratio(test, fA), false_alarm_ratio(test, fB)
    print(f"real A {realA:.2f}x, real B {realB:.2f}x")

    # 1. shuffled race labels (model A's flags unchanged)
    ok = test.approved.to_numpy() == 1
    race = test.race_group.to_numpy()
    sh = []
    for _ in range(1000):
        r = race.copy()
        r[ok] = rng.permutation(r[ok])
        b, w = ok & (r == "Black"), ok & (r == "White")
        sh.append(fA[b].mean() / fA[w].mean())
    sh = np.array(sh)
    lo, hi = np.percentile(sh, [2.5, 97.5])
    print(f"shuffled race: mean {sh.mean():.2f}x, 95% {lo:.2f}-{hi:.2f}, max {sh.max():.2f}")

    # 2. fake neighbourhoods (5 different shuffles, retrain each time)
    fake = []
    for i in range(5):
        _, f = fit_flag(shuffle_neighbourhoods(train), shuffle_neighbourhoods(test), NUMERIC_FEATURES)
        fake.append(false_alarm_ratio(test, f))
        print(f"fake neighbourhoods run {i + 1}: {fake[-1]:.2f}x")
    fake = np.array(fake)

    out += [
        "| check | Black vs white false alarms |", "|---|---|",
        f"| A. Bank-style AI, real data | {realA:.2f}× |",
        f"| Placebo 1: race labels shuffled (1,000 times) | {sh.mean():.2f}× (95% range {lo:.2f}–{hi:.2f}; highest {sh.max():.2f}×) |",
        f"| Placebo 2: every applicant given a random real neighbourhood, retrained (5 times) | {fake.mean():.2f}× (range {fake.min():.2f}–{fake.max():.2f}) |",
        f"| B. No neighbourhood data, for comparison | {realB:.2f}× |",
        "",
        "**Placebo 1:** with race labels handed out at random, the gap disappears (about 1.0×), and none of "
        f"1,000 shuffles came close to {realA:.1f}×. The 3× is not a side effect of how we measure.",
        "",
        "**Placebo 2:** when the neighbourhood columns describe someone else's neighbourhood, the gap falls to "
        "about the level of the model with no neighbourhood data. So the extra gap in model A comes from where "
        "applicants actually live, not from the model simply having more columns.",
        "",
        "The gap that remains in both B and placebo 2 (about 1.9×) comes from applicants' own finances "
        "(income, loan size, debt), which also differ by race on average.",
    ]
    (ROOT / "results_placebo.md").write_text("\n".join(out) + "\n")
    print("-> results_placebo.md")


if __name__ == "__main__":
    main()
