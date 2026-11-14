"""Make the charts for the write-up, video and finals slides.

Run:  .venv/bin/python src/charts.py      (after prep.py, train.py, fix.py)
Saves PNGs to figures/.

Charts
  1_denial_by_group.png        Raw 2023 denial rates by group (the starting fact)
  2_neighbourhood_proxy.png    Denial rate vs % minority residents in the
                               neighbourhood: why neighbourhood is a stand-in
  3_the_fix.png                The main result: false-alarm gap for models A, B, C
                               with 95% confidence ranges
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# One small, consistent style for every chart.
INK, MUTED, ACCENT, GOOD = "#1d2a24", "#7a8a82", "#c2410c", "#15803d"
plt.rcParams.update({
    "font.family": "Helvetica Neue", "font.size": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved figures/" + name)


def chart_denial_by_group(df):
    d = df[(df.year == 2023) & df.race_group.isin(["White", "Asian", "Hispanic", "Black"])]
    rate = (1 - d.groupby("race_group").approved.mean()).sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [ACCENT if g == "Black" else MUTED for g in rate.index]
    bars = ax.barh(rate.index, rate.values, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, color=INK)
    ax.set_xlabel("Share of home-purchase applications denied")
    ax.set_title("Georgia 2023: Black applicants were denied 2.7× as often as white applicants",
                 loc="left", fontsize=12, color=INK)
    ax.set_xlim(0, rate.max() * 1.2)
    save(fig, "1_denial_by_group.png")


def chart_neighbourhood_proxy(df):
    d = df[df.year == 2023].copy()
    d["minority_bin"] = pd.cut(d.tract_minority_population_percent, bins=range(0, 101, 10))
    by_bin = d.groupby("minority_bin", observed=True).agg(
        denial=("approved", lambda s: 100 * (1 - s.mean())),
        black_share=("race_group", lambda s: 100 * (s == "Black").mean()))
    labels = [f"{int(b.left)}–{int(b.right)}" for b in by_bin.index]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(labels, by_bin.denial, marker="o", color=ACCENT, lw=2.2, label="Denial rate (left)")
    ax.set_xlabel("% minority residents in the applicant's neighbourhood (census tract)")
    ax.set_ylabel("Denial rate (%)", color=ACCENT)
    ax.set_ylim(0, by_bin.denial.max() * 1.15)
    # Second y-axis on the right for the share of applicants who are Black.
    ax2 = ax.twinx()
    ax2.plot(labels, by_bin.black_share, marker="o", color=MUTED, lw=1.5, ls="--",
             label="Applicants who are Black (right)")
    ax2.set_ylabel("Applicants who are Black (%)", color=MUTED)
    ax2.set_ylim(0, 100)
    ax2.spines["right"].set_visible(True)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=10, loc="upper left")
    ax.set_title("Denials rise with the neighbourhood's minority share, so the model can\n"
                 "learn race from location without ever seeing race",
                 loc="left", fontsize=12, color=INK)
    plt.setp(ax.get_xticklabels(), rotation=30)
    save(fig, "2_neighbourhood_proxy.png")


def chart_the_fix():
    r = pd.read_csv(ROOT / "results_fix.csv")
    lo_hi = r["95% range"].str.split("–", expand=True).astype(float)
    ratio = r["Black false alarms vs White"]
    names = ["A. Full model", "B. Remove\nneighbourhood", "C. + Fairness\ntraining"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = [ACCENT, "#d97706", GOOD]
    bars = ax.bar(names, ratio, color=colors, width=0.55,
                  yerr=[ratio - lo_hi[0], lo_hi[1] - ratio], capsize=6, ecolor=INK)
    for bar, v, auc in zip(bars, ratio, r["PR-AUC"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.22, f"{v:.2f}×",
                ha="center", fontsize=14, fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.08, f"PR-AUC {auc:.3f}",
                ha="center", fontsize=9.5, color="white")
    ax.axhline(1, color=INK, lw=1, ls=":")
    ax.text(2.42, 1.04, "equal", fontsize=9, color=INK)
    ax.set_ylabel("Qualified Black applicants wrongly\nflagged, vs. white applicants (×)")
    ax.set_ylim(0, 3.3)
    kept = r["PR-AUC"].iloc[2] / r["PR-AUC"].iloc[0]  # accuracy kept by the fair model
    ax.set_title(f"Race-blind models: false-alarm gap cut from {ratio.iloc[0]:.1f}× to {ratio.iloc[2]:.2f}×\n"
                 f"while keeping {kept:.0%} of the accuracy (PR-AUC)",
                 loc="left", fontsize=12, color=INK)
    save(fig, "3_the_fix.png")


if __name__ == "__main__":
    data = pd.read_parquet(ROOT / "data" / "processed" / "hmda_ga.parquet")
    chart_denial_by_group(data)
    chart_neighbourhood_proxy(data)
    chart_the_fix()
