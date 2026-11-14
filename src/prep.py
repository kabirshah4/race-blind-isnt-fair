"""Clean raw HMDA Georgia files into a modelling table.

Run:  .venv/bin/python src/prep.py
Reads data/raw/ga_{2022,2023}.csv, writes data/processed/hmda_ga.parquet.

Key decisions (each one is explained in the write-up):
  * Scope: first-lien, owner-occupied, site-built single-family home purchases,
    not business loans, not reverse mortgages, not lines of credit.
  * Label: approved = action_taken 1 (originated) or 2 (approved, not accepted);
    denied = action_taken 3. Withdrawn/incomplete files were never downloaded.
  * Leakage: columns that only exist after a decision (interest rate, costs,
    denial reasons, purchaser) or whose "not applicable" codes mostly appear on
    denials (AUS result, credit score model) are NOT used as features.
  * LTV is recomputed from loan_amount / property_value because the reported LTV
    is missing far more often for denied files (34%) than approved ones (7.5%).
  * Protected attributes (race, ethnicity, sex, age) are kept ONLY for the audit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "hmda_ga.parquet"

DTI_BINS = {"<20%": 15, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}

NUMERIC_FEATURES = [
    "income", "loan_amount", "property_value", "ltv", "dti",
    "loan_term_360", "has_coapplicant",
    "tract_minority_population_percent", "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income", "tract_population",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]
# Two more leaks we caught later (an independent review found them):
#   preapproval: denied pre-approval requests are filed under a different action code
#     (7) that we never download, so "asked for pre-approval" = 0% denied in our data.
#   initially_payable_to_institution: code 3 means "not applicable", which mostly
#     happens when no loan was made, so it is 83% denied.
# Both give away the answer, so neither is a feature.
CATEGORICAL_FEATURES = ["loan_type", "conforming_loan_limit", "submission_of_application"]
# Features that describe the NEIGHBOURHOOD, not the applicant: candidate proxies.
TRACT_FEATURES = [c for c in NUMERIC_FEATURES if c.startswith(("tract_", "ffiec_"))]
PROTECTED = ["race_group", "sex_group", "age_group"]

# ONE XGBoost setting used by every script (train, audit, fix), so every number in the
# write-up comes from the same kind of model.
XGB_PARAMS = dict(n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.8,
                  colsample_bytree=0.8, n_jobs=-1, random_state=42)


def parse_dti(v):
    if pd.isna(v):
        return np.nan
    v = str(v)
    if v in DTI_BINS:
        return DTI_BINS[v]
    try:
        return float(v)
    except ValueError:  # "Exempt"
        return np.nan


def race_group(row):
    if row.derived_ethnicity == "Hispanic or Latino":
        return "Hispanic"
    r = row.derived_race
    if r == "White":
        return "White"
    if r == "Black or African American":
        return "Black"
    if r == "Asian":
        return "Asian"
    if r in ("Race Not Available", "Free Form Text Only"):
        return "Not reported"
    return "Other/Joint"


def load_year(year: int, state: str = "ga") -> pd.DataFrame:
    """Same cleaning for any state; Georgia is the main study, NC is the second-state check."""
    df = pd.read_csv(RAW / f"{state}_{year}.csv", low_memory=False, dtype=str)
    n0 = len(df)
    keep = (
        (df.lien_status == "1")
        & (df.occupancy_type == "1")
        & (df.derived_dwelling_category == "Single Family (1-4 Units):Site-Built")
        & (df.business_or_commercial_purpose == "2")
        & (df.reverse_mortgage == "2")
        & (df["open-end_line_of_credit"] == "2")
        & (df.action_taken.isin(["1", "2", "3"]))
    )
    df = df[keep].copy()

    out = pd.DataFrame(index=df.index)
    out["year"] = year
    out["approved"] = df.action_taken.isin(["1", "2"]).astype(int)

    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    out["income"] = num("income")                      # $ thousands
    out["loan_amount"] = num("loan_amount") / 1000      # -> $ thousands
    out["property_value"] = num("property_value") / 1000
    out["ltv"] = (out.loan_amount / out.property_value * 100).clip(upper=150)
    out["dti"] = df.debt_to_income_ratio.map(parse_dti)
    out["loan_term_360"] = (df.loan_term == "360").astype(int)
    out["has_coapplicant"] = (~df["co-applicant_age"].isin(["9999"])).astype(int)
    for c in TRACT_FEATURES:
        out[c] = num(c)
    for c in CATEGORICAL_FEATURES:
        out[c] = df[c].fillna("NA")

    out["race_group"] = df.apply(race_group, axis=1)
    out["sex_group"] = df.derived_sex.replace({"Sex Not Available": "Not reported"})
    out["age_group"] = df.applicant_age.replace({"8888": "Not reported"})
    out["county"] = df.county_code
    out["lender"] = df.lei

    # Drop rows we can't use. Logged so the write-up can report it honestly.
    before = len(out)
    # Missing address: HMDA stores a missing census tract as all-zero tract columns.
    # Denied files are ~5x more likely to have no tract, so "address missing" would
    # leak the answer. We drop these rows (~0.15%).
    has_tract = out.county.notna() & (out.tract_population > 0)
    out = out[(out.income > 0) & out.property_value.notna() & out.dti.notna() & has_tract]
    print(f"{state.upper()} {year}: raw {n0:,} -> scoped {before:,} -> usable {len(out):,} "
          f"| denial rate {1 - out.approved.mean():.1%}")
    return out


def main():
    df = pd.concat([load_year(2022), load_year(2023)], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"saved {len(df):,} rows -> {OUT.relative_to(ROOT)}")
    print("\nDenial rate by race group (raw data, 2023):")
    d23 = df[df.year == 2023]
    print((1 - d23.groupby("race_group").approved.mean()).round(3).sort_values().to_string())


if __name__ == "__main__":
    main()
