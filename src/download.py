"""Download the raw HMDA data used in this project (public, free, no account needed).

Run:  python src/download.py
Saves data/raw/{ga,nc}_{2022,2023}.csv (about 270 MB total). Georgia is the main study;
North Carolina is the second-state check.

What we ask the CFPB HMDA Data Browser API for:
  Georgia (GA), one year at a time, home-purchase loans (loan_purposes=1),
  and only applications that ended in a decision:
  action_taken 1 = originated, 2 = approved but not accepted, 3 = denied.
"""
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = ("https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
       "?states={state}&years={year}&actions_taken=1,2,3&loan_purposes=1")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for state in ("GA", "NC"):
      for year in (2022, 2023):
        out = RAW / f"{state.lower()}_{year}.csv"
        print(f"downloading {state} {year} ...")
        r = requests.get(URL.format(state=state, year=year), timeout=600)  # follows the redirect to the file
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"  saved {out} ({len(r.content) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
