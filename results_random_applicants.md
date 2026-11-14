# Random real applicants: does the address alone change the answer?

500 applications drawn at random (seed 7) from all 117,547 Georgia 2023 test applications. Each one's real finances are scored by model A at three typical neighbourhood profiles (median tract values for under 20%, 40-60% and over 90% minority, `models/neighbourhoods.json`). A 'flip' means model A approves at one profile and flags at another.

**Model A flips for 55 of 500 (11%, about 1 in 9).** Model C never sees the neighbourhood, so 0 by design.

Caveat: the neighbourhoods are three typical profiles, not each applicant's own alternatives. This is the same calculation the live demo runs in the browser.
