# Case study: one bank-approved applicant model A gets wrong

A Black applicant in Georgia, 2023: about $85K a year, borrowing about $285K on a $295K home (97% loan-to-value), debts around 55% of income, FHA loan, no co-applicant. Their census tract is about 100% minority.

* **The bank approved them.**
* **Model A wrongly flags them:** 15% chance of denial, above its 15% cut-off.
* **Same person, typical mostly-white tract** (only the 7 neighbourhood numbers change): 11%, so model A would approve.
* **Our fixed model C** (no neighbourhood data): 9%, below its 15% cut-off, so it approves.

How common this is: of 2,718 bank-approved Black applicants model A wrongly flagged in 2023, 1,438 (53%) would be approved by model A at the mostly-white tract.

**Why we think model A gets this wrong:** nothing about this person's finances changed between the two runs; only the neighbourhood numbers did. Model A learned from past bank decisions that applications from high-minority tracts were denied more often, so it carries that pattern forward as if the address were a risk factor. Model C never sees the neighbourhood, so it can't.

Picked by `src/case_study.py`: the wrongly-flagged, address-flips, model-C-approves applicant closest to that group's median income, loan and debt-to-income. Numbers rounded; no lender, county or tract named.
