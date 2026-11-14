# FairView: technical details

Everything behind the one-page [WRITEUP.md](WRITEUP.md): methods, every check, and the caveats a careful reviewer should know. Every number cites a `results_*.md` file produced by a script in `src/`.

**Where every headline number comes from**

| Number in the write-up | File | Script |
|---|---|---|
| PR-AUC 0.516 vs 0.342; 3.01× → 1.88× → 1.21×; 12.0% vs 4.0%; 2,718 | `results_fix.md` | `src/fix.py` |
| 2.9× average (2.85–2.95×) over 5 retrains | `results_seeds.md` | `src/seeds.py` |
| 18.4% flagged vs 15.7% denied (Black); 6.1% vs 5.8% (white); proxy ROC-AUC 0.88 | `results_audit.md` | `src/audit.py` |
| 1 in 9 of 500 random real applicants (55) | `results_random_applicants.md` | `src/export_models.py` |
| 3.11× with county credit data; 2.78–3.28× by tract | `results_robustness.md` | `src/robustness.py` |
| White false alarms 4.0% → 5.6%; misses; examples | `results_failures.md` | `src/failures.py` |
| North Carolina 2.09× → 1.42× | `results_nc.md` | `src/other_state.py` |
| 30 of 60 GA lenders (after correction); NC 24 of 42 | `results_lenders_strict.md`, `results_lenders_strict_nc.md` | `src/lender_chart.py` |

**Leakage (three rounds).** (1) Interest rate, rate spread, loan costs and denial reasons only exist after a decision, so we dropped them; the reported loan-to-value was missing for 34% of denials vs. 7.5% of approvals, so we recalculated it as loan ÷ property value. (2) Missing addresses (stored as all-zero tract data) were 5× more common among denials, so we removed those 0.15% of rows. (3) An independent review found `preapproval` (denied pre-approval requests use action code 7, which we never download, so "asked for pre-approval" meant 0% denied) and `initially_payable_to_institution` (code 3 = 83% denied). We removed both and re-ran everything. All numbers are after all three fixes.

**Set-up.** One XGBoost setting is used by every script. Each model's cut-off is set to flag the 2022 denial rate on 2022 data; on 2023 that is 10.3% of applicants for A and 9.5% for C. At the strictest DTI rule's flag rate (DTI > 55%, 2.7% of applicants), the model's precision is 0.94 vs. 0.86 for the rule. A plain logistic regression (PR-AUC 0.338) did not beat the rule, which is why we used XGBoost.

**Proxy detail.** Detective ROC-AUC for guessing Black vs. white: all inputs 0.88, neighbourhood only 0.845, applicant finances only 0.765. Dropping only "% minority residents" leaves 2.32× (2.19–2.43), so the proxy is spread across several neighbourhood inputs.

**Counterfactual detail** (`results_counterfactual.md`). Swapping all 7 neighbourhood numbers to a typical mostly-white tract flips 53% of wrongly flagged bank-approved Black applicants vs. 22% of white ones. Swapping only "% minority" flips 49% vs. 16%; only the other 6 numbers, 30% vs. 17%. Restricted to people near the cut-off (risk under 1.5× the cut-off), all 7: 70% vs. 37%.

**Fairness-constrained model C.** fairlearn's Exponentiated Gradient with false-positive-rate parity. The rule was chosen on a 20% validation split of 2022: the default *difference* rule ("within 1 point") left 1.74× at our cut-off, the *ratio* rule (≥ 0.9) left 1.31×. fairlearn enforces its rule on its own randomized classifier at a 0.5 threshold, while we average the sub-models' scores and use our cut-off, so this is an observation, not a proof, and C's fairness is measured on held-out data, not guaranteed. C was trained on a 60,000-row sample of 2022 because the method is slow, so some of its PR-AUC drop may come from less data.

**Costs by group** (`results_fix.md`). False alarms (white / Black): A 4.0% / 12.0%, C 5.6% / 6.8%. Recall: A 39.5% / 52.4%, C 41.2% / 46.5%. Precision: C 31.2% / 56.1%. Equal false-alarm rates with different base rates mean unequal precision (the Chouldechova / Kleinberg trade-off); we chose false alarms because wrongly denying a bank-approved person is the harm we care about.

**Confidence intervals** resample only the 2023 test rows (200 draws). They don't include training randomness or clustering by lender or tract, so they are somewhat too narrow.

**Second state (North Carolina, `src/other_state.py`, `results_nc.md`).** NC 2023 denial rates: white 5.0%, Black 13.7%. Test 1, Georgia models unchanged on NC 2023: A gives 1.95× (1.81–2.08), C gives 1.46× (1.38–1.56). Test 2, the same recipe trained on NC 2022: A 2.09×, B 1.72×, C 1.42× (1.31–1.54); PR-AUC 0.489 / 0.488 / 0.475. The direction of every result repeats, but the gaps are smaller and the fairness rule removes less of the gap than in Georgia.

**Robustness** (`src/robustness.py`, `results_robustness.md`). The missing-credit-score objection, tested four ways. Adding the Urban Institute's county credit data (share with debt in collections, card delinquency; Aug 2023 bureau records, `data/external/`) to model A: 3.11× (vs 3.01×). Dropping every application denied for credit history (HMDA denial reasons) and retraining: 2.97×. Confidence ranges that resample whole census tracts (2,740): 2.78–3.28×. At the same flag rate as A, C is 1.22×; C retrained on all 146,793 rows (`results_robustness_c.md`) reaches 0.95× at PR-AUC 0.494. Baselines: tuned logistic regression 0.439, HistGradientBoosting 0.515 (and 2.64×, so the gap is not XGBoost-specific). Note: county credit data itself carries race; it pushes model B from 1.88× to 2.53×. Lender audit with county credit: median gap 1.65× → 1.50×, 26 of 60 still above 1 (simple ranges).

**Where it fails** (`src/failures.py`, `results_failures.md`). Model A misses 53% of real denials; 53% of those misses were denied for reasons public data can't show (credit history, incomplete or unverifiable applications), using HMDA's own denial-reason field. Swapping only the neighbourhood clears 53% of the 2,718 Black false alarms. Model C newly flags 1,215 bank-approved white applicants (net +747), catches 44% of denials vs 47%, and also lowers Hispanic false alarms (1.61× → 0.96× the white rate).

**Lender audit** (`src/audit_lender.py`, `results_lenders.md`, `results_lenders_nc.md`). A finances-only XGBoost (no race, no neighbourhood) trained on the previous year gives each application an expected denial chance. For each lender: excess = observed ÷ expected denial rate per group, and finance-adjusted gap = Black excess ÷ white excess, with a 1,000-draw bootstrap range; `src/lender_chart.py` adds a Benjamini-Hochberg correction for testing 60 lenders at once (30 of 60 still stand out) and a chart, `figures/4_lenders.png`. Lenders need 50+ Black and white applications and 5+ denials per group. Results are anonymised; single-lender reports (`--lei`) are kept out of the repo.

**Demo example.** The demo's applicant is constructed: one of 174 combinations we tested where the address alone flips model A. Model C ignores neighbourhood, so its answer can't change with the address by design. The demo illustrates the result; the evidence is the test-set audit above.

---

**Base rates vs amplification (the most important caveat).** Banks denied 15.7% of Black and 5.8% of white applicants in 2023, so even a perfectly calibrated race-blind model would have a higher false-alarm rate for Black applicants. The sign that model A *amplifies* the pattern rather than just copying it: it flags 18.4% of Black applicants vs 15.7% actually denied (×1.17), but 6.1% vs 5.8% of white applicants (×1.05), and 7.1% vs 8.9% of Asian applicants (`results_audit.md`). The near-cut-off address swap (70% of borderline Black false alarms flip vs 37% for white, `results_counterfactual.md`) points the same way.

**How much is "neighbourhood"? It depends on the scale.** Dropping neighbourhood inputs moves A from 3.01× to 1.88×. As a share of the excess ratio that is (3.01−1.88)/(3.01−1) = 56%; on a log scale it is ln(3.01/1.88)/ln(3.01) = 43%; in percentage points of the false-alarm gap it is about 45% (12.0−4.0 vs 9.4−5.0). So "about half". This is a drop-and-retrain ablation, not an attribution: finance inputs can absorb some neighbourhood signal once tract data is removed, and B is still 1.88×.

**What each robustness check can and can't show** (`results_robustness.md`). County credit data (Urban Institute "Debt in America", 2024 update built from August 2023 credit-bureau records; applied to both 2022 and 2023 rows) is a county average, not applicant credit, and it carries race itself (it raises model B from 1.88× to 2.53×). Removing credit-history denials only changes the *training* data and the denied side of the test set; false alarms are measured on approved applicants, so this is a weak test of whether the model learned the gap from credit-history denials, not proof that credit doesn't matter. The strongest checks are the five retrains (2.85–2.95×; our published seed is the high draw), the tract-clustered ranges (2.78–3.28×), and a different model family (HistGradientBoosting, 2.64×).

**Model C is effective but not stable in size.** 1.21× as published (60,000-row training sample), 1.12–1.28× across seeds, 1.22× at A's flag rate, 0.95× when trained on all 146,793 rows. It averages fairlearn's sub-models instead of using the randomized classifier, so fairness is measured, not guaranteed. **In North Carolina, the Georgia-trained C raises false alarms for both groups** (white 4.3% → 6.3%, Black 8.4% → 9.2%) while narrowing the ratio from 1.95× to 1.46×; retrained on NC data it does better (white 4.2%, Black 5.9%).

**Lender counts.** The number of Georgia lenders whose simple 95% range sits above 1 moves between 30 and 32 with the random bootstrap draws (`results_lenders.md`, `results_robustness.md`). The headline 30 of 60 uses `results_lenders_strict.md`: 1,000 draws plus a Benjamini-Hochberg correction for testing 60 lenders at once. North Carolina's 24 of 42 holds after the same correction (`results_lenders_strict_nc.md`, `figures/4_lenders_nc.png`).

**Demo details.** The demo's "1 in 9" counts a flip if model A's answer changes between three typical neighbourhood profiles (median tract values, which also differ in metro median income); submission channel is fixed to "direct", and the conforming-loan limit is the 2023 value ($726,200).
