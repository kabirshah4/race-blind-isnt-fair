# Race-Blind Isn't Fair: Auditing a Mortgage Model for Neighbourhood Proxies

**Track:** AI + Finance · **Team:** Kabir Shah, Vishnu Veer
**Starting point:** Extends suggested project 4.2.3 (fairness audit of a loan model) to real 2022–2023 US mortgage data

*This project is a prototype and is not financial advice.*

## The problem

Lenders increasingly use machine learning to approve mortgages, and a common defence is that the model "never sees race." But where you live is closely tied to race, so a model can rebuild redlining (outlawed in 1968) without anyone writing it in. We built a race-blind mortgage model and tested whether it treats Black and white applicants differently. Our users are **fair-lending auditors and community groups** checking a lender's model.

## What we built

A denial-prediction model trained on **264,340 real Georgia home-purchase applications** (public HMDA data) that never sees race, sex or age; a **fairness audit** of its mistakes by group; a **"proxy detective"** that shows which inputs carry race; two **fixes** with their costs; and a **demo app** where you change only an applicant's neighbourhood and compare the original model with the fixed one.

## What we added (beyond 4.2.3)

Real, current data with a **train-on-2022, test-on-2023** split; **three data leaks** found and removed; the proxy detective; a **"same person, different address"** test; a **fairness-constrained model** whose rule was chosen on 2022 validation data only; and **95% bootstrap confidence intervals**.

## How it works

**Data:** Georgia home-purchase applications that were approved or denied (146,793 in 2022, 117,547 in 2023; 9.8% denied). **Leaks removed:** columns that only exist after a decision (interest rate, costs), rows with missing addresses, and two columns that gave the answer away (`preapproval`, `initially_payable_to_institution`). **Model:** XGBoost on 17 inputs (income, loan, home value, loan-to-value, debt-to-income, loan type, plus 7 neighbourhood statistics). **Fairness measure:** the *false-alarm rate*, meaning the share of applicants the bank approved that the model flags as "likely denied," compared between Black and white applicants.

## Results (2023 test data, 117,547 applications)

| Model | PR-AUC | Black vs. white false alarms (95% CI) |
|---|---|---|
| Baseline: deny if debt-to-income > 43% | 0.342 | — |
| **A. XGBoost, all inputs** | **0.516** | **3.01×** (2.86–3.17) |
| B. Drop all neighbourhood inputs | 0.511 | 1.88× (1.79–1.98) |
| C. B + fairness constraint (race used only in training) | 0.501 | **1.21×** (1.14–1.27) |

- **The model works:** 1.5× the PR-AUC of the classic lending rule.
- **Race-blind, yet unfair:** model A wrongly flags **12.0%** of qualified Black applicants vs. **4.0%** of qualified white applicants.
- **Neighbourhood is a major proxy:** a second model can guess race from A's inputs with ROC-AUC **0.88** (0.5 = coin flip). Neighbourhood inputs explain about **56%** of the excess gap; the rest comes through loan-to-value, debt-to-income and loan type.
- **Same person, different address:** for qualified Black applicants A wrongly flags, changing **only "% minority residents"** in their neighbourhood flips **49%** to approved (16% for white applicants).
- **The fixes:** dropping neighbourhood inputs costs almost nothing (0.516 → 0.511) and needs no race data, a *less discriminatory alternative*. Adding a fairness constraint cuts the gap to **1.21×** while keeping 97% of the PR-AUC. **Cost:** white false alarms rise from 4.0% to 5.6%.

**Limitations.** HMDA has no credit scores. Our labels are past bank decisions, so we measure **disparities, not intent**. Equal false-alarm rates mean unequal precision by group, a known fairness trade-off. Using race in training is legally debated; fix B avoids it. One state, two years. Details are in the Technical appendix.

## What we'd do next

Repeat the audit across all states and years with lender-level effects, and test it against lenders' real underwriting rules.

## Sources

Data: CFPB / FFIEC HMDA, https://ffiec.cfpb.gov/data-browser/ · Libraries: pandas, scikit-learn, XGBoost (https://xgboost.ai), fairlearn (https://fairlearn.org), matplotlib, Gradio (https://gradio.app) · Laws: Fair Housing Act 1968, ECOA 1974, HMDA 1975 · **AI tools:** Claude Code (Anthropic, https://claude.com/claude-code), used to help write code, review methods and draft this write-up; we reviewed and can explain every step. Explainer graphics and voice: HyperFrames, Kokoro TTS.

---

## Technical appendix

**Leakage (three rounds).** (1) Interest rate, rate spread, loan costs and denial reasons only exist after a decision, so we dropped them; the reported loan-to-value was missing for 34% of denials vs. 7.5% of approvals, so we recalculated it as loan ÷ property value. (2) Missing addresses (stored as all-zero tract data) were 5× more common among denials, so we removed those 0.15% of rows. (3) An independent review found `preapproval` (denied pre-approval requests use action code 7, which we never download, so "asked for pre-approval" meant 0% denied) and `initially_payable_to_institution` (code 3 = 83% denied). We removed both and re-ran everything. All numbers are after all three fixes.

**Set-up.** One XGBoost setting is used by every script. Each model flags the same share of applicants (the 2022 denial rate), with the cut-off chosen on 2022 data. At the strictest DTI rule's flag rate (DTI > 55%, 2.7% of applicants), the model's precision is 0.94 vs. 0.86 for the rule. A plain logistic regression (PR-AUC 0.338) did not beat the rule, which is why we used XGBoost.

**Proxy detail.** Detective ROC-AUC for guessing Black vs. white: all inputs 0.88, neighbourhood only 0.845, applicant finances only 0.765. Dropping only "% minority residents" leaves 2.32× (2.19–2.43), so the proxy is spread across several neighbourhood inputs.

**Counterfactual detail** (`results_counterfactual.md`). Swapping all 7 neighbourhood numbers to a typical mostly-white tract flips 53% of wrongly flagged qualified Black applicants vs. 22% of white ones. Swapping only "% minority" flips 49% vs. 16%; only the other 6 numbers, 30% vs. 17%. Restricted to people near the cut-off (risk under 1.5× the cut-off), all 7: 70% vs. 37%.

**Fairness-constrained model C.** fairlearn's Exponentiated Gradient with false-positive-rate parity. The rule was chosen on a 20% validation split of 2022: the default *difference* rule ("within 1 point") left 1.74× at our cut-off, the *ratio* rule (≥ 0.9) left 1.31×. fairlearn enforces its rule on its own randomized classifier at a 0.5 threshold, while we average the sub-models' scores and use our cut-off, so this is an observation, not a proof, and C's fairness is measured on held-out data, not guaranteed. C was trained on a 60,000-row sample of 2022 because the method is slow, so some of its PR-AUC drop may come from less data.

**Costs by group** (`results_fix.md`). False alarms (white / Black): A 4.0% / 12.0%, C 5.6% / 6.8%. Recall: A 39.5% / 52.4%, C 41.2% / 46.5%. Precision: C 31.2% / 56.1%. Equal false-alarm rates with different base rates mean unequal precision (the Chouldechova / Kleinberg trade-off); we chose false alarms because wrongly denying a qualified person is the harm we care about.

**Confidence intervals** resample only the 2023 test rows (200 draws). They don't include training randomness or clustering by lender or tract, so they are somewhat too narrow.

**Demo example.** The demo's applicant is constructed: one of 174 combinations we tested where the address alone flips model A. Model C ignores neighbourhood, so its answer can't change with the address by design. The demo illustrates the result; the evidence is the test-set audit above.

---

## How to run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download.py        # 1. download the public HMDA data (Georgia 2022–2023)
python src/prep.py            # 2. clean it, remove leaky columns -> data/processed/
python src/train.py           # 3. baselines vs. models -> results_models.csv
python src/fix.py             # 4. models A/B2/B/C, validation choice of fairness rule, 95% CIs, per-group costs -> results_fix.md
python src/audit.py           # 5. fairness audit of model A + proxy detective -> results_audit.md
python src/counterfactual.py  # 6. "same person, different address" -> results_counterfactual.md
python src/charts.py          # 7. charts -> figures/
python app/app.py             # 8. the demo app -> http://127.0.0.1:7860
```

Every script starts with a plain-English explanation of what it does and why.
**[NOTES.md](NOTES.md)** is our full step-by-step project log, including the leaks we caught and the review that found the third one.

| Folder | What's in it |
|---|---|
| `src/` | The pipeline: download → prep → train → fix → audit → counterfactual → charts |
| `app/` | The Gradio demo |
| `figures/` | Charts used in the write-up |
| `models/` | Trained models (so the demo runs without retraining) |
| `results_*.csv/md` | Raw result tables behind every number above |

![The fix](figures/3_the_fix.png)
