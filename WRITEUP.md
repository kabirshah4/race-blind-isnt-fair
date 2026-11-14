# Race-Blind Isn't Fair: Auditing a Mortgage-Approval Model for Hidden Redlining

**Track:** AI + Finance
**Team:** Kabir Shah, [partner's name]
**Starting point:** Our own idea (related to suggested project 4.2.3, but on real 2022–2023 US mortgage data instead of German Credit)

*This project is a prototype and is not financial advice.*

## The problem

Lenders increasingly use machine-learning models to decide who gets a mortgage, and a common defence is that the model "never sees race." Redlining (denying loans based on who lives in a neighbourhood) was outlawed in 1968, but a model can rebuild it without anyone writing it in, because where you live is strongly tied to your race. We test whether a race-blind mortgage model is still unfair, find out why, and fix it. Our users are **a first-time home buyer who wants to know if a decision about them was fair**, and **the community groups and regulators** who audit lenders.

## What we built

1. A mortgage-denial model trained on **264,340 real Georgia home-purchase applications** (public HMDA data). Race, sex and age are never given to it.
2. A **fairness audit** that measures, for each group, how often the model wrongly flags people the bank actually approved (false alarms).
3. A **"proxy detective"** test that shows which inputs secretly carry race information.
4. Two **fixes**, with the accuracy cost of each.
5. A **Gradio demo**: enter an applicant, switch only the neighbourhood, and compare the standard model with the fairness-checked one. If an applicant is flagged, it shows the smallest realistic change that would flip the decision.

## What we added

This isn't one of the suggested projects. Beyond a standard fairness audit we added: **real, current US data** (HMDA 2022–2023) with a **train-on-2022, test-on-2023** split; **two data-leakage fixes** we found ourselves; a **proxy-detection test**; a **fairness-constrained model**; **bootstrap confidence intervals**; and a finding about how the choice of fairness metric can hide unfairness (see Results).

## How it works

**Data.** HMDA, the Home Mortgage Disclosure Act (1975), requires lenders to publish every application. We used Georgia, home purchases, 2022 and 2023, first-lien, owner-occupied, site-built homes, approved or denied (146,793 applications in 2022, 117,547 in 2023; 9.8% denied). Race, sex and age are kept only for the audit.

**Leakage checks.** We compared how often each column is blank for approved vs. denied applications. Interest rate, rate spread and loan costs are **100% blank for denials**, so we dropped them. We dropped denial reasons and underwriting-system codes too. The reported loan-to-value ratio was missing 34% of the time for denials vs. 7.5% for approvals, so we **recalculated it** from loan amount ÷ property value. We later found that **missing addresses** (stored as all-zero tract data) were 5× more common among denials, so we removed those 0.15% of rows and re-ran everything.

**Model.** 19 inputs: income, loan amount, property value, loan-to-value, debt-to-income, loan term and type, co-applicant, pre-approval, plus 7 census-tract (neighbourhood) statistics. We compared a logistic regression with **XGBoost**. Denials are rare, so we report precision, recall and PR-AUC rather than accuracy. Each model flags the same share of applicants (the 2022 denial rate), with the cut-off chosen on 2022 data only.

**Fairness tests.** (1) The false-alarm rate per group among bank-approved applicants, as a ratio to white applicants. (2) A "detective" XGBoost that tries to predict Black vs. white from the lending model's inputs (ROC-AUC). (3) Retraining without the neighbourhood features. (4) Fairlearn's Exponentiated Gradient with a false-positive-rate parity constraint. Race is used only during training, **never to make a decision** (US lending law bans using race in credit decisions).

## Results (all on held-out 2023 data, 117,547 applications)

| Model | PR-AUC | Precision / Recall | Black vs. white false alarms (95% CI) |
|---|---|---|---|
| Baseline: approve everyone | 0.098 | 0 / 0 | — |
| Baseline: deny if DTI > 43% | 0.342 | 0.15 / 0.67 | — |
| **A. XGBoost, all features** | **0.617** | 0.55 / 0.56 | **2.54×** (2.40–2.70) |
| B. No neighbourhood features | 0.613 | | **1.56×** (1.46–1.65) |
| C. B + fairness constraint | 0.597 | | **1.05×** (0.99–1.12) |

- **The model works:** its PR-AUC is about 1.8× the classic DTI rule's.
- **It is unfair even though it is race-blind:** model A wrongly flags qualified Black applicants **2.5×** as often as qualified white applicants (9.0% vs. 3.5%).
- **Neighbourhood is the proxy:** the detective predicts race with ROC-AUC **0.88** from the model's inputs (0.845 from the neighbourhood features alone). The strongest single proxy is the tract's minority share (0.81).
- **Same person, different address:** 1,131 qualified Black applicants were flagged by model A but not model C. Changing *only their neighbourhood* to a typical mostly-white tract made model A approve **85%** of them.
- **The fix:** removing neighbourhood data cut the gap to 1.56× for almost no accuracy loss. Adding the fairness constraint brought it to **1.05× (no measurable gap)** while keeping **97%** of the accuracy.
- **What surprised us:** fairlearn's default *difference*-based constraint said the model was already fair. False-alarm rates are small (2–9%), so a gap of a few points passed even though it meant about 1.4× more false alarms for Black applicants. Switching to a *ratio* constraint fixed it. **How you measure fairness decides whether you find unfairness.**

**Where it fails / limitations.** HMDA has **no credit scores**, a major factor in real lending, so some of the remaining gaps may come from information we can't see. Our labels are past bank decisions, so the model learns the banks' patterns; we measure **disparities, not intent**. A 1.6× gap remained after removing neighbourhood data, because loan-to-value and debt-to-income also carry race signal (reflecting real wealth gaps). One state, two years.

## What we'd do next

1. Repeat the audit across all 50 states and more years to see whether the neighbourhood-proxy effect is universal.
2. Test the fix against real lender policies, and measure how the fair model changes outcomes for Hispanic and Asian applicants as well.

## Sources

- Data: CFPB / FFIEC HMDA Data Browser, https://ffiec.cfpb.gov/data-browser/ (Georgia, 2022–2023)
- Libraries: pandas, scikit-learn, XGBoost, fairlearn (Microsoft), SHAP, matplotlib, Gradio
- Background: Fair Housing Act (1968), Equal Credit Opportunity Act (1974), Home Mortgage Disclosure Act (1975)
- **AI tools:** Claude Code (Anthropic), used to help write the data-cleaning, modelling and audit code and to draft this write-up. We reviewed every step and can explain all of it.
