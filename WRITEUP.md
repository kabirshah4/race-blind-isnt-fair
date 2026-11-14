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
