# Race-Blind Isn't Fair: Auditing a Mortgage Model for Neighbourhood Proxies

**Track:** AI + Finance · **Team:** KV Gods (Kabir Shah, Vishnu Veer)
**Starting point:** Extends suggested project 4.2.3 (fairness audit of a loan model) to real 2022–2023 US mortgage data
**Live demo:** https://kabirshah4.github.io/race-blind-isnt-fair/ · **Code:** https://github.com/kabirshah4/race-blind-isnt-fair

*This project is a prototype and is not financial advice.*

## Why we care

Both of our families immigrated from India, and our parents went through buying a home. We wanted to know whether something besides their finances, like race, could quietly affect that decision. In Georgia's 2023 data, banks denied Asian Indian applicants 10.8% of the time vs. 5.8% for white applicants (before accounting for finances). Our race-blind AI didn't over-flag qualified Asian Indian applicants, but it did wrongly flag qualified **Black** applicants **3×** as often. A bias you can't see can hit any family, so it has to be checked.

## The problem

Lenders use machine learning to approve mortgages and often defend it because the model "never sees race." But where you live is closely tied to race, so a model can rebuild redlining (outlawed in 1968) without anyone writing it in. Our user is **a fair-lending analyst at a housing nonprofit or state regulator** deciding whether a lender's "race-blind" model, or the lender itself, deserves a closer look.

## What we built

A denial-prediction model built on **264,340 real Georgia home-purchase applications** (public HMDA data; trained on 146,793 from 2022, tested on 117,547 from 2023) that never sees race, sex or age; a **fairness audit** of its mistakes by group; a **"proxy detective"** that shows which inputs carry race; two **fixes** with their costs; a **lender-audit tool** an analyst can run on any lender; and a **demo app** where you change only an applicant's address.

## What we added (beyond 4.2.3)

Real, current data with a time-based split; a **second-state check in North Carolina**, where we live; **three data leaks** found and removed; the proxy detective; a **"same person, different address"** test; a fairness rule chosen on 2022 validation data only; bootstrap confidence intervals; and a **lender-by-lender audit** separating gaps explained by finances from gaps that aren't.

## How it works

**Model:** XGBoost on 17 inputs (income, loan, home value, loan-to-value, debt-to-income, loan type, plus 7 neighbourhood statistics). **Leaks removed:** columns only filled in after a decision, rows with missing addresses, and two columns that gave the answer away. **Fairness measure:** the *false-alarm rate*, the share of applicants the bank approved that the model flags as "likely denied," compared between Black and white applicants.

## Results (2023 test data)

| Model | PR-AUC | Black vs. white false alarms (95% CI) |
|---|---|---|
| Baseline: deny if debt-to-income > 43% | 0.342 | — |
| **A. XGBoost, all inputs** | **0.516** | **3.01×** (2.86–3.17) |
| B. Drop all neighbourhood inputs | 0.511 | 1.88× (1.79–1.98) |
| C. B + fairness constraint (race used only in training) | 0.501 | **1.21×** (1.14–1.27) |

- **The model works:** 1.5× the PR-AUC of the classic lending rule.
- **Race-blind, yet unfair:** model A wrongly flags **12.0%** of qualified Black applicants vs. **4.0%** of white: **2,718 qualified Black Georgians** in one year.
- **Neighbourhood is a major proxy:** a second model guesses race from A's inputs with ROC-AUC **0.88** (0.5 = coin flip). Neighbourhood inputs explain about **56%** of the excess gap; the rest comes through loan-to-value, debt-to-income and loan type. Changing **only "% minority residents"** flips 49% of wrongly flagged Black applicants to approved (16% for white).
- **The fixes:** dropping neighbourhood data costs almost nothing (0.516 → 0.511) and needs no race data, a *less discriminatory alternative*. The fairness constraint cuts the gap to **1.21×** with 97% of the PR-AUC: **1,191 fewer** qualified Black applicants flagged, at a cost of **747 more** qualified white applicants (4.0% → 5.6%).
- **It holds where we live:** in **North Carolina**, the same recipe gives **2.09× → 1.72× → 1.42×**. The bias is smaller there, and so is the fix's effect.
- **Which lenders to look at:** our audit tool compares each lender's denials with what applicants' finances alone predict. Across 60 Georgia lenders, the median Black/white denial gap is **2.49×**; after finances, still **1.65×**, and **27 lenders (45%)** have a gap finances can't explain (NC: 24 of 42). Published **anonymised**: without credit scores, it is a reason to look closer, not proof.

**Limitations.** HMDA has no credit scores; labels are past bank decisions, so we measure **disparities, not intent**. Equal false-alarm rates mean unequal precision by group, a known trade-off. Using race in training is legally debated; fix B avoids it. Two states, two years. Details are in the Technical appendix.

## What we'd do next

Extend the audit to all 50 states with lender-level effects, and test it against lenders' real underwriting rules.

## Sources

Data: CFPB / FFIEC HMDA, https://ffiec.cfpb.gov/data-browser/ · Libraries: pandas, scikit-learn, XGBoost (https://xgboost.ai), fairlearn (https://fairlearn.org), matplotlib, Gradio (https://gradio.app) · Laws: Fair Housing Act 1968, ECOA 1974, HMDA 1975 · **AI tools:** Claude Code (Anthropic, https://claude.com/claude-code), used to help write code, review methods and draft this write-up; we reviewed and can explain every step. Explainer graphics and voice: HyperFrames, Kokoro TTS.
