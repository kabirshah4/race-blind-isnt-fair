# FairView · Race-Blind Isn't Fair: Auditing a Mortgage Model for Neighbourhood Proxies

**Track:** AI + Finance · **Team:** KV Gods (Kabir Shah, Vishnu Veer)
**Starting point:** Extends suggested project 4.2.3 (fairness audit of a loan model) to real 2022–2023 US mortgage data
**Live demo:** https://fairview-ai.pages.dev (backup: https://kabirshah4.github.io/race-blind-isnt-fair/) · **Code:** https://github.com/kabirshah4/race-blind-isnt-fair

*This project is a prototype and is not financial advice.*

## Why we care

Both of our families immigrated from India, and our parents went through buying a home. We wanted to know whether something besides their finances, like race, could quietly affect that decision. In Georgia's 2023 data, banks denied Asian Indian applicants 10.8% of the time vs. 5.8% for white applicants (before accounting for finances). Our race-blind AI didn't over-flag bank-approved Asian Indian applicants (0.49× the white rate, `results_asian_indian.md`), but it wrongly flagged bank-approved **Black** applicants about **3×** as often. A bias you can't see can hit any family, so it has to be checked.

## The problem

Lenders use machine learning to approve mortgages and often defend it because the model "never sees race." But where you live is closely tied to race, so a model can rebuild redlining (outlawed in 1968) without anyone writing it in. We can't open a real bank's model, so we **built one the way lenders do** (trained on their past decisions) and audited that; "typical bank AI" in our demo means this model. Our user is **a fair-lending analyst at a housing nonprofit or state regulator** deciding whether a lender's "race-blind" model, or the lender itself, deserves a closer look.

## What we built (beyond 4.2.3)

A denial-prediction model on **264,340 real Georgia home-purchase applications** (public HMDA data; trained on 2022, tested on 117,547 from 2023) that never sees race, sex or age; a **fairness audit** of its mistakes by group; a **"proxy detective"** that shows which inputs carry race; two **fixes** with their costs, chosen on 2022 validation data only; a **second-state check in North Carolina**, where we live; a **lender-audit tool**; and a **live demo** where you change only an applicant's address.

## How it works

**Model:** XGBoost on 17 inputs (income, loan, home value, loan-to-value, debt-to-income, loan type, plus 7 neighbourhood statistics). **Leaks removed:** post-decision columns, rows with missing addresses, and two columns that gave the answer away. **Fairness measure:** the *false-alarm rate*, the share of applicants the bank approved that the model flags as "likely denied," compared between Black and white applicants.

## Results (2023 test data)

| Model | PR-AUC | Black vs. white false alarms (95% CI) |
|---|---|---|
| Baseline: deny if debt-to-income > 43% | 0.342 | — |
| **A. XGBoost, all inputs** | **0.516** | **3.01×** (2.86–3.17) |
| B. Drop all neighbourhood inputs | 0.511 | 1.88× (1.79–1.98) |
| C. B + fairness constraint (race used only in training) | 0.501 | **1.21×** (1.14–1.27) |

Retrained 5 times (`results_seeds.md`): A 2.85–2.95×, B 1.83–1.89×, C 1.12–1.28×. Not a lucky run.

- **The model works:** it predicts bank denials with 1.5× the PR-AUC of the classic "debts over 43%" rule.
- **Race-blind, yet unfair:** model A wrongly flags **12.0%** of bank-approved Black applicants vs. **4.0%** of white: **2,718 Black Georgians** the bank approved, in one year (about 1,800 more than at the white rate). Part of this is how bias gets copied: banks denied **15.7%** of Black applicants vs **5.8%** of white, and a model trained on those decisions learns that pattern, including for people the bank approved.
- **Neighbourhood is a major proxy:** a second model guesses race from A's inputs with ROC-AUC **0.88** (0.5 = coin flip). Neighbourhood inputs explain about **56%** of the excess gap. Changing **only "% minority residents"** clears 49% of the Black false alarms (16% for white); for **11%** of 500 random real applicants in our demo, the address alone flips A's answer.
- **The fixes:** dropping neighbourhood data costs almost nothing (0.516 → 0.511) and needs no race data, a *less discriminatory alternative*. The fairness constraint cuts the gap to **1.21×** with 97% of the PR-AUC: **1,191 fewer** bank-approved Black applicants flagged, at a cost of **747 more** white ones (4.0% → 5.6%). C also flags slightly fewer people overall (9.5% vs 10.3%), which accounts for part of the drop.
- **Where we live:** the Georgia model, run unchanged on North Carolina, is still unfair (**1.95×**). Retrained in NC: **2.09× → 1.72× → 1.42×**; the bias is smaller, and the fix helps less.
- **Which lenders to look at:** our audit tool compares each lender's denials with what applicants' finances alone predict. Median gap across 60 Georgia lenders: **2.49×**, and **1.65×** after finances; **30 of 60 (50%)** have a gap the finances we can see don't explain, even after correcting for testing 60 lenders at once (`results_lenders_strict.md`; NC: 24 of 42). Published **anonymised**: without credit scores, it is a lead, not proof.

## Where it fails

Real 2023 cases (rounded; all in `results_failures.md`):

- **It misses half of real denials (53%)**, and **53%** of those misses were denied for reasons public data can't show (credit history, incomplete or unverifiable applications). *Example:* $150K income, 18% down, debts 33% of income: our model says 2% risk; the bank denied it for credit history.
- **The address alone flags people.** *Example:* $60K income, 18% down, in a 99%-minority neighbourhood. The bank approved it; model A flags it (17%). The same file in a mostly white area: 10%, not flagged. Swapping only the neighbourhood clears **53%** of the 2,718 Black false alarms.
- **The fix has costs.** Model C newly flags **1,215** white applicants the bank approved (net +747) and catches 44% of denials vs 47%. It was tuned for Black vs white only; it also lowers Hispanic false alarms (1.61× → 0.96× the white rate).
- We measure **disparities, not intent**. Fix C uses race in training (legally debated under ECOA); fix B needs none, so it is the safer choice for a real lender. Two states, two years.

## What we'd do next

Add credit-score data (e.g. through a lender or regulator partnership), extend the audit to all 50 states, and test it against lenders' real underwriting rules.

## Sources

Data: CFPB / FFIEC HMDA, https://ffiec.cfpb.gov/data-browser/ · Libraries: pandas, scikit-learn, XGBoost (https://xgboost.ai), fairlearn (https://fairlearn.org), matplotlib, Gradio (https://gradio.app) · Laws: Fair Housing Act 1968, ECOA 1974, HMDA 1975 · **AI tools:** Claude Code (Anthropic, https://claude.com/claude-code), used to help write code, review methods and draft this write-up; we reviewed and can explain every step. Explainer graphics and voice: HyperFrames, Kokoro TTS.
