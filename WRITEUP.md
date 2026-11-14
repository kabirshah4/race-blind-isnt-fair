# FairView · Race-Blind Isn't Fair: Auditing a Mortgage Model for Neighbourhood Proxies

**Track:** Finance · **Team:** KV Gods (Kabir Shah, Vishnu Veer)
**Live demo:** https://fairview-ai.pages.dev (lender audit: https://fairview-ai.pages.dev/lenders.html) · **Code:** https://github.com/kabirshah4/race-blind-isnt-fair

*This project is a prototype and is not financial advice.*

## The problem

Lenders use machine learning to approve mortgages and often defend it because the model "never sees race." But where you live is closely tied to race, so a model can rebuild redlining (outlawed in 1968) without anyone writing it in. Our user is **a fair-lending analyst at a housing nonprofit or state regulator** deciding whether a lender's "race-blind" model, or the lender itself, deserves a closer look. We care because both our families immigrated from India and bought homes: a bias you can't see can hit any family. (In Georgia 2023, banks denied 10.8% of Asian Indian applicants vs 5.8% of white; our model didn't over-flag them, 0.49× the white rate, `results_asian_indian.md`.)

## What we built

**In:** a mortgage application (income, loan, home price, debts, loan type) and its neighbourhood. **Out:** whether a race-blind AI would flag it as "likely denied," from a typical model and from our fixed one. An analyst uses it three ways: the **audit** shows who a race-blind model wrongly flags; the **live demo** changes only an applicant's address to show the proxy at work; the **lender audit** ranks real (anonymised) lenders by denial gaps that applicants' finances don't explain.

We can't open a real bank's model, so we **built one the way lenders do** (trained on their past decisions) and audited that; "typical bank AI" in our demo means this model.

## What we added

Starting point: suggested project **4.2.3** (fairness audit of a loan model). We added real, current data (**264,340** Georgia applications instead of a toy dataset) with a time split; a **"proxy detective"** showing which inputs carry race; a **"same person, different address"** test; two **fixes** with their costs, chosen on 2022 validation data only; a **second-state check in North Carolina**, where we live; a **lender-audit tool** and page; and **robustness checks** against missing credit data.

## How it works

**Data:** public HMDA records (CFPB/FFIEC): home-purchase, first-lien, owner-occupied applications that were approved or denied. Trained on 146,793 from 2022, tested on 117,547 from 2023. **Leaks removed:** post-decision columns, rows with missing addresses, and two columns that gave the answer away. **Model:** XGBoost on 17 inputs (applicant finances plus 7 neighbourhood statistics); it beat a tuned logistic regression (0.439) and matched scikit-learn's boosted trees (0.515). **Fairness measure:** the *false-alarm rate*, the share of applicants the bank approved that the model flags, Black vs white.

## Results (2023 held-out test data)

| Model | PR-AUC | Black vs. white false alarms (95% CI) |
|---|---|---|
| Baseline: deny if debt-to-income > 43% | 0.342 | — |
| **A. XGBoost, all inputs** | **0.516** | **3.01×** (2.86–3.17) |
| B. Drop all neighbourhood inputs | 0.511 | 1.88× (1.79–1.98) |
| C. B + fairness constraint (race used only in training) | 0.501 | **1.21×** (1.14–1.27) |

- **Race-blind, yet unfair:** model A flags **12.0%** of bank-approved Black applicants vs **4.0%** of white: **2,718** people in one year, about 1,800 more than at the white rate. Banks denied **15.7%** of Black applicants vs **5.8%** of white, and a model trained on those decisions copies the pattern, including for people the bank approved.
- **Neighbourhood is the main proxy:** a second model guesses race from A's inputs with ROC-AUC **0.88**; neighbourhood explains about **56%** of the excess gap. For **11%** of 500 random real applicants, the address alone flips A's answer.
- **The fixes:** dropping neighbourhood data costs almost nothing (0.516 → 0.511) and needs no race data. The fairness constraint cuts the gap to **1.21×** with 97% of the PR-AUC: **1,191 fewer** Black false alarms, at a cost of **747 more** white ones (4.0% → 5.6%).
- **Where we live:** retrained in North Carolina, **2.09× → 1.72× → 1.42×**; smaller bias, and the fix helps less.
- **Lenders:** median Black/white denial gap **2.49×**, **1.65×** after finances; **30 of 60** Georgia lenders have a gap the finances we can see don't explain, after correcting for testing 60 at once (NC: 24 of 42). Anonymised; a lead, not proof.

**Robustness** (`results_robustness.md`, `results_seeds.md`, `results_robustness_c.md`): the gap survives every check. Five retrains: A 2.85–2.95×. **County credit data** (Urban Institute, collections and card delinquency) added to A: **3.11×**. Applicants denied for credit history removed: **2.97×**. Ranges resampling whole neighbourhoods: 2.78–3.28×. A different boosted-tree model is also unfair (2.64×). C at the same flag rate as A: 1.22×; retrained on all of 2022: 0.95×. With county credit in the lender audit, gaps shrink (1.65× → 1.50×) but 26 of 60 lenders still stand out.

**Where it fails** (`results_failures.md`):

- **It misses half of real denials (53%)**; over half of those were denied for reasons public data can't show (credit history, incomplete applications). *Example:* $150K income, 18% down, debts 33%: our model says 2% risk; the bank denied it for credit history.
- **The address alone flags people.** *Example:* $60K income, 18% down, 99%-minority neighbourhood, approved by the bank: model A flags it (17%); in a mostly white area, 10%, not flagged.
- **The verdict depends on the fairness measure.** By precision, model A looks *better* for Black applicants (44.7% vs 37.8%); both can't be equal at once. County credit data also carries race: it pushes model B's gap from 1.88× to 2.53×.
- We measure **disparities, not intent**. Fix C uses race in training (legally debated under ECOA); fix B needs none, so it is the safer choice for a real lender.

## What we'd do next

Add applicant-level credit scores (through a lender or regulator partnership) and extend the audit to all 50 states.

## Sources

Data: CFPB / FFIEC HMDA, https://ffiec.cfpb.gov/data-browser/ · Urban Institute, Debt in America (county credit data, Aug 2023), https://apps.urban.org/features/debt-interactive-map/ · Libraries: pandas, scikit-learn, XGBoost (https://xgboost.ai), fairlearn (https://fairlearn.org), matplotlib, Gradio (https://gradio.app) · Laws: Fair Housing Act 1968, ECOA 1974, HMDA 1975 · **AI tools:** Claude Code (Anthropic, https://claude.com/claude-code), used to help write code, review methods and draft this write-up; we reviewed and can explain every step. Video: HyperFrames.
