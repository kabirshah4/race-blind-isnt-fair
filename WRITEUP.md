# FairView · Race-Blind Isn't Fair

**Track:** Finance · **Team:** KV Gods (Kabir Shah, Vishnu Veer)
**Live demo:** https://fairview-ai.pages.dev · **Demo video (2 min):** https://www.youtube.com/watch?v=ThfnVjNuL9M · **Code:** https://github.com/kabirshah4/race-blind-isnt-fair · **Details:** [TECHNICAL.md](TECHNICAL.md)

*This project is a prototype and is not financial advice.*

**Main takeaway:** a mortgage AI that never sees race still flagged Black applicants the bank approved about **3×** as often as white ones, largely through neighbourhood data; dropping that data and adding fairness training brings it to about **1.2×** at almost no accuracy cost.

## The problem

Lenders defend AI underwriting because the model "never sees race." But where you live is tied to race, so a model can quietly rebuild redlining. Our user is **a fair-lending analyst** at a housing nonprofit or state regulator, deciding whether a lender's "race-blind" model, or the lender itself, deserves a closer look. We care because both our families immigrated from India and bought homes.

## What we built

**In:** a mortgage application and its neighbourhood. **Out:** whether a bank-style AI would flag it as "likely denied," plus our fixed AI's answer. Three tools: an **audit** of who a race-blind model wrongly flags; a **live demo** that changes only the address; and a **lender audit** ranking 60 anonymised Georgia lenders by denial gaps their applicants' finances don't explain.

## What we added

Starting point: suggested project **4.2.3** (fairness audit of a loan model). We added **264,340 real, current applications** with a time split, a **"same person, different address"** test, two **fixes** with their costs, a **second state** (North Carolina, where we live), a **lender audit** and **robustness checks** for missing credit data.

## How it works

**Data:** public HMDA records (CFPB): Georgia home-purchase applications, approved or denied; trained on 2022 (146,793), tested on 2023 (117,547). We removed three leaks, including columns filled in only after the decision. **Model:** XGBoost on 17 inputs (finances plus 7 neighbourhood statistics). We can't open a real bank's model, so we built one the way lenders do, trained on their past decisions. **Fairness measure:** false alarms, the share of bank-approved applicants the model flags.

## Results

| Model (2023 test data) | PR-AUC | Black vs white false alarms |
|---|---|---|
| Baseline: deny if debts > 43% of income | 0.342 | — |
| **A. Bank-style AI, all inputs** | **0.516** | **3.01×** (5 retrains: 2.85–2.95×) |
| B. No neighbourhood data | 0.511 | 1.88× |
| C. B + fairness training | 0.501 | **1.21×** |

- **It amplifies, not just copies.** Banks denied 15.7% of Black vs 5.8% of white applicants. Model A flags 18.4% of Black applicants (more than were denied) but 6.1% of white ones (about as many).
- **The address matters:** for 1 in 9 random real applicants, changing only the neighbourhood flips the answer. Neighbourhood data accounts for roughly half the gap.
- **Robust:** with county credit data added, still **3.11×**; resampling whole neighbourhoods, 2.78–3.28×.
- **Lenders:** 30 of 60 have a gap the finances we can see don't explain (a lead, not proof).

**Where it fails:** it misses half of real denials, often for reasons public data can't show (credit history). The fix has a cost: white false alarms rise from 4.0% to 5.6%, and the Georgia-trained fix raises false alarms for both groups in North Carolina. We measure disparities, not intent.

## What we'd do next

Add applicant-level credit scores through a lender or regulator partnership, and extend the audit to all 50 states.

## Sources

HMDA data, CFPB/FFIEC (https://ffiec.cfpb.gov/data-browser/) · Urban Institute, Debt in America (https://apps.urban.org/features/debt-interactive-map/) · pandas, scikit-learn, XGBoost (https://xgboost.ai), fairlearn (https://fairlearn.org) · **AI tool:** Claude Code (https://claude.com/claude-code), used to help write code, review methods and draft text; we reviewed and can explain every step. Video: HyperFrames.
