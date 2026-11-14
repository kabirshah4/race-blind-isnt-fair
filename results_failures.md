# Where it fails (Georgia 2023 test data)

## 1. Misses: the bank denied them, model A didn't flag them

Model A misses **6,169 of 11,545** denials (53%).
**53%** of the misses were denied for reasons public data can't show (credit history, incomplete application, unverifiable information, other), vs 20% of the denials it caught.
**16%** of the misses were denied for **credit history**, which public data doesn't include (vs 8% of the denials it caught). The model mostly catches debt-to-income denials, which it *can* see.

| bank's main denial reason | share of misses | share of caught denials |
|---|---|---|
| debt-to-income | 17% | 64% |
| collateral (home value) | 17% | 11% |
| credit history | 16% | 8% |
| application incomplete | 16% | 4% |
| other | 11% | 4% |
| information couldn't be verified | 10% | 4% |

*Example:* income $150K, loan $375K, 18% down, debts 33% of income. Model A: 2% chance of denial (flags above 15%). The bank denied it for credit history. On paper this applicant looks safe; the reason is invisible to us.

## 2. False alarms: the bank approved them, model A flagged them

Model A flags **2,718** Black applicants the bank approved. Moving them, on paper, to a typical mostly white neighbourhood (nothing else changed) clears **1,438 (53%)** of those flags.

*Example:* income $60K, loan $185K, 18% down, debts 40% of income, in a neighbourhood that is 99% minority. The bank approved it. Model A: 17% (flagged). Same file in a mostly white neighbourhood: 10% (not flagged). Model C: 7%.

## 3. Where the fix backfires

- Model C newly flags **1,215** white applicants the bank approved and stops flagging 468 others: a net **+747** (white false alarms 4.0% → 5.6%).
- For Black applicants the bank approved, model C stops flagging **1,544** and newly flags 353: a net **1,191 fewer** (false alarms 12.0% → 6.8%).
- Model C catches **44%** of real denials vs **47%** for model A.

False-alarm rate (bank-approved applicants flagged), by group:

| group | applicants | model A | model C | vs white (A → C) |
|---|---|---|---|---|
| White | 46,735 | 4.0% | 5.6% | 1.00× → 1.00× |
| Black | 22,568 | 12.0% | 6.8% | 3.01× → 1.21× |
| Hispanic | 9,692 | 6.4% | 5.4% | 1.61× → 0.96× |
| Asian | 8,517 | 3.3% | 3.3% | 0.82× → 0.59× |
| Not reported | 15,814 | 6.9% | 5.9% | 1.72× → 1.06× |

C was tuned only for Black vs white, so check the other rows before trusting it for everyone.
