# Robustness checks (Georgia, trained 2022, tested 2023)

## 1. Credit stand-ins and the model's race gap

| model                                              |   PR-AUC | Black vs white false alarms   | 95% range (by tract)   |
|:---------------------------------------------------|---------:|:------------------------------|:-----------------------|
| A. Full model (as published)                       |    0.516 | 3.01×                         | 2.78–3.28              |
| A + county credit (collections, card delinquency)  |    0.515 | 3.11×                         | 2.85–3.34              |
| B. No neighbourhood (as published)                 |    0.511 | 1.88×                         | 1.77–2.01              |
| B + county credit                                  |    0.516 | 2.53×                         | 2.36–2.70              |
| A, credit-history denials removed (train and test) |    0.51  | 2.97×                         | 2.71–3.21              |
| C. Fairness-trained (as published)                 |    0.501 | 1.21×                         | 1.13–1.29              |

County credit data from the Urban Institute's Debt in America (Aug 2023 credit-bureau records) matched 100% of applications. Ranges resample whole census tracts (2,740 tracts), so they're wider than the row-level ranges in results_fix.md.

## 2. Comparing the fix at the same flag rate

| comparison | share flagged | A: Black vs white | C: Black vs white |
|---|---|---|---|
| as published | A 10.3%, C 9.5% | 3.01× | 1.21× |
| both flag 10.3% | 10.3% | 3.01× | 1.22× |
| both flag 9.5% | 9.5% | 2.85× | 1.21× |

Cut-offs here are set on 2023 data only to equalise flag rates for this comparison.

## 3. Stronger baselines (same 17 inputs)

| model                                                             |   PR-AUC | Black vs white false alarms   |
|:------------------------------------------------------------------|---------:|:------------------------------|
| Logistic regression, tuned (log money, DTI as categories, scaled) |    0.439 | 4.21×                         |
| scikit-learn boosted trees (HistGradientBoosting)                 |    0.515 | 2.64×                         |
| Our XGBoost (model A)                                             |    0.516 | 3.01×                         |
| Classic rule: deny if debts > 43% of income                       |    0.342 | n/a                           |

## 4. Lender audit with credit stand-ins

| lender audit                                  |   lenders | median finance-adjusted gap   | lenders whose 95% range is above 1   |
|:----------------------------------------------|----------:|:------------------------------|:-------------------------------------|
| finances only (as published)                  |        60 | 1.65×                         | 32 (53%)                             |
| finances + county credit                      |        60 | 1.50×                         | 26 (43%)                             |
| finances only, credit-history denials removed |        50 | 1.59×                         | 22 (44%)                             |

Simple 95% ranges (1,000 draws), no multiple-testing correction, so compare rows with each other; the published headline (30 of 60) uses the stricter correction in results_lenders_strict.md.
