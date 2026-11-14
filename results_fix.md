## Models on 2023 test data

| model                             |   PR-AUC |   Black false alarms vs White | 95% range   |   Hispanic vs White |   flagged % |
|:----------------------------------|---------:|------------------------------:|:------------|--------------------:|------------:|
| A. Full model                     |    0.516 |                          3.01 | 2.86–3.17   |                1.61 |        10.3 |
| B2. Drop only % minority          |    0.513 |                          2.32 | 2.19–2.43   |                1.41 |        10.2 |
| B. No neighbourhood               |    0.511 |                          1.88 | 1.79–1.98   |                1.32 |        10.1 |
| C. Fairness-trained (ratio ≥ 0.9) |    0.501 |                          1.21 | 1.14–1.27   |                0.96 |         9.5 |

## What each fix costs, by group (2023 test data)

| model                             |   White false alarm % |   White recall % |   White precision % |   Black false alarm % |   Black recall % |   Black precision % |
|:----------------------------------|----------------------:|-----------------:|--------------------:|----------------------:|-----------------:|--------------------:|
| A. Full model                     |                   4   |             39.5 |                37.8 |                  12   |             52.4 |                44.7 |
| B2. Drop only % minority          |                   4.6 |             40   |                34.8 |                  10.7 |             50.8 |                46.9 |
| B. No neighbourhood               |                   5   |             41.6 |                34   |                   9.4 |             49.3 |                49.4 |
| C. Fairness-trained (ratio ≥ 0.9) |                   5.6 |             41.2 |                31.2 |                   6.8 |             46.5 |                56.1 |

## How the fairness rule was chosen (2022 validation data only)

| fairness rule        |   validation PR-AUC |   validation Black/White false alarms |
|:---------------------|--------------------:|--------------------------------------:|
| difference ≤ 1 point |               0.479 |                                  1.74 |
| ratio ≥ 0.9          |               0.476 |                                  1.31 |
