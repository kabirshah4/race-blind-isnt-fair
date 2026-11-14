# Training randomness: 5 seeds per model (2023 test data)

| model                             | PR-AUC (mean ± sd)   | Black vs. white false alarms (mean ± sd)   | lowest–highest of 5 runs   |
|:----------------------------------|:---------------------|:-------------------------------------------|:---------------------------|
| A. Full model                     | 0.515 ± 0.000        | 2.91× ± 0.04                               | 2.85–2.95×                 |
| B. No neighbourhood               | 0.511 ± 0.000        | 1.86× ± 0.02                               | 1.83–1.89×                 |
| C. Fairness-trained (ratio ≥ 0.9) | 0.497 ± 0.003        | 1.17× ± 0.07                               | 1.12–1.28×                 |

Each model was retrained 5 times (seeds 0–4) with everything else identical to fix.py; model C also draws a different 60,000-row training sample each time. This is the spread our test-set confidence intervals leave out.

## Every run

| model                             |   seed |   PR-AUC |   ratio |
|:----------------------------------|-------:|---------:|--------:|
| A. Full model                     |      0 |    0.515 |   2.942 |
| B. No neighbourhood               |      0 |    0.511 |   1.865 |
| C. Fairness-trained (ratio ≥ 0.9) |      0 |    0.496 |   1.157 |
| A. Full model                     |      1 |    0.515 |   2.935 |
| B. No neighbourhood               |      1 |    0.511 |   1.834 |
| C. Fairness-trained (ratio ≥ 0.9) |      1 |    0.494 |   1.121 |
| A. Full model                     |      2 |    0.515 |   2.887 |
| B. No neighbourhood               |      2 |    0.511 |   1.889 |
| C. Fairness-trained (ratio ≥ 0.9) |      2 |    0.501 |   1.281 |
| A. Full model                     |      3 |    0.515 |   2.85  |
| B. No neighbourhood               |      3 |    0.511 |   1.859 |
| C. Fairness-trained (ratio ≥ 0.9) |      3 |    0.498 |   1.153 |
| A. Full model                     |      4 |    0.515 |   2.945 |
| B. No neighbourhood               |      4 |    0.511 |   1.873 |
| C. Fairness-trained (ratio ≥ 0.9) |      4 |    0.495 |   1.121 |
