# Second-state check: North Carolina

NC applications: 151,488 (2022), 124,302 (2023).

## Test 1: Georgia-trained models, unchanged, on NC 2023

| model                        |   PR-AUC |   Black vs white false alarms | 95% range   |   white false alarm % |   Black false alarm % |
|:-----------------------------|---------:|------------------------------:|:------------|----------------------:|----------------------:|
| A. Georgia model, all inputs |    0.478 |                          1.95 | 1.81–2.08   |                   4.3 |                   8.4 |
| C. Georgia fair model        |    0.466 |                          1.46 | 1.38–1.56   |                   6.3 |                   9.2 |

## Test 2: same recipe trained on NC 2022, tested on NC 2023

| model                      |   PR-AUC |   Black vs white false alarms | 95% range   |   white false alarm % |   Black false alarm % |
|:---------------------------|---------:|------------------------------:|:------------|----------------------:|----------------------:|
| A. All inputs              |    0.489 |                          2.09 | 1.95–2.22   |                   4   |                   8.3 |
| B. No neighbourhood        |    0.488 |                          1.72 | 1.61–1.84   |                   4.2 |                   7.2 |
| C. B + ratio fairness rule |    0.475 |                          1.42 | 1.31–1.54   |                   4.2 |                   5.9 |
