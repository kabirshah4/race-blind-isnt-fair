# Placebo checks (Georgia, trained 2022, tested 2023)

| check | Black vs white false alarms |
|---|---|
| A. Bank-style AI, real data | 3.01× |
| Placebo 1: race labels shuffled (1,000 times) | 1.00× (95% range 0.94–1.06; highest 1.09×) |
| Placebo 2: every applicant given a random real neighbourhood, retrained (5 times) | 1.88× (range 1.84–1.90) |
| B. No neighbourhood data, for comparison | 1.88× |

**Placebo 1:** with race labels handed out at random, the gap disappears (about 1.0×), and none of 1,000 shuffles came close to 3.0×. The 3× is not a side effect of how we measure.

**Placebo 2:** when the neighbourhood columns describe someone else's neighbourhood, the gap falls to about the level of the model with no neighbourhood data. So the extra gap in model A comes from where applicants actually live, not from the model simply having more columns.

The gap that remains in both B and placebo 2 (about 1.9×) comes from applicants' own finances (income, loan size, debt), which also differ by race on average.
