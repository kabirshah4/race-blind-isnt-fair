## Full model (race never used)

| group    |   applicants |   real denial % |   model flags % |   false alarm % |   flags vs White (x) |   false alarms vs White (x) |
|:---------|-------------:|----------------:|----------------:|----------------:|---------------------:|----------------------------:|
| White    |        49609 |             5.8 |             6.1 |             4   |                 1    |                        1    |
| Black    |        26766 |            15.7 |            18.4 |            12   |                 3.02 |                        3    |
| Hispanic |        10880 |            10.9 |            10.8 |             6.4 |                 1.77 |                        1.6  |
| Asian    |         9351 |             8.9 |             7.1 |             3.3 |                 1.16 |                        0.82 |

## Model without neighbourhood features

| group    |   applicants |   real denial % |   model flags % |   false alarm % |   flags vs White (x) |   false alarms vs White (x) |
|:---------|-------------:|----------------:|----------------:|----------------:|---------------------:|----------------------------:|
| White    |        49609 |             5.8 |             7.1 |             5   |                 1    |                        1    |
| Black    |        26766 |            15.7 |            15.6 |             9.4 |                 2.2  |                        1.88 |
| Hispanic |        10880 |            10.9 |            10.9 |             6.6 |                 1.54 |                        1.32 |
| Asian    |         9351 |             8.9 |             7.5 |             3.7 |                 1.06 |                        0.74 |

## Proxy detective

| features the detective sees   |   ROC-AUC guessing Black vs White |
|:------------------------------|----------------------------------:|
| All model features            |                             0.88  |
| Neighbourhood features only   |                             0.845 |
| Applicant/loan features only  |                             0.765 |

## Race signal of single features

| feature                           |   race signal (AUC, 0.5 = none) |
|:----------------------------------|--------------------------------:|
| tract_minority_population_percent |                           0.811 |
| ltv                               |                           0.691 |
| dti                               |                           0.67  |
| tract_to_msa_income_percentage    |                           0.65  |
| ffiec_msa_md_median_family_income |                           0.614 |
| has_coapplicant                   |                           0.603 |
| income                            |                           0.587 |
| property_value                    |                           0.579 |
| tract_owner_occupied_units        |                           0.558 |
| tract_one_to_four_family_homes    |                           0.541 |
| tract_population                  |                           0.53  |
| tract_median_age_of_housing_units |                           0.526 |
| loan_term_360                     |                           0.515 |
| loan_amount                       |                           0.501 |

## Trade-off

| model            |   PR-AUC |   Black flags vs White (x) |   Black false alarms vs White (x) |
|:-----------------|---------:|---------------------------:|----------------------------------:|
| Full             |    0.516 |                       3.02 |                              3    |
| No neighbourhood |    0.511 |                       2.2  |                              1.88 |
