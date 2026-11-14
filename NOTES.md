# Project Notes: Mortgage Fairness Audit (HMDA, Georgia)

A plain-English log of everything we did and why. Both of us should be able to
explain every line of this to a judge. Newest entries are at the bottom.

**Track:** AI + Finance · **Hackathon:** Veritas AI, Nov 14–15, 2026
**Deadline:** Sunday Nov 15, 11:59 PM ET (aim to submit a first version by ~6 PM)

---

## The big idea (say this in one breath)

> We built a mortgage-approval model on real 2022–2023 Georgia data, removed
> race, sex and age, and tested whether it *still* treats groups differently
> through hidden "stand-in" features like the neighbourhood someone lives in.
> Then we tried to fix it, and we measured what the fix costs.

**Who it helps:** a first-time home buyer who got denied and wants to know why,
and a community group or regulator checking whether a lending model is fair.

**Background you need to know:**
- **Redlining:** for decades, banks refused loans in Black neighbourhoods. It
  was outlawed by the **Fair Housing Act (1968)** and the **Equal Credit
  Opportunity Act (1974)**.
- **HMDA (Home Mortgage Disclosure Act, 1975)** doesn't ban anything. It forces
  lenders to *publish* their mortgage applications so the public can check
  whether they follow those laws. That public data is what we use.
- Analogy: HMDA is like making companies publish salaries by gender. It doesn't
  make pay fair by itself, but it lets people check.

---

## Log

### Step 0: Setup (Sat ~1:00 AM)
- Made the project folder structure:
  ```
  data/raw/         original downloads (never edited)
  data/processed/   cleaned data our code makes
  src/              Python scripts
  notebooks/        Jupyter notebooks for exploring and charts
  app/              the demo app (later)
  ```
- Created a Python 3.12 environment in `.venv` (the Mac's Python 3.15 is too new
  for some ML libraries). Installed: pandas (tables), scikit-learn (ML models),
  xgboost (a stronger model), fairlearn (Microsoft's fairness toolkit), shap
  (explains model decisions), matplotlib/seaborn (charts), gradio (demo app).
- **To run anything:** `.venv/bin/python src/<script>.py`

### Step 1: Downloading the data
- Source: the **CFPB HMDA Data Browser API** (ffiec.cfpb.gov). It's free and public.
- What we asked for: **Georgia, home-purchase loans, 2022 and 2023**, only
  applications that ended in a decision (approved, or denied).
- Why Georgia: Atlanta has a well-documented history of redlining, the state
  is racially diverse, and the size is manageable on a laptop.
- What we got: 195,115 applications (2022) and 158,596 (2023), 99 columns each.
- Files: `data/raw/ga_2022.csv`, `data/raw/ga_2023.csv`

### Step 2: Checking for "cheating" columns (data leakage)
**What leakage is:** if a column is only filled in *after* the bank decides,
the model can "peek" at the answer. It then scores amazingly in testing and is
useless in real life. The judges specifically warn about results that look
too good to be true.

How we checked: for every column, we compared how often it's blank for
approved vs. denied applications. Big differences mean the column gives away
the answer.

| Column | Blank when approved | Blank when denied | What we did |
|---|---|---|---|
| interest_rate, rate_spread, loan costs, points | 3–68% | **100%** | Dropped: only exist after approval |
| denial_reason | always blank | filled | Dropped: literally the answer |
| loan_to_value_ratio | 7.5% | **34%** | Recalculated ourselves as loan ÷ property value |
| aus (automated underwriting), credit score type | — | "not applicable" mostly on denials | Dropped: they're traces of the process, not facts about the applicant |

### Step 3: Cleaning (`src/prep.py`)
**Which applications we kept** (so we compare like with like):
- First mortgage (not a second loan), the buyer will live in the home,
  a normal built-on-site house (not manufactured or multifamily),
  not a business loan, reverse mortgage or line of credit.

**The label (what the model predicts):** `approved` = 1 if the bank approved
it, 0 if it denied it.

**Features the model is allowed to use:**
| About the applicant/loan | About the neighbourhood (census tract) |
|---|---|
| income, loan amount, property value | % minority residents |
| loan-to-value (LTV) | tract income vs. area income |
| debt-to-income (DTI) | area median family income |
| loan term = 30 years? | population, # owner-occupied homes |
| has a co-applicant? | # single-family homes, median age of homes |
| loan type (conventional / FHA / VA / USDA) | |
| preapproval, conforming loan limit, how the application was submitted | |

**Kept ONLY for the fairness audit, never given to the model:**
race/ethnicity group, sex, age group.

**Cleaning details:**
- DTI is stored partly as ranges ("20%-<30%"), so we turned each range into its
  middle value (e.g. 25).
- Dropped rows with no income, no property value or no DTI (about 3%).

**Result:** 264,674 usable applications (146,953 from 2022, 117,721 from 2023).
About 10% were denied. Saved to `data/processed/hmda_ga.parquet`.

### First finding: raw denial rates, Georgia 2023
| Group | Denial rate |
|---|---|
| White | 5.8% |
| Asian | 8.9% |
| Hispanic | 10.9% |
| Black | **15.7%** |

Black applicants were denied about **2.7× as often** as white applicants.
**Careful:** this is the *raw* gap. It doesn't account for income, debt or
loan size yet. The model and the audit test how much of the gap is still there
once those are taken into account. And remember the data has **no credit score**,
so we can never fully explain the gap. We say this openly.

### Step 4: Baselines vs. models (`src/train.py`)
**Setup:** train on **2022**, test on **2023**. The model never sees the future,
just like a real bank using last year's data to judge this year's applicants.

**What we predict:** *denial* (the rare outcome, about 10%).

**Why not accuracy?** "Approve everyone" is 90% accurate and catches zero
denials. So we use:
- **Precision:** when we say "likely denied," how often are we right?
- **Recall:** of everyone who really was denied, how many did we catch?
- **PR-AUC:** one score combining both. Random guessing gets about 0.10.

**Fair comparison trick:** each model flags the same share of people (about 10%,
the 2022 denial rate), and that cut-off was picked using 2022 data only.

| Method | Flagged | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Baseline: approve everyone | 0% | 0 | 0 | 0.099 |
| Baseline: deny if DTI > 43% | 44% | 0.150 | 0.668 | 0.343 |
| Logistic regression | 10.8% | 0.443 | 0.484 | 0.485 |
| **XGBoost** | 10.1% | **0.546** | **0.558** | **0.617** |

**Takeaway:** XGBoost's PR-AUC is about **1.8× the classic DTI rule's**, and its flags
are right 55% of the time vs. 15% for the rule. The numbers are believable (not
99%), a good sign that the leakage cleanup worked.

**Models saved** in `models/` so the audit and the demo can reuse them.
Results table saved as `results_models.csv`.

### Step 5: The fairness audit (`src/audit.py`), our main findings
The model **never sees race**. We only use race afterwards, to check the
model's decisions group by group on the 2023 test data.

**Q1 + Q2: does the model treat groups differently?**

| Group | Real denial % | Model flags % | False alarm %* | Flags vs White | False alarms vs White |
|---|---|---|---|---|---|
| White | 5.8 | 6.5 | 3.6 | 1.00× | 1.00× |
| Black | 15.7 | 16.4 | 8.7 | **2.52×** | **2.42×** |
| Hispanic | 10.9 | 10.4 | 5.1 | 1.60× | 1.42× |
| Asian | 8.9 | 8.1 | 3.3 | 1.25× | 0.92× |

*False alarm % = of the people the **bank actually approved**, how many the
model wrongly flagged as "likely denied."

➡️ **Even without race, the model wrongly flags qualified Black applicants
2.4× as often as qualified white applicants.**

**Q3: how does it "know"? Finding the stand-in (proxy) features.**
We trained a "detective" model to guess Black vs. White from only the features
the lending model uses. ROC-AUC: 0.5 = can't tell at all, 1.0 = perfect.

| Detective sees… | How well it guesses race |
|---|---|
| All lending-model features | **0.880** |
| Neighbourhood features only | 0.844 |
| Applicant/loan features only | 0.765 |

Strongest single stand-in: **% minority residents in the census tract (0.81)**,
then loan-to-value (0.69) and debt-to-income (0.67).

➡️ **Removing the race column doesn't remove race.** The neighbourhood
someone lives in carries most of the same information. This is how redlining can
come back inside an algorithm without anyone writing it in.

**Q4: the fix. Retrain without the 7 neighbourhood features.**

| Model | PR-AUC (accuracy) | Black flags vs White | Black false alarms vs White |
|---|---|---|---|
| Full model | 0.617 | 2.52× | 2.42× |
| No neighbourhood features | **0.613** | **2.00×** | **1.61×** |

➡️ **Dropping neighbourhood features cut the false-alarm gap by about a third
(2.42× → 1.61×) and cost almost nothing in accuracy (0.617 → 0.613).**
The neighbourhood features barely helped predict denials, but they carried a
lot of race.

**What's still left (be honest):** a 1.6× false-alarm gap remains. Applicant
features like LTV and DTI also carry race signal, because of real wealth gaps
(e.g. smaller down payments). And with no credit score in the data, we can't
fully explain the remaining gap. Next idea: test fairlearn's fairness-constrained
training and report its cost.

### Step 6: A stronger fix, with confidence intervals (`src/fix.py`)
Two questions judges will ask: *"Is the gap real, or luck?"* and *"Can you do
better than deleting columns?"*

**Confidence intervals (bootstrap):** we re-drew the 2023 test set at random 200
times and recomputed the gap each time. The "95% range" is where the gap landed
in 95% of those re-draws. If the ranges of two models don't overlap, the
difference is real.

**Fix C, fairness-trained model:** we used Microsoft's **fairlearn** library
(method: *Exponentiated Gradient*). During **training** it looks at race and
keeps re-weighting applicants until groups have similar false-alarm rates. When it
makes **decisions** it never sees race. That matters because US law (ECOA) bans
using race in credit decisions.

| Model | PR-AUC | Black false alarms vs White | 95% range | Hispanic vs White |
|---|---|---|---|---|
| A. Full model | 0.619 | **2.62×** | 2.46–2.80 | 1.54× |
| B. No neighbourhood | 0.614 | **1.63×** | 1.54–1.75 | 1.24× |
| C. B + fairness training | 0.590 | **1.12×** | 1.05–1.19 | 0.97× |

(Small differences from Step 5's numbers come from slightly different model
settings. The story is the same.)

**A gotcha we discovered (good for Q&A):** fairlearn's default check measures
the *absolute* gap between false-alarm rates. Since those rates are small (2–5%),
a 1-point gap "passes," even though it means 1.45× more false alarms for Black
applicants. The default fairness check said the model was fine when it wasn't.
We switched to a *ratio* check (`ratio_bound=0.9`), and that's what worked.
**Lesson: how you measure fairness decides whether you find unfairness.**

### Step 7: Charts (`src/charts.py` → `figures/`)
| File | What it shows | Use it for |
|---|---|---|
| `1_denial_by_group.png` | Raw 2023 denial rates: Black 15.7% vs White 5.8% | Opening the video/talk: the problem |
| `2_neighbourhood_proxy.png` | As a neighbourhood's minority share rises, denial rates AND the share of Black applicants rise together | Explaining *why* neighbourhood is a stand-in for race |
| `3_the_fix.png` | False-alarm gap 2.62× → 1.63× → 1.12×, with 95% ranges and accuracy under each bar | The main result slide |

### Step 8: We caught a second, smaller leak, and fixed it
While building the demo, some "neighbourhood" values came out as **all zeros**.
HMDA writes a *missing address* as zeros, and denied files are about **5× more
likely** to have no address (0.53% vs 0.11%). So the model could learn "no address
means deny," which is the same kind of cheating as in Step 2.

- It was tiny: only 0.15% of rows (just 19 approved Black applicants).
- We still **dropped those rows** in `prep.py` and **re-ran everything**.
- **The numbers in Steps 4–6 above are from BEFORE this fix.** The final,
  clean numbers are below and in `results_*.csv/md`. The story didn't change;
  if anything, it got slightly stronger.

**Final clean results (2023 test data, 117,547 applications):**

| Model | PR-AUC | Black false alarms vs White | 95% range |
|---|---|---|---|
| Baseline: deny if DTI > 43% | 0.342 | — | — |
| A. Full model | 0.617 | **2.54×** | 2.40–2.70 |
| B. No neighbourhood | 0.613 | **1.56×** | 1.46–1.65 |
| C. + Fairness training | 0.597 | **1.05×** | 0.99–1.12 |

Model C's range includes 1.00, which means **no measurable gap**, while keeping
**97%** of model A's accuracy.

**The "same person, different address" test:**
- **1,131 qualified Black applicants (5.0%)** were wrongly flagged by model A
  but NOT by model C. For white applicants: 401 (0.9%). About 5× more often.
- We took those 1,131 people and changed **only their neighbourhood** to a typical
  mostly-white Georgia neighbourhood, keeping the same income, debt and loan.
  Model A then **approved 85% of them.**

### Step 9: The demo app (`app/app.py`)
**Run it:** `.venv/bin/python app/app.py`, then open http://127.0.0.1:7860

- Enter an applicant (income, loan, home price, DTI, loan type, and so on).
- Pick **where the home is**: 3 real Georgia neighbourhood types, each built
  from the median of real census tracts (under 20%, 40–60%, over 90% minority).
- Two cards show the **standard model (A)** and the **fairness-checked model (C)**
  side by side, with denial risk and a verdict.
- If denied: **"What would change it"** tries realistic fixes (lower DTI,
  smaller loan, adding a co-applicant) and shows the first one that works.

**Default demo applicant** (found by searching for a borderline case):
$100K income, $100K loan on a $125K home (20% down), 46% DTI, conventional.
| Neighbourhood | Model A | Model C |
|---|---|---|
| Mostly white | 9.6% risk → **approved** | 12.8% → approved |
| Mostly minority | 18.9% risk → **denied** | 12.8% → approved |

Same person, same finances. Only the address changed, and model A flipped.

### Main takeaway (for the pitch)
> A mortgage model that never sees race still wrongly flagged qualified Black
> applicants **2.5×** as often as qualified white applicants, because the
> neighbourhood acts as a stand-in for race. For 85% of the Black applicants it
> alone wrongly flagged, changing *only their address* flipped the decision.
> Removing neighbourhood data and adding fairness-constrained training brought
> the gap to **1.05×** (no measurable gap) while keeping **97%** of the accuracy.

---

## Limitations we will state honestly
1. **No credit score in HMDA.** It's a major factor in real lending decisions.
2. **Our labels are past bank decisions,** so the model learns the banks'
   patterns, including any unfairness in them.
3. We measure **disparities**, not **intent**. We can't prove any bank
   discriminated on purpose.
4. One state (Georgia), two years.

## Credits (for the write-up)
- Data: CFPB / FFIEC HMDA Data Browser, https://ffiec.cfpb.gov/data-browser/
- Libraries: pandas, scikit-learn, XGBoost, fairlearn, SHAP, Gradio
- AI tools: Claude Code (helped write the cleaning and modelling code; we
  reviewed and can explain every step)
- *Prototype only. This project is not financial advice.*
