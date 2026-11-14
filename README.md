# Race-Blind Isn't Fair: Auditing a Mortgage-Approval Model for Hidden Redlining

**Track:** AI + Finance
**Team:** Kabir Shah, Vishnu Veer
**Starting point:** Extends suggested project 4.2.3 (fairness audit of a loan model), using real 2022–2023 US mortgage data instead of German Credit

*This project is a prototype and is not financial advice.*

## The problem

Lenders increasingly use machine learning to decide who gets a mortgage, and a common defence is that the model "never sees race." Redlining (refusing loans because of who lives in a neighbourhood) was outlawed in 1968, but a model can rebuild it without anyone writing it in, because where you live is closely tied to your race. Our users are **fair-lending auditors and community groups** checking a lender's model, and **first-time buyers** who want to understand how a decision like theirs is made.

## What we built

A mortgage-denial model trained on **264,340 real Georgia home-purchase applications** (public HMDA data) that is never given race, sex or age; a **fairness audit** of its false alarms by group; a **"proxy detective"** that shows which inputs carry race; **fixes** with their costs; and a **demo app** where you change only an applicant's neighbourhood and compare the original model with the fixed one.

## What we added (beyond 4.2.3)

Real, current data with a **train-on-2022, test-on-2023** split; **three data leaks** we found and removed; the proxy detective; a "same person, different address" test; a **fairness-constrained model** whose fairness rule was chosen on **2022 validation data only**; and **bootstrap confidence intervals**.

## How it works

**Data.** HMDA requires lenders to publish every application. We used Georgia home purchases (first lien, owner-occupied, site-built), approved or denied: 146,793 in 2022 and 117,547 in 2023, 9.8% denied. Race, sex and age are used only for the audit.

**Leakage.** We compared each column for approved vs. denied files. (1) Interest rate, loan costs and denial reasons only exist after a decision, so we dropped them, and we recalculated loan-to-value because the reported one was missing far more often for denials. (2) Missing addresses were 5× more common among denials, so we removed those 0.15% of rows. (3) An independent review found two more: `preapproval` (denied pre-approval requests are filed under a code we never download, so "asked for pre-approval" meant 0% denied) and `initially_payable_to_institution` (code 3 meant 83% denied). We removed both and re-ran everything. **All numbers below are after all three fixes.**

**Model.** XGBoost on 17 inputs (income, loan, home value, loan-to-value, debt-to-income, loan term and type, co-applicant, plus 7 census-tract statistics). Denials are rare, so we use PR-AUC, precision and recall. Every model flags the same share of applicants (the 2022 denial rate), with the cut-off set on 2022 data.

**Fairness.** For each group, the **false-alarm rate**: the share of applicants the *bank approved* that the model flags as "likely denied." We compare Black with white applicants as a ratio, with 95% bootstrap intervals over the 2023 test set. Fix C uses fairlearn's Exponentiated Gradient: race is used **only during training**. To make one decision we average its sub-models' scores and apply a cut-off, so its fairness is measured on held-out data rather than guaranteed.

## Results (2023 test data, 117,547 applications)

| Model | PR-AUC | Black vs. white false alarms (95% CI) |
|---|---|---|
| Baseline: deny if debt-to-income > 43% | 0.342 | — |
| **A. XGBoost, all inputs** | **0.516** | **3.01×** (2.86–3.17) |
| B2. Drop only "% minority residents" | 0.513 | 2.32× (2.19–2.43) |
| B. Drop all 7 neighbourhood inputs | 0.511 | 1.88× (1.79–1.98) |
| C. B + fairness constraint | 0.501 | **1.21×** (1.14–1.27) |

- **The model beats the rule:** PR-AUC 0.516 vs. 0.342 (1.5×).
- **Race-blind, yet unfair:** model A wrongly flags **12.0%** of qualified Black applicants vs. **4.0%** of qualified white applicants.
- **Neighbourhood is the proxy:** the detective guesses Black vs. white from model A's inputs with ROC-AUC 0.88 (0.5 = coin flip); the neighbourhood inputs alone reach 0.845. Removing only "% minority" leaves 2.32×, so the proxy is spread across several neighbourhood inputs.
- **Same person, different address:** for the 2,718 qualified Black applicants model A wrongly flags, changing **only their neighbourhood** to a typical mostly-white one makes A approve **53%** of them (22% for wrongly flagged white applicants).
- **The fix and its cost:** C cuts the gap from 3.01× to 1.21× while keeping 97% of A's PR-AUC. Black false alarms fall from 12.0% to 6.8%, but white false alarms rise from 4.0% to 5.6%, and Black recall falls from 52% to 47%.
- **Surprise:** on 2022 validation data, fairlearn's default *difference* rule ("false-alarm rates within 1 point") still left a 1.74× gap, because the rates are small. A *ratio* rule left 1.31×, so we chose it. **How you measure fairness decides whether you find unfairness.**

**Limitations.** HMDA has no credit scores, so some gaps may reflect information we can't see. Our labels are past bank decisions, so "qualified" means "the bank approved them," and the model learns the banks' patterns; we measure **disparities, not intent**. Equalising false alarms changes precision by group (56% vs. 31%), a known trade-off between fairness definitions. Using race in training is legally debated in US lending; fix B needs no race at all. One state, two years.

## What we'd do next

Run the audit across all states and years with lender-level effects, and test it on lenders' actual underwriting rules.

## Sources

- Data: CFPB / FFIEC HMDA Data Browser, https://ffiec.cfpb.gov/data-browser/
- pandas https://pandas.pydata.org · scikit-learn https://scikit-learn.org · XGBoost https://xgboost.ai · fairlearn https://fairlearn.org · matplotlib https://matplotlib.org · Gradio https://gradio.app
- Laws: Fair Housing Act (1968), Equal Credit Opportunity Act (1974), Home Mortgage Disclosure Act (1975)
- **AI tools:** Claude Code (Anthropic), https://claude.com/claude-code, used to help write the code, review the methods, and draft this write-up. We reviewed every step and can explain all of it. Explainer graphics and voice: HyperFrames and Kokoro text-to-speech.

---

## How to run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download.py        # 1. download the public HMDA data (Georgia 2022–2023)
python src/prep.py            # 2. clean it, remove leaky columns -> data/processed/
python src/train.py           # 3. baselines vs. models -> results_models.csv
python src/audit.py           # 4. fairness audit + proxy detective -> results_audit.md
python src/fix.py             # 5. fixes, validation choice of fairness rule, 95% CIs, per-group costs -> results_fix.md
python src/counterfactual.py  # 6. "same person, different address" -> results_counterfactual.md
python src/charts.py          # 7. charts -> figures/
python app/app.py             # 8. the demo app -> http://127.0.0.1:7860
```

Every script starts with a plain-English explanation of what it does and why.
**[NOTES.md](NOTES.md)** is our full step-by-step project log, including the leaks we caught and the review that found the third one.

| Folder | What's in it |
|---|---|
| `src/` | The pipeline: download → prep → train → audit → fix → counterfactual → charts |
| `app/` | The Gradio demo |
| `figures/` | Charts used in the write-up |
| `models/` | Trained models (so the demo runs without retraining) |
| `results_*.csv/md` | Raw result tables behind every number above |

![The fix](figures/3_the_fix.png)
