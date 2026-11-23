# Day 2 technical report

## Validation question

How well does LightGBM rank 90-day default risk for later applications from customers it has never seen in training? On Day 1, 1,226 customers appeared on both sides of the random split. On Day 2, I tested the ranking under the conditions a lender would face at submission time. The dataset contained 10,000 synthetic applications, of which 789 resulted in default. [Day 1 report](DAY1_TECHNICAL_REPORT.md) · [Executed notebook](../notebooks/02_validation_tuning.ipynb)

## What leaked into the prediction?

The feature audit identified two fields that become available after submission. `days_past_due_60` describes subsequent payment delays, and `collection_calls` records later collection activity. The deliberately leaky run included both fields and computed imputation medians across all rows. Its mean average precision (AP) reached 0.9988. The clean comparison used the same random folds and 80 trees, removed both fields and learned medians within each training fold. Mean AP fell to 0.3110. The 0.6879 gap measures the combined effect of those changes. The audit retained 22 application-time predictors and found no duplicate applications. [Feature audit](../artifacts/leakage_audit.csv) · [Validation summary](../artifacts/validation_summary.csv) · [Run record](../artifacts/day2_run.json)

## Forward validation and tuning

I evaluated three consecutive six-month windows from July 2023 through December 2024. Before each window, I kept only earlier training applications whose 90-day outcomes were already known. I then removed training applications belonging to customers in that validation window. The audit found zero shared customers in all three folds. [Fold audit](../artifacts/fold_audit.csv)

I reserved 1,935 historical applications for tuning. Their outcomes were known before the first outer validation window, and none of their customers appeared in any outer validation fold. Optuna completed eight trials on three internal folds. The best trial achieved 0.3333 mean AP with a learning rate of 0.0853, 31 leaves and a minimum of 40 samples per leaf. I froze those settings before outer evaluation and kept the challenge data out of model development. [Trial history](../artifacts/optuna_results.csv) · [Selected parameters](../artifacts/best_params.json)

## Results on later applications

| Model | Mean ROC-AUC | Mean AP | AP by fold, earliest to latest |
| --- | ---: | ---: | --- |
| Fixed, 80 trees | 0.7976 | 0.3153 | 0.3395, 0.3403, 0.2661 |
| Tuned | 0.7855 | 0.3133 | 0.3356, 0.3195, 0.2849 |

Each model scored the same 5,039 eligible applications, once per application and always out of fold. The other 4,961 applications preceded the first validation window. The fixed model led by 0.0020 on mean AP, while the tuned model had higher AP in the last window. Three windows on synthetic data give limited evidence for choosing between the models in practice. [Per-fold results](../artifacts/validation_report.csv) · [Prediction coverage](../artifacts/day2_oof_coverage.csv) · [Out-of-fold predictions](../artifacts/day2_oof_predictions.csv)

![Mean ROC-AUC and average precision across the four validation setups. Error bars show the sample standard deviation across three folds.](../artifacts/day2_validation_comparison.png)

The clean random control scored 0.0044 lower in mean AP than the fixed forward model. The random folds covered all 10,000 applications, while the forward folds covered only the later 5,039. That difference in coverage prevents attributing the small gap to the split method alone. [Day 2 reflection](../artifacts/day2_reflection.json)
