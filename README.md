# Tamweel Lite

Tamweel Lite studies how well machine learning models can rank loan applications by the risk of default within 90 days. The project uses synthetic applications and information available at the time of submission.

Author: Aljuhara Alshagri

Program: Advanced Machine Learning (SDA-DSC-211), SDAIA Academy

| Notebook | Contents | Open in Colab |
| --- | --- | --- |
| [Tamweel capstone](notebooks/Tamweel_aljuharaalshagri.ipynb) | Day 1 baseline, Day 2 validation, Day 3 review policy, and Day 4 explanations and calibration | [Run Days 1-4](https://colab.research.google.com/github/aljuharaalshagri/Tamweel_SDAIA_AdvanceML/blob/main/notebooks/Tamweel_aljuharaalshagri.ipynb) |
| [Day 3 decision thresholds](notebooks/03_decision_thresholds.ipynb) | The self-contained Day 3 analysis and completed reflections | [Run Day 3](https://colab.research.google.com/github/aljuharaalshagri/Tamweel_SDAIA_AdvanceML/blob/main/notebooks/03_decision_thresholds.ipynb) |

## Project overview

Day 1 asks whether gradient boosting improves on a Logistic Regression baseline. I trained XGBoost and LightGBM alongside the baseline, then compared all three on the same 2,000 held-out applications. Each application has a score from every model, so the comparison uses the same outcomes throughout. [Held-out predictions](artifacts/artifacts_Day1/day1_comparison_predictions.csv)

Day 2 tests whether LightGBM can rank default risk for later applications from customers excluded from training. I audited which fields were available at submission, set aside historical data for tuning, and compared random validation with three forward validation folds.

Day 3 turns the risk scores into a review rule. I compared three training strategies, then used the predeclared weighted strategy to select a threshold under a loss of 10 units per missed default and 1 unit per false positive. At most 12% of applications can be flagged in each validation period. [Day 3 operating points](artifacts/artifacts_Day3/day3_operating_points.csv)

Day 4 examines a weighted LightGBM model through permutation importance, global and local SHAP explanations, sigmoid calibration, customer-cluster bootstrap intervals, and a review-capacity audit. The completed run improves probability agreement while leaving ranking unchanged. Its expanded review queue exceeds capacity in both evaluation quarters. [Day 4 saved evidence](artifacts/artifacts_Day4/day4_saved_output_summary.json)

## Day 1 results

| Model | ROC-AUC | Average precision | Training time (s) |
| --- | ---: | ---: | ---: |
| Logistic Regression | **0.821** | 0.326 | 0.030 |
| XGBoost | 0.812 | **0.334** | 0.178 |
| LightGBM | 0.814 | 0.325 | 0.171 |

Logistic Regression led on ROC-AUC. XGBoost led on average precision, which matters here because only 158 of the 2,000 applications resulted in default (7.9%). Training times in this table come from the archived Day 1 export and include tree-count selection and refitting for the boosting models. The latest combined notebook contains another execution with the same rounded ranking metrics and different timings. [Full comparison table](artifacts/artifacts_Day1/day1_model_comparison.csv)

![ROC and precision-recall curves for Logistic Regression, XGBoost and LightGBM on the held-out applications.](artifacts/artifacts_Day1/day1_roc_pr.png)

*The dotted horizontal line in the precision-recall panel marks the 7.9% default rate. XGBoost had the highest average precision (0.3338).*

## Day 1 training pipeline

1. Select the 22 application-time predictors and reserve 2,000 of the 10,000 applications for comparison using a stratified random split.
2. Fit Logistic Regression on the 8,000 development applications after median imputation and feature scaling.
3. Within the development set, fit the boosting models on 6,000 applications and use the other 2,000 to select the tree count by log loss. XGBoost selected 61 trees and LightGBM selected 41.
4. Refit each boosting model on all 8,000 development applications at its selected tree count. Score all three models on the held-out applications and export the predictions, metrics and plots.

![Training and early-stopping log loss by boosting round for XGBoost and LightGBM.](artifacts/artifacts_Day1/day1_learning_curves.png)

*The vertical markers show the selected tree counts. Log loss on the early-stopping set was lowest at 61 trees for XGBoost and 41 for LightGBM.* [Day 1 technical report](reports/DAY1_TECHNICAL_REPORT.md)

## Day 2 validation design and results

Of the 10,000 synthetic applications, 7.89% resulted in default within 90 days. The audit retained 22 application-time predictors and found no duplicate applications. I tested four validation setups to see how the information available at training time affected the results.

![Training and validation application counts for the three forward folds starting in July 2023, January 2024 and July 2024.](artifacts/artifacts_Day2/day2_fold_sizes.png)

*The eligible training set grew from 3,223 to 5,731 applications as more 90-day outcomes became available. Each fold had zero shared customers between training and validation.* [Fold audit](artifacts/artifacts_Day2/fold_audit.csv)

| Validation setup | Applications evaluated | Mean ROC-AUC | Mean average precision (AP) |
| --- | ---: | ---: | ---: |
| Random folds, post-application fields and all-row imputation | 10,000 | 0.9999 | 0.9988 |
| Random folds, application-time predictors and training-fold imputation | 10,000 | 0.8010 | 0.3110 |
| Forward folds, separate customers, fixed 80 trees | 5,039 | 0.7976 | 0.3153 |
| Forward folds, separate customers, tuned model | 5,039 | 0.7855 | 0.3133 |

Each score is the mean across three folds. On identical random folds, mean AP fell by **0.6879** when I removed the two post-application fields and fitted the median imputer only on the training rows. The experiment changed both conditions together. Clean random validation scored **0.0044 lower** than the fixed model on the forward folds. That second comparison covers different applications, so the gap cannot be attributed to the split method alone. Both gaps use the unrounded scores. [Validation summary](artifacts/artifacts_Day2/validation_summary.csv) · [Per-fold metrics](artifacts/artifacts_Day2/validation_report.csv)

![Mean ROC-AUC and average precision for the two random controls and the fixed and tuned forward-validation models.](artifacts/artifacts_Day2/day2_validation_comparison.png)

*The horizontal bars show one sample standard deviation across three folds. The fixed model had slightly higher mean AP (0.3153), while the tuned model varied less across the folds.*

For tuning, I reserved 1,935 earlier applications whose customers were absent from every outer validation fold. All eight Optuna trials ran on this pool. The best trial reached 0.3333 mean AP during tuning. On the 5,039 later applications, the fixed 80-tree model scored slightly higher than the tuned model (0.3153 versus 0.3133 AP). Both models produced one out-of-fold prediction per eligible application. The other 4,961 applications belonged to the initial historical period. [Trial history](artifacts/artifacts_Day2/optuna_results.csv) · [Out-of-fold coverage](artifacts/artifacts_Day2/day2_oof_coverage.csv) · [Day 2 technical report](reports/DAY2_TECHNICAL_REPORT.md)

![Mean average precision for each of the eight Optuna trials and the best score reached so far on the historical tuning pool.](artifacts/artifacts_Day2/day2_search.png)

*The second trial reached the highest mean AP (0.3333) on the three tuning folds. The dashed line tracks that best score through the remaining trials.*

## Day 3 review policy and results

The Day 3 run produced out-of-fold (OOF) scores for 5,039 later applications, including 384 defaults (7.62%). Each validation fold uses a model trained on earlier applications with mature 90-day outcomes and no customers shared with that fold's validation set. Class weighting and oversampling affect training only; the validation applications retain their observed default rate. [Fold audit](artifacts/artifacts_Day3/day3_fold_audit.csv) · [Prediction coverage](artifacts/artifacts_Day3/day3_oof_coverage.csv)

| Training strategy | Pooled ROC-AUC | Pooled AP | Recall at 0.5 | Flags at 0.5 | Within every period's capacity at 0.5? |
| --- | ---: | ---: | ---: | ---: | --- |
| Unweighted | 0.7979 | 0.3091 | 12.50% | 87 | Yes |
| Weighted | 0.7969 | 0.3100 | 57.81% | 1,005 | No |
| Oversampled | 0.7979 | 0.3159 | 59.90% | 1,047 | No |

The weighted strategy was declared before comparison for the threshold exercise. The threshold analysis follows that predeclared choice. Oversampling had the highest pooled AP in this run. These pooled scores use the same 5,039 applications and differ from Day 2's means across folds. [Strategy comparison](artifacts/artifacts_Day3/day3_strategy_comparison.csv)

![ROC and precision-recall curves for the unweighted, weighted and oversampled Day 3 models.](artifacts/artifacts_Day3/day3_roc_pr.png)

The selected rule flags an application when its weighted score is **greater than or equal to 0.6583471436014694**. It minimizes the observed loss among the tested thresholds that satisfy the capacity limit in every period. The JSON keeps the full-precision threshold and the comparison operator. [Exact decision rule](artifacts/artifacts_Day3/day3_threshold.json)

| Weighted-model rule | Threshold | Flagged applications | Recall | Precision | Loss units | Within capacity? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 0.5000 | 1,005 (19.94%) | 57.81% | 22.09% | 2,403 | No |
| Minimum loss without capacity limit | 0.4486 | 1,141 (22.64%) | 64.06% | 21.56% | 2,275 | No |
| Minimum loss within every period's capacity | 0.6583 | 526 (10.44%) | 40.89% | 29.85% | 2,639 | Yes |

Thresholds in the table are rounded for display. The selected rule catches 157 defaults, misses 227, and incorrectly flags 369 non-defaulting applications. Its loss is `10 * 227 + 369 = 2,639` teaching units. Meeting capacity raises loss by 236 units relative to the default threshold and by 364 units relative to the unconstrained minimum. [Operating points](artifacts/artifacts_Day3/day3_operating_points.csv) · [Threshold sweep](artifacts/artifacts_Day3/day3_threshold_sweep.csv)

![Observed loss and flagged fraction across the weighted model's threshold sweep, with the selected feasible threshold marked.](artifacts/artifacts_Day3/day3_cost_curve.png)

| Validation period | Applications | Review capacity, rounded down | Flagged | Flagged fraction |
| --- | ---: | ---: | ---: | ---: |
| July-December 2023 | 1,632 | 195 | 137 | 8.39% |
| January-June 2024 | 1,674 | 200 | 183 | 10.93% |
| July-December 2024 | 1,733 | 207 | 206 | 11.89% |

Every period stays within its own limit. A pooled flagged fraction below 12% would not, by itself, establish this. [Period audit](artifacts/artifacts_Day3/day3_period_audit.csv)

![Application review counts against each period's capacity and regional false-positive rates at the shared threshold.](artifacts/artifacts_Day3/day3_capacity_regions.png)

Regional false-positive rates range from 7.64% in other to 8.29% in western, a gap of 0.6484 percentage points. The comparison has no confidence intervals and does not establish fairness. Regional recall also varies, from 34.62% to 48.15%. The threshold was selected using these OOF development outcomes, so its loss needs confirmation on untouched later applications. All costs are teaching units and all applications are synthetic. [Regional audit](artifacts/artifacts_Day3/day3_regional_audit.csv) · [Completed reflection](artifacts/artifacts_Day3/day3_reflection.json) · [Day 3 technical report](reports/DAY3_TECHNICAL_REPORT.md)

## Day 4 explanations and calibration

The Day 4 workflow separates model fitting, calibration, policy selection and evaluation by time and customer. The evaluation role contains 1,733 applications from 1,520 customers, with 139 defaults (8.02%), dated July-December 2024. The model, sigmoid calibrator and policy threshold were frozen before this evaluation. The notebook explicitly records prior course exposure, so this is a held-out evaluation within Day 4, not an untouched final test across the course. [Executed notebook](notebooks/Tamweel_aljuharaalshagri.ipynb)

| Data role | Applications | Customers | Defaults | Application dates |
| --- | ---: | ---: | ---: | --- |
| Model fitting | 2,516 | 1,853 | 217 | January 2022-April 2023 |
| Calibration | 584 | 538 | 40 | July-September 2023 |
| Policy selection | 589 | 566 | 55 | January-March 2024 |
| Evaluation | 1,733 | 1,520 | 139 | July-December 2024 |

### Global and local explanations

`bureau_score` and `dti` led both importance analyses. Permuting them reduced evaluation AP by 0.1270 and 0.0690. Their mean absolute SHAP contributions across the 300 sampled evaluation requests were 0.9042 and 0.5437 log-odds. SHAP decomposes the raw model margin; these values are not probability-point changes.

#### Permutation importance

![Training gain and evaluation average-precision decrease after permuting each feature or feature group.](artifacts/artifacts_Day4/permutation_importance.png)

*Training gain is shown beside the decrease in evaluation average precision after shuffling each feature or group. Whiskers show repeat standard deviation.*

#### Global SHAP beeswarm

![Global SHAP contributions in raw log-odds for 300 sampled evaluation requests.](artifacts/artifacts_Day4/shap_beeswarm.png)

*Each dot represents a sampled evaluation request. Horizontal position shows the signed SHAP contribution in raw log-odds; color represents the feature value.*

#### Local SHAP waterfall

For `TR-009585`, the highest-scoring request within the SHAP sample, the leading positive contributions were `bureau_score` (+2.2693), `dti` (+1.1981) and `loan_amount_sar` (+0.2007). Its raw score was 0.9031 and its calibrated probability was 0.4795. None of the three displayed reasons used an imputed value. These are explanations of the fitted model's behavior, with no causal or real-lending claim.

![Local SHAP waterfall explaining the raw log-odds of application TR-009585.](artifacts/artifacts_Day4/shap_waterfall.png)

*The waterfall reconstructs the request's raw log-odds from the baseline and feature contributions. The calibrated probability is computed separately.*

### Probability calibration

| Metric on the evaluation role | Raw score | Sigmoid calibrated probability |
| --- | ---: | ---: |
| ROC-AUC | 0.770804 | 0.770804 |
| Average precision | 0.258677 | 0.258677 |
| Brier score | 0.113027 | 0.067112 |
| Expected calibration error | 0.146871 | 0.022486 |
| Log loss | 0.357993 | 0.246749 |

Calibration lowered all three reported probability-error measures and preserved the ranking metrics. Sparse reliability bins remain noisy. In 200 paired customer-cluster bootstrap replicates, the 95% interval for the Brier change after calibration was [-0.054177, -0.037409]. These intervals hold the fitted model and calibrator fixed. [Saved metric tables and intervals](artifacts/artifacts_Day4/day4_saved_output_summary.json)

![Reliability curves and bin counts before and after sigmoid calibration on the Day 4 evaluation role.](artifacts/artifacts_Day4/reliability_curve.png)

### Stability of the results

![Customer-cluster bootstrap intervals for average precision and the distribution of Brier-score changes after calibration.](artifacts/artifacts_Day4/stability_summary.png)

*The 200 customer-cluster bootstrap replicates keep the fitted model and calibrator fixed. The left panel shows the 95% AP intervals; the right panel shows the Brier change after calibration, where negative values indicate improvement.*

### Review capacity

The policy-selected raw threshold was approximately 0.588195, transported to approximately 0.17331 on the calibrated scale. These are displayed values, not substitutes for the full-precision runtime thresholds. The diagnostic review band extends 0.02 on either side of the calibrated threshold. The candidate queue is the union of risk flags and near-threshold requests, counting each application once.

| Evaluation period | Applications | Capacity | Risk flags | Near threshold | Candidate queue | Excess queue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024Q3 | 836 | 100 | 97 | 23 | 109 | 9 |
| 2024Q4 | 897 | 107 | 109 | 21 | 122 | 15 |

The recorded status is `CAPACITY_REVIEW_REQUIRED`. Q4's risk flags alone also exceed capacity by two applications, so removing the diagnostic band would not resolve every breach. A revised policy needs selection on development or policy data and assessment on a new held-out period. The current evaluation set must not be reused for retuning.

![Review counts against the 12 percent capacity limit for each evaluation quarter.](artifacts/artifacts_Day4/review_zone.png)

The reflection returned `READY_FOR_REVIEW` with no missing fields. The export printed `TECHNICAL_READY | READY_FOR_REVIEW | LIVE | CPU`, alongside `CAPACITY_REVIEW_REQUIRED`. The readiness labels describe completed educational work; they do not indicate capacity acceptance or deployment approval. [Completed reflection](artifacts/artifacts_Day4/day4_reflection.json)

## Repository structure

| Location | Contents |
| --- | --- |
| [Combined notebook](notebooks/Tamweel_aljuharaalshagri.ipynb) | The complete Day 1-4 workflow, saved model outputs and completed reflections |
| [Day 3 notebook](notebooks/03_decision_thresholds.ipynb) | A self-contained entry point for the review-policy analysis |
| [Day 1 artifacts](artifacts/artifacts_Day1/) | Model comparison, held-out predictions, split assignments, figures and run records |
| [Day 2 artifacts](artifacts/artifacts_Day2/) | Leakage audits, validation scores, tuning history, OOF predictions and provenance |
| [Day 3 artifacts](artifacts/artifacts_Day3/) | Strategy comparison, exact threshold, loss sweep, period and regional audits, reflections and figures |
| [Day 4 evidence](artifacts/artifacts_Day4/) | Six original notebook figures, extracted displayed results, completed reflection and a source transcript |
| [Day 1 report](reports/DAY1_TECHNICAL_REPORT.md) | The baseline model choice and held-out evidence |
| [Day 2 report](reports/DAY2_TECHNICAL_REPORT.md) | Leakage controls and validation on later applications |
| [Day 3 report](reports/DAY3_TECHNICAL_REPORT.md) | The review policy, capacity tradeoff and interpretation of the selected threshold |
| [Day 4 report](reports/DAY4_TECHNICAL_REPORT.md) | Model explanations, calibration, stability and review capacity; [PDF](reports/DAY4_TECHNICAL_REPORT.pdf) and [Word](reports/DAY4_TECHNICAL_REPORT.docx) versions |

## Quick start

1. To run this downloaded version, upload `notebooks/Tamweel_aljuharaalshagri.ipynb` to Google Colab for the full capstone, or upload `notebooks/03_decision_thresholds.ipynb` for Day 3 alone. The **Run Days 1-4** and **Run Day 3** links above load the files on GitHub's `main` branch and reflect this version after it is uploaded there. Day 3 prepares its own data and models, so it does not require running Days 1 and 2 first. Select a CPU runtime; the saved results use fast mode.
2. Choose **Runtime > Run all**. If dependency installation restarts the session, reconnect and choose **Run all** again.
3. For Day 2, check for `Day 2 setup verified`. The saved run completed eight tuning trials and the reflection check returned `READY_FOR_REVIEW` with no missing answers. [Day 2 run](artifacts/artifacts_Day2/day2_run.json) · [Day 2 reflection](artifacts/artifacts_Day2/day2_reflection.json)
4. For Day 3, keep `MODEL_FOR_DECISION = "weighted"` to reproduce the documented policy. Check that the source is `LIVE`, OOF coverage includes all 5,039 eligible applications, and every period passes its capacity audit. An `EDUCATIONAL_EXAMPLE` fallback is for discussion only.
5. For Day 4, run its setup and role-separation cells before the explanation cells. Keep `USE_SHAP_EXAMPLE = False`. The saved run reports `LIVE`, 300 SHAP sample rows, 200 bootstrap replicates and a completed reflection. Its capacity warning is an observed result that should remain visible.
6. Run each day's export cell and download the resulting ZIP. The supplied combined notebook writes runtime outputs under `artifacts/`; its Day 4 export creates `artifacts/day4_artifacts.zip` and `reports/INTERPRETABILITY_REPORT.md`. Evidence already stored in this repository is organized by day under `artifacts/artifacts_Day1/` through `artifacts/artifacts_Day4/`.
7. The Day 4 folder supplied with this report contains figures and displayed results extracted from the saved notebook. The notebook reports exporting 28 files, but its original ZIP was not attached. Full predictions, bootstrap replicates, SHAP arrays and the trained booster must be obtained from the notebook's runtime export. They have not been reconstructed from summary tables.

## Dataset and models

The target, `default_within_90d`, records whether an application resulted in default within 90 days. Predictors include income in Saudi riyals and bureau score. Identifiers, dates and split-control fields are excluded from the model inputs. The recorded run used version `tamweel-lite-1.0` of the synthetic data and random seed `211`. [Environment](artifacts/artifacts_Day1/environment.json) · [Run settings](artifacts/artifacts_Day1/day1_run.json)

## Day 1 interpretation and limitations

Although the development and comparison sets contain distinct applications, 1,226 customers appear in both sets. This overlap motivated the Day 2 test on later applications from customers excluded from training. I would reconsider XGBoost if another model consistently achieved higher average precision on later cohorts. [Day 1 reflection](artifacts/artifacts_Day1/day1_reflection.json)

## Daily answers | الإجابات اليومية

### Day 1 | اليوم الأول

**1. Is your environment ready, with no errors shown in red?**

Yes. The saved Day 1 section completed its 11 code cells without errors. Its export cell saved the results and reported `TECHNICAL_READY` and `READY_FOR_REVIEW`. [Executed notebook](notebooks/Tamweel_aljuharaalshagri.ipynb)

**2. Which model did you choose, and what evidence supports your choice?**

I chose XGBoost because it had the highest average precision on the 2,000 held-out applications (0.3338), ahead of Logistic Regression (0.3258) and LightGBM (0.3248). Only 158 applications resulted in default (7.9%), so I prioritized how precisely the model identified defaults as recall increased. [Model scores](artifacts/artifacts_Day1/day1_model_comparison.csv)

### Day 2 | اليوم الثاني

**What leakage risk did you identify, and how did the validation design prevent it?**

I removed `days_past_due_60` and `collection_calls` because they record repayment behavior after submission. Using either field would give the model information a lender could not have when scoring the application.

For each forward fold, I trained only on earlier applications whose 90-day outcome window had closed before validation began. I also excluded every customer in that fold's validation set from its training set. The audit confirmed zero shared customers in all three folds. I selected hyperparameters using the separate historical tuning pool and froze them before outer evaluation. The challenge data played no role in parameter selection. [Feature audit](artifacts/artifacts_Day2/leakage_audit.csv) · [Fold audit](artifacts/artifacts_Day2/fold_audit.csv) · [Run record](artifacts/artifacts_Day2/day2_run.json)

### Day 3 | اليوم الثالث

**Why is a fixed threshold of 0.5 not always suitable for this review policy?**

A threshold of 0.5 does not account for this policy's unequal error costs or its review limit. Missing a default costs 10 teaching units, while incorrectly flagging a non-defaulting application costs 1 unit. Reviewers can handle at most 12% of applications in each period. The weighted model at 0.5 flags 1,005 of 5,039 applications (19.94%) and fails the capacity check, despite its lower loss of 2,403 units.

I therefore selected the threshold that minimizes observed loss while meeting capacity in every period: `score >= 0.6583471436014694`. It flags 526 applications, stays within all three period limits, and has a loss of 2,639 units. This choice accepts more missed defaults to keep the workload feasible. The weighted scores have not been established as calibrated probabilities, so neither 0.5 nor the theoretical cost-only threshold `1 / 11` can be assumed appropriate. The latter requires calibrated probabilities, this loss matrix and no capacity constraint. [Operating points](artifacts/artifacts_Day3/day3_operating_points.csv) · [Period audit](artifacts/artifacts_Day3/day3_period_audit.csv) · [Day 3 reflection](artifacts/artifacts_Day3/day3_reflection.json)

### Day 4 | اليوم الرابع

#### 1 What is the difference between global and local explanations

Global importance summarizes behavior across applications. `bureau_score` and `dti` had the largest mean absolute SHAP contributions (0.9042 and 0.5437 log-odds) and the largest permutation AP drops (0.1270 and 0.0690). The local explanation describes one request, `TR-009585`, whose leading positive contributions were `bureau_score` (+2.2693), `dti` (+1.1981) and `loan_amount_sar` (+0.2007). The global ranking reports average magnitude; local signed values show how features move that request's raw score relative to its SHAP base value.

#### 2 What units do the SHAP outputs use

The contributions use raw log-odds. Adding all contributions to the base value reconstructs the raw margin; applying `expit` gives the raw probability. The separate calibration mapping produces the calibrated probability. For `TR-009585`, these probabilities were 0.9031 and 0.4795. The maximum additivity error was about 5.7e-15. The displayed SHAP values do not directly decompose the calibrated probability.

#### 3 What can the reason codes establish

The reason codes describe associations in a model trained on synthetic data. They do not establish causation, fairness or suitability for real lending. Correlated predictors can share signal. The three displayed reasons for `TR-009585` used observed values, with no imputation, but changing those values does not guarantee a particular real-world outcome.

#### 4 What evidence supports calibration

On 1,733 evaluation applications with 139 defaults, Brier score fell from 0.113027 to 0.067112, ECE from 0.146871 to 0.022486, and log loss from 0.357993 to 0.246749. AP remained 0.258677 and ROC-AUC remained 0.770804. The model, calibrator and threshold were frozen before the Day 4 evaluation. The evidence supports improved probability agreement in this period, with limited precision in sparse reliability bins.

#### 5 How stable are the results and explanations

The 200 paired customer-cluster bootstrap replicates gave a 95% AP interval of [0.197442, 0.337854] for both variants and a Brier-change interval of [-0.054177, -0.037409]. AP was 0.27457 in Q3 and 0.27050 in Q4. The intervals cover evaluation-sample variability with the model and calibrator fixed. They exclude retraining and future distribution changes. For the selected request, changing `bureau_score` from 497 to 496 or 498 left both probabilities and all three leading positive reasons unchanged. That is a narrow local check.

#### 6 Does the review zone fit capacity

No. The candidate review queue contains 109 requests against capacity 100 in Q3 and 122 against capacity 107 in Q4. Q4 risk flags alone total 109, so removing the near-threshold band would still leave a breach. The recorded status is `CAPACITY_REVIEW_REQUIRED`. I would revise the operating policy using designated development or policy data and test it on a new held-out period, without retuning on the current evaluation set. The band is a diagnostic score range, not an individual confidence interval.

The full responses are preserved in the [completed reflection](artifacts/artifacts_Day4/day4_reflection.json).

#### Day 4 technical report

[Read the technical report](reports/DAY4_TECHNICAL_REPORT.md) | [Download PDF](reports/DAY4_TECHNICAL_REPORT.pdf) | [Download editable Word report](reports/DAY4_TECHNICAL_REPORT.docx)

The report develops these six answers with the experimental design, measured results, all six notebook figures, and a documented interpretation of the capacity warning.

## Training-program attribution

This project is part of SDAIA Academy's Advanced Machine Learning program (SDA-DSC-211) with Mrs. Meaad Al-Marri. [SDAIA Academy](https://github.com/SDAIAAcademy)
