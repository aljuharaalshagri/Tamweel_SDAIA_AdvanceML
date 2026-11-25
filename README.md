# Tamweel Lite

Tamweel Lite studies how well machine learning models can rank loan applications by the risk of default within 90 days. The project uses synthetic applications and information available at the time of submission.

Author: Aljuhara Alshagri

Program: Advanced Machine Learning (SDA-DSC-211), SDAIA Academy

| Notebook | Contents | Open in Colab |
| --- | --- | --- |
| [Tamweel capstone](notebooks/Tamweel_aljuharaalshagri.ipynb) | Gates 1-5 with saved outputs; Gate 5 ends at the reflection heading | [Run Gates 1-5](https://colab.research.google.com/github/aljuharaalshagri/Tamweel_SDAIA_AdvanceML/blob/main/notebooks/Tamweel_aljuharaalshagri.ipynb) |
| [Gate 3 decision thresholds](notebooks/03_decision_thresholds.ipynb) | The self-contained Gate 3 analysis and completed reflections | [Run Gate 3](https://colab.research.google.com/github/aljuharaalshagri/Tamweel_SDAIA_AdvanceML/blob/main/notebooks/03_decision_thresholds.ipynb) |
| [Gate 5 final project](notebooks/05_final_project.ipynb) | Gate 5 saved outputs and completed reflection cell to run | [Run Gate 5](https://colab.research.google.com/github/aljuharaalshagri/Tamweel_SDAIA_AdvanceML/blob/main/notebooks/05_final_project.ipynb) |

## Project overview

Gate 1 asks whether gradient boosting improves on a Logistic Regression baseline. I trained XGBoost and LightGBM alongside the baseline, then compared all three on the same 2,000 held-out applications. Each application has a score from every model, so the comparison uses the same outcomes throughout. [Held-out predictions](artifacts/artifacts_Day1/day1_comparison_predictions.csv)

Gate 2 tests whether LightGBM can rank default risk for later applications from customers excluded from training. I audited which fields were available at submission, set aside historical data for tuning, and compared random validation with three forward validation folds.

Gate 3 turns the risk scores into a review rule. I compared three training strategies, then used the predeclared weighted strategy to select a threshold under a loss of 10 units per missed default and 1 unit per false positive. At most 12% of applications can be flagged in each validation period. [Gate 3 operating points](artifacts/artifacts_Day3/day3_operating_points.csv)

Gate 4 examines a weighted LightGBM model through permutation importance, global and local SHAP explanations, sigmoid calibration, customer-cluster bootstrap intervals, and a review-capacity audit. The completed run improves probability agreement while leaving ranking unchanged. Its expanded review queue exceeds capacity in both evaluation quarters. [Gate 4 saved evidence](artifacts/artifacts_Day4/day4_saved_output_summary.json)

Gate 5 compares three single models and three ensembles using nested forward OOF predictions, then selects a final procedure and applies one capacity rule to the full challenge batch. The gate keeps Logistic Regression. The batch contains 2,500 requests, with 300 final flags after the 12% cap. Its sigmoid calibration diagnostics worsen, so the Gate 4 calibration improvement does not carry over to the final model. [Gate 5 results](artifacts/artifacts_Day5/evidence.json) · [Gate 5 technical report](reports/DAY5_TECHNICAL_REPORT.md)

## Gate 1 results

| Model | ROC-AUC | Average precision | Training time (s) |
| --- | ---: | ---: | ---: |
| Logistic Regression | **0.821** | 0.326 | 0.030 |
| XGBoost | 0.812 | **0.334** | 0.178 |
| LightGBM | 0.814 | 0.325 | 0.171 |

Logistic Regression led on ROC-AUC. XGBoost led on average precision, which matters here because only 158 of the 2,000 applications resulted in default (7.9%). Training times in this table come from the archived Gate 1 export and include tree-count selection and refitting for the boosting models. The latest combined notebook contains another execution with the same rounded ranking metrics and different timings. [Full comparison table](artifacts/artifacts_Day1/day1_model_comparison.csv)

![ROC and precision-recall curves for Logistic Regression, XGBoost and LightGBM on the held-out applications.](artifacts/artifacts_Day1/day1_roc_pr.png)

*The dotted horizontal line in the precision-recall panel marks the 7.9% default rate. XGBoost had the highest average precision (0.3338).*

## Gate 1 training pipeline

1. Select the 22 application-time predictors and reserve 2,000 of the 10,000 applications for comparison using a stratified random split.
2. Fit Logistic Regression on the 8,000 development applications after median imputation and feature scaling.
3. Within the development set, fit the boosting models on 6,000 applications and use the other 2,000 to select the tree count by log loss. XGBoost selected 61 trees and LightGBM selected 41.
4. Refit each boosting model on all 8,000 development applications at its selected tree count. Score all three models on the held-out applications and export the predictions, metrics and plots.

![Training and early-stopping log loss by boosting round for XGBoost and LightGBM.](artifacts/artifacts_Day1/day1_learning_curves.png)

*The vertical markers show the selected tree counts. Log loss on the early-stopping set was lowest at 61 trees for XGBoost and 41 for LightGBM.* [Gate 1 technical report](reports/DAY1_TECHNICAL_REPORT.md)

## Gate 2 validation design and results

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

For tuning, I reserved 1,935 earlier applications whose customers were absent from every outer validation fold. All eight Optuna trials ran on this pool. The best trial reached 0.3333 mean AP during tuning. On the 5,039 later applications, the fixed 80-tree model scored slightly higher than the tuned model (0.3153 versus 0.3133 AP). Both models produced one out-of-fold prediction per eligible application. The other 4,961 applications belonged to the initial historical period. [Trial history](artifacts/artifacts_Day2/optuna_results.csv) · [Out-of-fold coverage](artifacts/artifacts_Day2/day2_oof_coverage.csv) · [Gate 2 technical report](reports/DAY2_TECHNICAL_REPORT.md)

![Mean average precision for each of the eight Optuna trials and the best score reached so far on the historical tuning pool.](artifacts/artifacts_Day2/day2_search.png)

*The second trial reached the highest mean AP (0.3333) on the three tuning folds. The dashed line tracks that best score through the remaining trials.*

## Gate 3 review policy and results

The Gate 3 run produced out-of-fold (OOF) scores for 5,039 later applications, including 384 defaults (7.62%). Each validation fold uses a model trained on earlier applications with mature 90-day outcomes and no customers shared with that fold's validation set. Class weighting and oversampling affect training only; the validation applications retain their observed default rate. [Fold audit](artifacts/artifacts_Day3/day3_fold_audit.csv) · [Prediction coverage](artifacts/artifacts_Day3/day3_oof_coverage.csv)

| Training strategy | Pooled ROC-AUC | Pooled AP | Recall at 0.5 | Flags at 0.5 | Within every period's capacity at 0.5? |
| --- | ---: | ---: | ---: | ---: | --- |
| Unweighted | 0.7979 | 0.3091 | 12.50% | 87 | Yes |
| Weighted | 0.7969 | 0.3100 | 57.81% | 1,005 | No |
| Oversampled | 0.7979 | 0.3159 | 59.90% | 1,047 | No |

The weighted strategy was declared before comparison for the threshold exercise. The threshold analysis follows that predeclared choice. Oversampling had the highest pooled AP in this run. These pooled scores use the same 5,039 applications and differ from Gate 2's means across folds. [Strategy comparison](artifacts/artifacts_Day3/day3_strategy_comparison.csv)

![ROC and precision-recall curves for the unweighted, weighted and oversampled Gate 3 models.](artifacts/artifacts_Day3/day3_roc_pr.png)

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

Regional false-positive rates range from 7.64% in other to 8.29% in western, a gap of 0.6484 percentage points. The comparison has no confidence intervals and does not establish fairness. Regional recall also varies, from 34.62% to 48.15%. The threshold was selected using these OOF development outcomes, so its loss needs confirmation on untouched later applications. All costs are teaching units and all applications are synthetic. [Regional audit](artifacts/artifacts_Day3/day3_regional_audit.csv) · [Completed reflection](artifacts/artifacts_Day3/day3_reflection.json) · [Gate 3 technical report](reports/DAY3_TECHNICAL_REPORT.md)

## Gate 4 explanations and calibration

The Gate 4 workflow separates model fitting, calibration, policy selection and evaluation by time and customer. The evaluation role contains 1,733 applications from 1,520 customers, with 139 defaults (8.02%), dated July-December 2024. The model, sigmoid calibrator and policy threshold were frozen before this evaluation. The notebook explicitly records prior course exposure, so this is a held-out evaluation within Gate 4, not an untouched final test across the course. [Executed notebook](notebooks/Tamweel_aljuharaalshagri.ipynb)

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

![Reliability curves and bin counts before and after sigmoid calibration on the Gate 4 evaluation role.](artifacts/artifacts_Day4/reliability_curve.png)

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

## Gate 5 final model and review policy

Gate 5 selected the single Logistic Regression model. It achieved the highest mean AP across three forward periods, while the weighted ensemble had slightly lower fold variability and ECE. The recorded gate decision is `KEEP SINGLE`, with candidate `Logistic`. This choice uses a different validation cohort and procedure from Gate 1; the scores should not be read as a direct performance trend across gates. [Candidate comparison](artifacts/artifacts_Day5/day5_model_comparison.csv)

### Data roles and validation

| Data role | Applications | Customers | Defaults | First application | Last application |
| --- | ---: | ---: | ---: | --- | --- |
| Fitting and selection | 6,576 | 3,931 | 537 | 2022-01-01 | 2024-04-01 |
| Calibration only | 836 | 788 | 78 | 2024-07-01 | 2024-09-30 |

The role split excludes 2,588 rows. The nested comparison produced `LIVE` OOF predictions for 2,155 applications across three forward periods. Warm-up applications have no outer OOF prediction. The design separates inner ensemble fitting from outer scoring and addresses chronology, customer separation and 90-day label availability. Candidate selection and threshold selection use this OOF evidence, so the results still need an untouched later evaluation. Fold SD describes variation across three folds; it is not a confidence interval. [Data roles](artifacts/artifacts_Day5/day5_data_roles.csv) · [Saved evidence and provenance](artifacts/artifacts_Day5/evidence.json)

### Single model and ensemble comparison

| Candidate | Mean AP | Fold SD of AP | Mean Brier | Mean ECE | AP difference from best single |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.34549 | 0.04348 | 0.06608 | 0.02311 | -0.04617 |
| XGBoost | 0.35263 | 0.02904 | 0.06566 | 0.02276 | -0.03903 |
| Logistic Regression | 0.39166 | 0.02981 | 0.06327 | 0.01882 | 0.00000 |
| Equal averaging | 0.37170 | 0.03258 | 0.06435 | 0.02038 | -0.01996 |
| Weighted ensemble | 0.38942 | 0.02906 | 0.06332 | 0.01772 | -0.00224 |
| Stacking | 0.38314 | 0.02949 | 0.06603 | 0.03106 | -0.00852 |

The probability scores in this table use raw OOF predictions before the final sigmoid mapping. No ensemble passed the worth-it gate. Weighted was the closest on AP, but its score was lower by 0.00224 and its Brier score was marginally higher. Its small SD and ECE advantages did not provide an AP gain to justify maintaining several models and a combination step. The differences do not establish statistical superiority. [Full comparison](artifacts/artifacts_Day5/day5_model_comparison.csv)

![Mean AP with one fold standard deviation and raw OOF Brier and ECE for the six Gate 5 candidates.](artifacts/artifacts_Day5/day5_ensemble_comparison.png)

*The AP whiskers show one fold SD across three forward periods. Brier and ECE are lower-is-better scores; the plot precedes final calibration.*

![Pairwise correlations of OOF probabilities and residuals for LightGBM, XGBoost and Logistic Regression.](artifacts/artifacts_Day5/day5_diversity.png)

*The displayed probability correlations range from 0.88 to 0.96, while residual correlations range from 0.98 to 0.99. These rounded plot labels suggest substantial shared prediction behavior, but correlation alone does not determine whether an ensemble will improve AP.* [Displayed correlations](artifacts/artifacts_Day5/day5_probability_correlation_displayed.csv)

### OOF threshold and regional diagnostics

The selected raw threshold is `0.16892161427109176`. At this threshold, 245 of 2,155 OOF applications are flagged, approximately 11.37%. The maximum flagged fraction in any forward period is approximately 11.75%, below the 12% limit. The operating point contains 84 true positives, 161 false positives, 95 false negatives and 1,815 true negatives. Recall is 46.93%, precision is 34.29%, and the false-positive rate is 8.15%. The educational loss is `10 * 95 + 161 = 1,111`. These figures describe the threshold selected on OOF development outcomes. [OOF operating point](artifacts/artifacts_Day5/day5_oof_operating_point.csv) · [Exact threshold metadata](artifacts/artifacts_Day5/evidence.json)

| Region | Negative cases | Positive cases | False positives | True positives | FPR | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Central | 489 | 48 | 39 | 24 | 7.98% | 50.00% |
| Western | 486 | 47 | 50 | 21 | 10.29% | 44.68% |
| Eastern | 507 | 35 | 32 | 16 | 6.31% | 45.71% |
| Other | 494 | 49 | 40 | 23 | 8.10% | 46.94% |

FPR divides false positives by actual negative cases; recall divides true positives by actual positive cases. The largest FPR gap is approximately 3.98 percentage points between western and eastern. Each region contains only 35-49 positives, and the table includes no confidence intervals. These are OOF threshold diagnostics and do not establish fairness or describe the challenge batch after its cap. [Regional audit](artifacts/artifacts_Day5/day5_regional_audit.csv)

![OOF teaching loss across thresholds and regional false-positive rates with their negative-case denominators.](artifacts/artifacts_Day5/day5_policy_regions.png)

At the same fixed threshold, loss is 921, 1,111 and 1,301 teaching units when false-negative cost is 8, 10 and 12, respectively. The false-positive cost remains 1. These are sensitivity calculations for a fixed decision rule, not three independently optimized policies. [Cost sensitivity](artifacts/artifacts_Day5/day5_cost_sensitivity.csv)

### Calibration of the final model

The final classifier is refitted on the 6,576 fitting and selection applications. A separate block of 836 applications, including 78 defaults, is used to fit the sigmoid mapping. The displayed diagnostics are scored on those same calibration-fit rows. Holding them out from classifier fitting does not make them an independent test of the calibration mapping.

| Metric on calibration-fit rows | Raw probability | Sigmoid probability |
| --- | ---: | ---: |
| ROC-AUC | 0.789037 | 0.789037 |
| Average precision | 0.287803 | 0.287803 |
| Brier score | 0.076473 | 0.078058 |
| Log loss | 0.266489 | 0.277296 |
| ECE with 10 fixed-width bins | 0.021121 | 0.034871 |

All three probability-error scores worsen after sigmoid calibration, while the ranking metrics remain unchanged. The notebook nevertheless freezes and exports the mapping. These results do not support a calibration benefit; the mapping needs comparison on a separate later labeled cohort before deciding whether to retain it. [Calibration diagnostics](artifacts/artifacts_Day5/day5_calibration_diagnostics.csv)

![Raw and sigmoid reliability diagrams for the 836 calibration-fit applications with counts shown beside populated bins.](artifacts/artifacts_Day5/day5_calibration_fit.png)

*Both panels use the rows that fitted the sigmoid mapping. Bin labels are counts, and sparse bins limit precision.*

### Challenge batch and capacity

The raw policy threshold is transported through the fitted mapping to `0.12225843144286948`. The notebook applies the threshold, then orders eligible requests by probability and applies the 12% cap once to all 2,500 requests. Equal-score blocks are retained together; a boundary block is dropped if it cannot fit. The policy does not use outcomes or application IDs to break ties. [Full batch audit](artifacts/artifacts_Day5/day5_full_batch_audit_full_metadata.csv)

| Batch measure | Requests |
| --- | ---: |
| Full challenge batch | 2,500 |
| Threshold eligible | 330 |
| Capacity | 300 |
| Final flags | 300 |
| Removed by the cap | 30 |

Threshold eligibility is 13.2% of the batch, and the final flagged fraction is 12%. The saved boundary score is `0.12990841067105013`. The 30 removals were eligible under the fixed threshold but excluded by capacity; their outcomes are unknown. Challenge labels are unavailable, so the batch cannot provide measured recall, precision, calibration or outcome-based fairness. [Batch results](artifacts/artifacts_Day5/day5_full_batch_audit.csv)

![The challenge probability distribution and counts before and after the full-batch capacity cap.](artifacts/artifacts_Day5/day5_challenge_capacity.png)

### Explanation scope and completion

Gate 4's explanations describe a weighted LightGBM fitted on 2,516 rows. They do not explain the final Logistic Regression refit on 6,576 rows. New explanations must use the frozen Logistic pipeline and its preprocessing, with checks of output units, additivity, reason codes, imputation and local stability. The sigmoid mapping and the threshold-plus-cap decision require separate checks. A model-score explanation alone does not explain a removal caused by the batch cap.

The supplied combined notebook has saved outputs through challenge scoring and ends at the Gate 5 reflection heading. The [standalone Gate 5 notebook](notebooks/05_final_project.ipynb) adds the completed reflection cell, which must be run in the notebook environment. Its checkpoint has not been reported as executed. The [reflection text](artifacts/artifacts_Day5/day5_reflection.json) is included with this update. The evidence folder contains the five saved figures and extracted tables, with their source locations; it does not reconstruct the trained model files, complete OOF predictions or complete challenge submission. [Source manifest](artifacts/artifacts_Day5/source_manifest.json)

The project remains a fictional training exercise using synthetic applications. The final artifact must not be used for real loan approval, rejection, pricing, credit limits or customer-facing adverse-action reasons. [Gate 5 technical report](reports/DAY5_TECHNICAL_REPORT.md) · [Editable Word report](reports/DAY5_TECHNICAL_REPORT.docx)

## Repository structure

| Location | Contents |
| --- | --- |
| [Combined notebook](notebooks/Tamweel_aljuharaalshagri.ipynb) | Gates 1-5 with saved outputs; Gate 5 ends at the reflection heading |
| [Gate 3 notebook](notebooks/03_decision_thresholds.ipynb) | A self-contained entry point for the review-policy analysis |
| [Gate 5 notebook](notebooks/05_final_project.ipynb) | The final comparison and policy workflow with a completed reflection cell to run |
| [Gate 1 artifacts](artifacts/artifacts_Day1/) | Model comparison, held-out predictions, split assignments, figures and run records |
| [Gate 2 artifacts](artifacts/artifacts_Day2/) | Leakage audits, validation scores, tuning history, OOF predictions and provenance |
| [Gate 3 artifacts](artifacts/artifacts_Day3/) | Strategy comparison, exact threshold, loss sweep, period and regional audits, reflections and figures |
| [Gate 4 evidence](artifacts/artifacts_Day4/) | Six original notebook figures, extracted displayed results, completed reflection and a source transcript |
| [Gate 5 evidence](artifacts/artifacts_Day5/) | Five original notebook figures, extracted tables, full batch policy metadata, reflection text and source manifest |
| [Gate 1 report](reports/DAY1_TECHNICAL_REPORT.md) | The baseline model choice and held-out evidence |
| [Gate 2 report](reports/DAY2_TECHNICAL_REPORT.md) | Leakage controls and validation on later applications |
| [Gate 3 report](reports/DAY3_TECHNICAL_REPORT.md) | The review policy, capacity tradeoff and interpretation of the selected threshold |
| [Gate 4 report](reports/DAY4_TECHNICAL_REPORT.md) | Model explanations, calibration, stability and review capacity; [PDF](reports/DAY4_TECHNICAL_REPORT.pdf) and [Word](reports/DAY4_TECHNICAL_REPORT.docx) versions |
| [Gate 5 report](reports/DAY5_TECHNICAL_REPORT.md) | Final model selection, nested OOF evidence, calibration limits, regional diagnostics and capacity; [Word](reports/DAY5_TECHNICAL_REPORT.docx) version |

## Quick start

1. To run this downloaded version, upload `notebooks/Tamweel_aljuharaalshagri.ipynb` to Google Colab for the full capstone, `notebooks/03_decision_thresholds.ipynb` for Gate 3 alone, or `notebooks/05_final_project.ipynb` for Gate 5 and its completed reflection cell. The Colab links above load GitHub's `main` branch and reflect this version after the files are uploaded there. Gate 3 and Gate 5 each prepare their required data and support code. Select a CPU runtime; the saved results use fast mode.
2. Choose **Runtime > Run all**. If dependency installation restarts the session, reconnect and choose **Run all** again.
3. For Gate 2, check for `Day 2 setup verified`. The saved run completed eight tuning trials and the reflection check returned `READY_FOR_REVIEW` with no missing answers. [Gate 2 run](artifacts/artifacts_Day2/day2_run.json) · [Gate 2 reflection](artifacts/artifacts_Day2/day2_reflection.json)
4. For Gate 3, keep `MODEL_FOR_DECISION = "weighted"` to reproduce the documented policy. Check that the source is `LIVE`, OOF coverage includes all 5,039 eligible applications, and every period passes its capacity audit. An `EDUCATIONAL_EXAMPLE` fallback is for discussion only.
5. For Gate 4, run its setup and role-separation cells before the explanation cells. Keep `USE_SHAP_EXAMPLE = False`. The saved run reports `LIVE`, 300 SHAP sample rows, 200 bootstrap replicates and a completed reflection. Its capacity warning is an observed result that should remain visible.
6. Run the available export cells for Gates 1-4 and download the resulting ZIPs. The combined notebook writes runtime outputs under `artifacts/`; its Gate 4 export creates `artifacts/day4_artifacts.zip` and `reports/INTERPRETABILITY_REPORT.md`. Repository evidence is organized by gate under `artifacts/artifacts_Day1/` through `artifacts/artifacts_Day5/`.
7. The Gate 4 folder supplied with this report contains figures and displayed results extracted from the saved notebook. The notebook reports exporting 28 files, but its original ZIP was not attached. Full predictions, bootstrap replicates, SHAP arrays and the trained booster must be obtained from the notebook's runtime export. They have not been reconstructed from summary tables.
8. For Gate 5, run the standalone notebook from its setup cell with `USE_OOF_EXAMPLE = False`. Confirm that the source is `LIVE`, inspect the model gate and calibration diagnostics, and verify the full-batch cap. An example fallback cannot support final model export or challenge submission. Run the added reflection cell after the preceding cells; no previously executed Gate 5 reflection status is claimed.
9. The saved Gate 5 output includes evidence that model export and challenge scoring ran. To obtain the actual trained files and full prediction tables, rerun the notebook and collect them from its runtime. The supplied Gate 5 evidence contains extracted saved outputs, not a reconstructed runtime bundle.

## Dataset and models

The target, `default_within_90d`, records whether an application resulted in default within 90 days. Predictors include income in Saudi riyals and bureau score. Identifiers, dates and split-control fields are excluded from the model inputs. The recorded run used version `tamweel-lite-1.0` of the synthetic data and random seed `211`. [Environment](artifacts/artifacts_Day1/environment.json) · [Run settings](artifacts/artifacts_Day1/day1_run.json)

## Gate 1 interpretation and limitations

Although the development and comparison sets contain distinct applications, 1,226 customers appear in both sets. This overlap motivated the Gate 2 test on later applications from customers excluded from training. I would reconsider XGBoost if another model consistently achieved higher average precision on later cohorts. [Gate 1 reflection](artifacts/artifacts_Day1/day1_reflection.json)

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

### Day 5 final project

#### 1 Model selection

**Why did you choose the single model or the ensemble for the final project?**

I chose the single Logistic Regression model because it achieved the highest mean AP, 0.39166 with fold SD 0.02981. The weighted ensemble was closest at AP 0.38942 with SD 0.02906. Logistic had a slightly better raw OOF Brier score, 0.06327 versus 0.06332, while Weighted had lower ECE, 0.01772 versus 0.01882. No ensemble passed the worth-it gate. The small SD and ECE advantages did not justify adding several models without an AP gain. These differences support the choice made in this experiment, but do not prove statistical superiority. [Model comparison](artifacts/artifacts_Day5/day5_model_comparison.csv)

#### 2 Validation limits

The 2,155 LIVE OOF predictions cover three forward periods. The nested design separates inner ensemble fitting from outer scoring and addresses chronology, customer separation and mature 90-day labels. Warm-up rows have no outer OOF predictions. I used the outer results to select the candidate and threshold, so they remain selection evidence. Three folds and previously explored synthetic course data cannot replace an untouched later test. Fold SD is not a confidence interval, and the unlabeled challenge does not measure predictive performance. [Saved evidence](artifacts/artifacts_Day5/evidence.json)

#### 3 Calibration limits

The 836 calibration applications, with 78 positives, were held out from classifier fitting. The displayed sigmoid diagnostics nevertheless use the same rows that fitted the mapping. Brier worsened from 0.076473 to 0.078058, log loss from 0.266489 to 0.277296, and ECE from 0.021121 to 0.034871. AP and ROC-AUC stayed unchanged. I cannot claim a calibration benefit; the frozen mapping needs independent evaluation on a later labeled cohort. [Calibration diagnostics](artifacts/artifacts_Day5/day5_calibration_diagnostics.csv)

#### 4 Capacity and fairness

I applied the 12% cap once to the complete batch of 2,500 requests. It retained 300 of 330 threshold-eligible flags and removed 30 whose outcomes are unknown. Regional OOF FPR divides false positives by actual negatives: 39/489 in central, 50/486 in western, 32/507 in eastern and 40/494 in other. The largest FPR gap is about 3.98 percentage points. Recall uses actual positives, with only 35-49 per region. Capacity compliance does not establish fairness; selection and cap-removal patterns need regional checks, followed by outcome-based checks once labels mature. [Batch audit](artifacts/artifacts_Day5/day5_full_batch_audit_full_metadata.csv) · [Regional audit](artifacts/artifacts_Day5/day5_regional_audit.csv)

#### 5 Explanation scope

Day 4 explains a weighted LightGBM fitted on 2,516 rows. Day 5 refits Logistic Regression on 6,576 rows, so the earlier SHAP figures, importance rankings and reason codes do not explain this final model. I need a suitable explainer for the frozen Logistic pipeline and its preprocessing. I would recheck units, additivity, imputation, local reason codes and stability, then assess held-out permutation importance. Calibration and the threshold-plus-cap rule need separate checks. Model explanations do not establish causality or fairness.

#### 6 Intended use

This project ranks simulated 90-day default risk and examines review allocation using synthetic applications. The `10 * FN + FP` loss and 12% cap are teaching assumptions. The artifact is for coursework and reproducible experiments. It must not decide real loan approval, rejection, pricing or credit limits, or supply customer-facing adverse-action reasons.

#### 7 Monitoring plan

For each batch I would check schema, feature availability, missingness, duplicate IDs, feature distributions and score drift. I would record threshold eligibility, final flags and cap removals, enforce the full-batch 12% limit, and compare selection and removal rates by region. After 90-day labels mature, I would assess prevalence, AP, recall, precision, loss, Brier, log loss, ECE and reliability-bin counts. Regional outcome checks must show denominators and uncertainty. Any revised model, calibrator or policy needs assessment on fresh later data, and a refitted model needs updated explanations.

#### 8 Executive summary in Arabic

أختار الانحدار اللوجستي للمشروع التدريبي؛ فقد حقق أعلى متوسط AP بقيمة 0.39166 وانحراف معياري بين الطيات 0.02981، مقابل AP قدره 0.38942 للتجميع الموزون. لم يبرر التجميع زيادة التعقيد. طُبق سقف المراجعة بنسبة 12% على 2500 طلب، فبقيت 300 إشارة من أصل 330 فوق العتبة. تدهور Brier بعد المعايرة من 0.076473 إلى 0.078058، لذلك يلزم تقييم مستقل للمعايرة وإعادة تفسير النموذج النهائي. الاستخدام تعليمي ببيانات اصطناعية، ولا يشمل قرارات تمويل حقيقية.

#### 9 Executive summary in English

I selected Logistic Regression with mean AP 0.39166 and fold SD 0.02981, compared with AP 0.38942 for the weighted ensemble. No ensemble justified its added complexity. The full-batch 12% cap retained 300 of 2,500 requests and removed 30 of the 330 eligible flags. Sigmoid calibration worsened Brier from 0.076473 to 0.078058. Independent calibration evaluation and explanations for the final model are still needed. This artifact is restricted to fictional training use.

The nine answers are included in the [Day 5 reflection](artifacts/artifacts_Day5/day5_reflection.json) and the standalone notebook. The reflection code is complete but has not been executed as part of this documentation update.

[Read the Day 5 technical report](reports/DAY5_TECHNICAL_REPORT.md) · [Download the editable Word report](reports/DAY5_TECHNICAL_REPORT.docx)

## Training-program attribution

This project is part of SDAIA Academy's Advanced Machine Learning program (SDA-DSC-211) with Mrs. Meaad Al-Marri. [SDAIA Academy](https://github.com/SDAIAAcademy)
