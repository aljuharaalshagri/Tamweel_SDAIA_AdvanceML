# Day 3 technical report

## Decision question

Which score threshold should flag loan applications for review when missed defaults cost more than false alarms and review capacity is limited? Day 3 uses the forward-validation design from Day 2 to compare training strategies and choose a review rule. The selected weighted-model threshold is `0.6583471436014694`, applied with `>=`. On the 5,039 eligible applications, it flags 526, catches 157 defaults and stays within the 12% review limit in every period. [Combined notebook](../notebooks/Tamweel_aljuharaalshagri.ipynb) [Exact decision rule](../artifacts/artifacts_Day3/day3_threshold.json)

These results describe threshold development on synthetic applications. The loss values are teaching units. The threshold was selected and assessed using the same out-of-fold (OOF) outcomes.

## Policy and evaluation population

The policy was fixed before threshold selection. A false negative is an application that defaulted but was not flagged; a false positive is a non-defaulting application that was flagged. The objective is to minimize `10 * false_negatives + false_positives`, subject to a separate review limit for each period.

| Policy setting | Value |
| --- | --- |
| False-negative cost | 10 teaching loss units |
| False-positive cost | 1 teaching loss unit |
| Period review capacity | At most 12% of that period's applications, rounded down to a whole application count |
| Decision strategy | `weighted`, declared before the strategy comparison |
| Decision rule | Flag when the weighted score is greater than or equal to the selected threshold |

The source dataset has 10,000 applications and 789 defaults. The three forward-validation windows contain 5,039 applications and 384 defaults, a prevalence of 7.62%. The remaining 4,961 earlier applications form the warm-up period and have no OOF predictions. Coverage is complete for the eligible validation population. [Coverage audit](../artifacts/artifacts_Day3/day3_oof_coverage.csv) · [Run record](../artifacts/artifacts_Day3/day3_run.json)

| Validation window | Training applications | Validation applications | Validation defaults | Shared customers within the fold |
| --- | ---: | ---: | ---: | ---: |
| July-December 2023 | 3,223 | 1,632 | 110 | 0 |
| January-June 2024 | 4,460 | 1,674 | 135 | 0 |
| July-December 2024 | 5,731 | 1,733 | 139 | 0 |

Before each validation window, training uses only earlier applications whose 90-day outcomes were already known. Training applications from customers appearing in that validation window are excluded. Each OOF score therefore comes from a model that did not train on its validation fold or its customers. This design reduces optimism from training-set evaluation and prevents later outcomes from entering the earlier training cutoff. [Fold audit](../artifacts/artifacts_Day3/day3_fold_audit.csv) · [Prediction provenance](../artifacts/artifacts_Day3/day3_provenance.json)

## Training-strategy comparison

The fast-mode run uses 80 trees per model and compares unweighted, weighted and oversampled training. Weighting and oversampling are applied only during training. The validation applications and their default prevalence stay the same across all three strategies. The oversampled training sets have a 50% positive rate, while the weighted models use class weights without duplicating training rows. [Fold-level model report](../artifacts/artifacts_Day3/day3_model_report.csv)

| Training strategy | Pooled ROC-AUC | Pooled average precision | Accuracy at 0.5 | Recall at 0.5 | Flags at 0.5 | Loss at 0.5 | Capacity feasible at 0.5? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Unweighted | 0.7979 | 0.3091 | 92.56% | 12.50% | 87 | 3,399 | Yes |
| Weighted | 0.7969 | 0.3100 | 81.25% | 57.81% | 1,005 | 2,403 | No |
| Oversampled | 0.7979 | 0.3159 | 80.73% | 59.90% | 1,047 | 2,357 | No |

The notebook declares `MODEL_FOR_DECISION = "weighted"` before comparison and uses that strategy throughout threshold selection. Oversampling has the highest pooled AP in this run. Pooled metrics combine the 5,039 application-level predictions; Day 2's reported means average separate fold scores. [Strategy comparison](../artifacts/artifacts_Day3/day3_strategy_comparison.csv) · [OOF predictions](../artifacts/artifacts_Day3/day3_oof_predictions.csv)

![ROC and precision-recall curves for the three training strategies on the same OOF applications.](../artifacts/artifacts_Day3/day3_roc_pr.png)

Accuracy alone would give a misleading decision argument. A rule that flags nobody achieves 92.38% accuracy, yet misses all 384 defaults and incurs 3,840 loss units. Average precision evaluates the ranking under class imbalance. Recall, precision, loss and period workload then describe what happens at the selected operating threshold. [Saved no-flag comparison](../notebooks/03_decision_thresholds.ipynb)

## Threshold selection and the cost of capacity

The sweep evaluates the decision rules induced by the scores and includes the default 0.5 threshold. The selected threshold minimizes observed loss over the rules that satisfy the review limit in every validation period. The full-precision value and its `>=` operator must remain together when applying or exporting the rule. [Threshold sweep](../artifacts/artifacts_Day3/day3_threshold_sweep.csv) · [Exact decision rule](../artifacts/artifacts_Day3/day3_threshold.json)

| Rule | Threshold | True positives | False positives | False negatives | True negatives | Loss units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 0.5000 | 222 | 783 | 162 | 3,872 | 2,403 |
| Minimum loss without capacity limit | 0.4486 | 246 | 895 | 138 | 3,760 | 2,275 |
| Minimum loss within every period's capacity | 0.6583 | 157 | 369 | 227 | 4,286 | 2,639 |

The table rounds thresholds for readability. The selected rule is `score >= 0.6583471436014694`. Its loss is `10 * 227 + 369 = 2,639`, with 40.89% recall and 29.85% precision. It flags 526 applications, or 10.44% of the OOF population. [Operating points](../artifacts/artifacts_Day3/day3_operating_points.csv)

Without a capacity limit, the minimum-loss rule flags 1,141 applications and reaches 64.06% recall. Enforcing the limit reduces flags by 615 and catches 89 fewer defaults; loss rises by 364 units. Relative to the default threshold, the selected rule reduces flags by 479, catches 65 fewer defaults and increases loss by 236 units. Both lower-loss alternatives fail the period capacity requirement.

![Loss per 10,000 applications and review volume across thresholds. Tables show total loss on 5,039 applications.](../artifacts/artifacts_Day3/day3_cost_curve.png)

### Why is a fixed threshold of 0.5 not always suitable for this review policy?

A threshold of 0.5 does not encode the policy's unequal error costs or its review limit. In this run, the weighted model at 0.5 flags 1,005 applications, or 19.94%, and fails the capacity audit. Its loss of 2,403 units is lower than the selected feasible rule's loss, but the required reviews exceed what the policy permits.

The weighted scores have not been demonstrated to be calibrated default probabilities. A score of 0.6583 therefore cannot be read as a 65.83% default probability. With calibrated probabilities, zero cost for correct decisions, these error costs and no capacity limit, the cost-only threshold would be `1 / (10 + 1)`, approximately 0.0909. Those conditions do not justify applying that formula to the current weighted scores. The empirical sweep and the per-period capacity checks determine the documented threshold. [Operating points](../artifacts/artifacts_Day3/day3_operating_points.csv) · [Completed reflection](../artifacts/artifacts_Day3/day3_reflection.json)

## Period capacity and policy sensitivity

| Period | Applications | Maximum reviews | Actual flags | Flagged fraction | Within capacity? |
| --- | ---: | ---: | ---: | ---: | --- |
| July-December 2023 | 1,632 | 195 | 137 | 8.39% | Yes |
| January-June 2024 | 1,674 | 200 | 183 | 10.93% | Yes |
| July-December 2024 | 1,733 | 207 | 206 | 11.89% | Yes |

The last period is closest to its limit, with one review slot remaining. The shared threshold need not use every available slot in the earlier periods: feasibility is checked separately in all three. A combined flag rate below 12% would be insufficient evidence if one period exceeded its own limit. [Period audit](../artifacts/artifacts_Day3/day3_period_audit.csv)

Changing the false-negative cost to 8, 10 or 12 units leaves the selected feasible threshold unchanged in this run. The corresponding loss is 2,185, 2,639 or 3,093 units; the flag count remains 526 and recall remains 40.89%. This sensitivity check covers those tested costs on the same OOF scores. [Sensitivity results](../artifacts/artifacts_Day3/day3_sensitivity.csv)

## Regional differences at the shared threshold

The false-positive rate uses non-defaulting applications as its denominator. All regions use the same threshold.

| Region | Non-defaulting applications | Defaults | False positives | False-positive rate | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Central | 1,088 | 96 | 88 | 8.09% | 44.79% |
| Western | 1,158 | 104 | 96 | 8.29% | 34.62% |
| Eastern | 1,192 | 103 | 92 | 7.72% | 37.86% |
| Other | 1,217 | 81 | 93 | 7.64% | 48.15% |

The highest-to-lowest false-positive-rate gap is 0.6484 percentage points, between western and other. These are descriptive estimates without confidence intervals. They do not establish fairness or equivalent regional performance, and recall varies from 34.62% to 48.15%. [Regional audit](../artifacts/artifacts_Day3/day3_regional_audit.csv)

![Review counts relative to each period's capacity and regional false-positive rates at the selected threshold.](../artifacts/artifacts_Day3/day3_capacity_regions.png)

## Interpretation and reproducibility

OOF predictions separate model fitting from each scored validation application. Threshold selection still uses the OOF outcomes, so the selected threshold's measured loss can be optimistic. The frozen rule needs evaluation on untouched later applications before making a claim about future performance. A fixed threshold that met historical capacity may also exceed capacity if future score distributions change.

The synthetic dataset and teaching loss matrix do not establish financial savings on real lending data. Calibration and operational costs need their own evidence. The repository records the exact threshold, predictions, fold assignments, completed reflections and run provenance so the decision can be traced to the experiment. [Run record](../artifacts/artifacts_Day3/day3_run.json) · [Environment](../artifacts/artifacts_Day3/day3_environment.json) · [Reflection](../artifacts/artifacts_Day3/day3_reflection.json)

The combined notebook includes all three days. The standalone Day 3 notebook prepares its own verified course data and helpers. Its final export cell writes the tables, JSON records and figures to `artifacts/artifacts_Day3/` and packages them as `day3_artifacts.zip`. Re-running the analysis can change timings; the stored threshold must be taken from the resulting decision file without rounding.
