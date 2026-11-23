# Tamweel Lite

Tamweel Lite studies how well machine learning models can rank loan applications by the risk of default within 90 days. The project uses synthetic applications and information available at the time of submission.

**Author** Aljuhara Alshagri

**Program** Advanced Machine Learning (SDA-DSC-211), SDAIA Academy

[![Open the Day 1 notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aljuharaalshagri/Tamweel_SDAIA_AdvanceML/blob/main/notebooks/01_baseline_boosting.ipynb)

## Project overview

Day 1 asks whether gradient boosting improves on a Logistic Regression baseline. I trained XGBoost and LightGBM alongside the baseline, then compared all three on the same 2,000 held-out applications. Each application has a score from every model, so the comparison uses the same outcomes throughout. [Held-out predictions](artifacts/day1_comparison_predictions.csv)

## Key results

| Model | ROC-AUC | Average precision | Training time (s) |
| --- | ---: | ---: | ---: |
| Logistic Regression | **0.821** | 0.326 | 0.082 |
| XGBoost | 0.812 | **0.334** | 0.458 |
| LightGBM | 0.814 | 0.325 | 0.341 |

Logistic Regression led on ROC-AUC. XGBoost led on average precision, which matters here because only 158 of the 2,000 applications resulted in default (7.9%). Training times are from the recorded CPU run and include tree-count selection and refitting for the boosting models. [Full comparison table](artifacts/day1_model_comparison.csv)

![ROC and precision-recall curves for all three models. The precision-recall panel marks the 7.9% default rate.](artifacts/day1_roc_pr.png)

## Technical pipeline

1. Select the 22 application-time predictors and reserve 2,000 of the 10,000 applications for comparison using a stratified random split.
2. Fit Logistic Regression on the 8,000 development applications after median imputation and feature scaling.
3. Within the development set, fit the boosting models on 6,000 applications and use the other 2,000 to select the tree count by log loss. XGBoost selected 61 trees and LightGBM selected 41.
4. Refit each boosting model on all 8,000 development applications at its selected tree count. Score all three models on the held-out applications and export the predictions, metrics and plots.

[Learning curves](artifacts/day1_learning_curves.png) · [Technical report](reports/DAY1_TECHNICAL_REPORT.md)

## Repository structure

| Location | Contents |
| --- | --- |
| [notebooks/01_baseline_boosting.ipynb](notebooks/01_baseline_boosting.ipynb) | Executed Day 1 notebook, including the model comparison and written reflection |
| [artifacts/](artifacts/) | Eight original run outputs, including row-level predictions, split membership, figures and environment details |
| [reports/DAY1_TECHNICAL_REPORT.md](reports/DAY1_TECHNICAL_REPORT.md) | Training method, model choice and proposed validation test |

## Quick start

1. Open the Colab badge above and select a CPU runtime. The default settings are `FAST_MODE=True` and `FULL_MODE=False`.
2. Choose **Runtime > Run all**. If the first cell installs the pinned package versions, it restarts the session automatically. After Colab reconnects, choose **Run all** again.
3. Check for `Day 1 setup verified` in the setup output. The last code cell prints `TECHNICAL_READY | READY_FOR_REVIEW` when the comparison and reflection checks pass.
4. Download `tamweel/artifacts/day1_artifacts.zip` from the Colab Files panel to inspect the outputs of your own run. The [saved notebook](notebooks/01_baseline_boosting.ipynb) also displays the original results on GitHub.

## Dataset and models

The target, `default_within_90d`, records whether an application resulted in default within 90 days. Predictors include income in Saudi riyals and bureau score. Identifiers, dates and split-control fields are excluded from the model inputs. The recorded run used version `tamweel-lite-1.0` of the synthetic data and random seed `211`. [Environment](artifacts/environment.json) · [Run settings](artifacts/day1_run.json)

## Interpretation and limitations

Although the development and comparison sets contain distinct applications, 1,226 customers appear in both sets. To assess whether the ranking transfers to new customers, I would train on earlier applications whose 90-day outcomes were known at the training cutoff, then test on later applications from different customers. I would reconsider XGBoost if it consistently achieved higher average precision and found more defaults at the same review capacity. [Day 1 reflection](artifacts/day1_reflection.json)

## Daily answers | الإجابات اليومية

### Day 1 | اليوم الأول

**1. Is your environment ready, with no errors shown in red?**

Yes. All 11 code cells in my saved notebook ran in order. The final cell exported the results and reported `TECHNICAL_READY` and `READY_FOR_REVIEW`. [Executed notebook](notebooks/01_baseline_boosting.ipynb)

**2. Which model did you choose, and what evidence supports your choice?**

I chose Logistic Regression as my initial baseline. It had the highest ROC-AUC (0.821), and its average precision (0.326) was only 0.008 below XGBoost's (0.334). That trade-off favored the simpler model for the next comparison. [Model scores](artifacts/day1_model_comparison.csv)

## Git workflow

Keep each day's executed notebook, matching outputs and README answer in one commit. Review the staged files with `git diff --cached --stat` before committing. The Day 1 commit can be titled `Organize Day 1 model comparison and results`.

## Training-program attribution

This project is part of SDAIA Academy's Advanced Machine Learning program (SDA-DSC-211). Meaad Al-Marri prepared the course template, synthetic data and lab materials. This submission contains my executed notebook, model comparison and interpretation. [SDAIA Academy](https://github.com/SDAIAAcademy) · [Course template](https://github.com/almiyead-rgb/sda-dsc-211-student-template) · [Day 1 submission guide](https://github.com/almiyead-rgb/sda-dsc-211-student-template/blob/main/DAY1_GUIDE.md)
