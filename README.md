# Tamweel Lite

Tamweel Lite uses synthetic loan applications to study how well machine learning models can rank applicants by their risk of default within 90 days. Each model scores an application using only information available when it was submitted.

This project is part of SDAIA Academy's **Advanced Machine Learning** training program. [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy)


## Day 1 technical notes

The target, `default_within_90d`, records whether a default occurred within 90 days after submission. The models use 22 application-time predictors. Customer IDs, dates, split-control columns, and the target are excluded from the inputs.

The notebook uses a stratified random split of 10,000 applications. For XGBoost and LightGBM, it fits trees on 6,000 rows and selects the tree count using log loss on another 2,000. It then fits each final model on all 8,000 development rows and evaluates it on the same 2,000 held-out applications. Logistic Regression uses median imputation and feature scaling.

| Model | ROC-AUC | Average precision |
| --- | ---: | ---: |
| Logistic Regression | **0.821** | 0.326 |
| XGBoost | 0.812 | **0.334** |
| LightGBM | 0.814 | 0.325 |

Defaults accounted for 7.9% of the comparison set. ROC-AUC measures how well the model ranks defaults above non-defaults. Average precision shows how precisely it identifies defaults as recall increases. These scores describe one random split of synthetic data. A test on later applications from customers excluded from training would better assess performance on new applicants.

### Reading the exported files

| To inspect | Open in `artifacts/` |
| --- | --- |
| Model scores and training times | `day1_model_comparison.csv` |
| Actual outcomes and model scores for individual applications | `day1_comparison_predictions.csv` |
| Which rows were used for fitting, early stopping, or comparison | `day1_split_membership.csv` |
| How training and stopping loss changed as trees were added | `day1_learning_curves.png` |
| ROC and precision-recall curves | `day1_roc_pr.png` |
| Run settings, software environment, and checkpoint record | `day1_run.json`, `environment.json`, `day1_reflection.json` |

## Daily answers | الإجابات اليومية

### Day 1

**1. Is your environment ready, with no errors shown in red?**

My Colab environment was ready for the model comparison. It produced ROC and precision-recall curves for all three models. Verifying that every cell ran without errors requires the full notebook output.

**2. Which model did you choose, and what evidence supports your choice?**

I selected **Logistic Regression** as my initial model. It achieved the highest ROC-AUC (0.821), and its average precision (0.326) was only 0.008 below XGBoost's (0.334). Given that trade-off, I chose the simpler model as my baseline.

## Git workflow

Keep each commit focused on one change. For a model run, commit the executed notebook, its matching results, and the README entry together so a reader can trace the reported scores to the code that produced them. Review the staged changes before committing. For example, after saving the Day 1 files:

```bash
git status --short
git add -- README.md notebooks/01_baseline_boosting.ipynb
git add -- artifacts/day1_*.csv artifacts/day1_*.png artifacts/day1_*.json artifacts/environment.json
git diff --cached --stat
git diff --cached -- README.md
git commit -m "Compare baseline and boosting models in Day 1 lab"
```
