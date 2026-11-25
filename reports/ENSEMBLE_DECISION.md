# قرار التجميع

KEEP SINGLE — Logistic

I kept Logistic Regression because it had the highest mean outer-fold AP (0.39166, fold SD 0.02981). The best ensemble by AP was Weighted (0.38942, SD 0.02906), so its AP was lower by 0.00224. Logistic also had a slightly lower raw OOF Brier score (0.06327 versus 0.06332), although Weighted had a lower ECE (0.01772 versus 0.01882). Equal averaging and stacking had lower AP, at 0.37170 and 0.38314. No ensemble passed the worth-it gate. These small differences do not establish statistical superiority, but they provide no evidence of an AP gain that justifies maintaining multiple models, blend weights or a stacking model.

The nested forward OOF comparison produced 2,155 LIVE predictions across three forward periods. The design separates inner ensemble fitting from outer scoring and addresses time ordering, customer separation and 90-day label availability. Warm-up rows have no outer OOF predictions. However, these outer predictions were used to choose the candidate and its policy threshold, so the reported results are selection evidence and still require a separate final evaluation. Three folds provide limited evidence about temporal stability, and fold SD is not a confidence interval. The data are synthetic and have already been explored during the course. The 2,500 challenge requests have no labels, so they cannot establish predictive performance.

الدليل: artifacts/ensemble_comparison.csv وday5_ensemble_gate.json. SD وصفي، وليس اختبار دلالة.
