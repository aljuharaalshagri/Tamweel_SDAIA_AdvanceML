# تقريرك: التفسير والمعايرة — Tamweel Lite

**الحالة:** جاهز للمراجعة؛ لا يعني اعتمادًا أو درجة

**مصدر التفسير:** LIVE. **النموذج والمعايرة:** LIVE. **السعة:** CAPACITY_REVIEW_REQUIRED.

## النموذج والأدوار
LightGBM موزون، 80 شجرة. الهدف حدث تعثر اصطناعي خلال90 يومًا بعد الطلب. الأدوار منفصلة زمنيًا وبالعملاء: تدريب 2516، معايرة 584 (40 موجب)، سياسة 589، تقييم 1733. الفجوات والتداخلات مستبعدة. سبق استخدام بيانات التقييم في الدورة، فهي ليست اختبارًا نهائيًا لم يمسّ.

## التفسير العام والمحلي
Permutation يقيس انخفاضAP على التقييم؛ إشارات المنطقة تُبدّل معًا. SHAP يفسر النموذج الخام بوحدةlog-odds وخلفية مسارات أشجار التدريب. base+sum(SHAP)=raw margin، ثمsigmoid للمجموع فقط. القيم ليست نقاط احتمال ولا تفسيرًا مباشرًا للنموذج المعاير.

Global explanations summarize how the model uses features across many applications. In the LIVE SHAP sample of 300 evaluation requests, bureau_score and dti had the largest mean absolute contributions, at 0.9042 and 0.5437 log-odds. Held-out permutation importance agreed: shuffling these features reduced average precision by 0.1270 and 0.0690, respectively. The local explanation describes TR-009585, the highest-scoring request in that sample. Its main positive contributions were bureau_score (+2.2693), dti (+1.1981), and loan_amount_sar (+0.2007). Global importance describes average influence, while local SHAP values explain the direction and size of contributions for one request.

The SHAP values are contributions to the model's raw log-odds, not percentage-point changes in default probability. Adding all SHAP values to the base value reconstructs the raw margin, and applying expit converts that total to the raw probability. The separate sigmoid calibration mapping then produces the calibrated probability. For TR-009585, the raw score was 0.9031 and the calibrated probability was 0.4795. The maximum additivity error was approximately 5.7e-15, confirming numerical agreement with the raw margin. These SHAP values do not directly decompose the calibrated probability.

الطلب الاصطناعي TR-009585: الدرجة الخام 0.90308 والاحتمال المعاير 0.47952. اختير أعلى درجة داخل عينةSHAP دون استخدام النتيجة الفعلية.
- استخدم النموذج درجة ائتمانية اصطناعية عند الطلب بالقيمة 497 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+2.2693 log-odds؛ قيمة معوضة: False)
- استخدم النموذج نسبة الالتزام مع القسط المقترح إلى الدخل بالقيمة 1.2806 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+1.1981 log-odds؛ قيمة معوضة: False)
- استخدم النموذج مبلغ التمويل المطلوب بالقيمة 93437.3 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+0.2007 log-odds؛ قيمة معوضة: False)

The reason codes describe associations learned by the fitted model from synthetic data. They do not establish that a feature causes default or that changing it will produce a particular outcome. For TR-009585, the displayed reasons used bureau_score of 497, dti of 1.2806, and loan_amount_sar of 93437.29. All three values were observed, with was_imputed=False. Correlated features can share predictive information, which affects how importance is allocated. These explanations help inspect the model's behavior, but they do not establish fairness or suitability for real lending decisions.

## دليل المعايرة
على 1733 صفًا و139 موجب: Brier 0.113027 → 0.067112؛ ECE 0.146871 → 0.022486. عشر حاويات متساوية العرض مع أعدادها فيday4_reliability_bins.csv. AP 0.258677 → 0.258677؛ ROC-AUC 0.770804 → 0.770804. هذه نتائج هذه العينة وليست ضمانًا لتحسن مستقبلي.

On 1733 evaluation applications containing 139 defaults, sigmoid calibration reduced the Brier score from 0.113027 to 0.067112, ECE from 0.146871 to 0.022486, and log loss from 0.357993 to 0.246749. The model and calibrator were fitted using separate data roles and frozen before evaluation. ROC-AUC remained 0.770804 and average precision remained 0.258677, so calibration improved the measured agreement between probabilities and outcomes while preserving ranking. The reliability curve also moved closer to the agreement line. Sparse bins still limit how precisely calibration can be assessed in those score ranges.

## الاستقرار
200 تكرارbootstrap صالح بسحب العملاء؛ فترات مئينية95% مع تثبيت النموذج والمعاير. لا تشمل تعلم النموذج أو المعايرة أو الانجراف المستقبلي، ولا تصف احتمال فرد. انحرافAP بين ربعي التقييم وصفي فقط. اختبارbureau_score±1 نُفذ؛ راجع day4_local_stability.csv.

Across 200 paired customer-cluster bootstrap replicates, the 95% percentile interval for average precision was [0.197442, 0.337854] for both raw and calibrated predictions. The interval for Brier change, calculated as after minus before, was [-0.054177, -0.037409]. Its entirely negative range supports the measured improvement under this resampling procedure. Quarterly AP was 0.27457 in 2024Q3 and 0.27050 in 2024Q4. These intervals describe evaluation-sample variability with the model and calibrator fixed; they exclude uncertainty from retraining, refitting calibration, and future distribution changes. Locally, changing bureau_score from 497 to 496 or 498 left both probabilities and all three leading positive reasons unchanged. This supports stability only for those small changes to that particular request.

## العتبة ومنطقة المراجعة
العتبة الخام 0.5881953696965011 اختيرت علىpolicy بخسارة10×FN+FP وسقف12% ثم نُقلت إلى 0.17331013263107387. لم تعدل باستخدام التقييم. المنطقة[0.15331, 0.19331] تشخيصية بعرض±0.02 وليست فترة ثقة. الاتحاد يحسب الطلب مرة واحدة.
- 2024Q3: السقف 100، الإشارات 97، اتحاد المراجعة 109.
- 2024Q4: السقف 107، الإشارات 109، اتحاد المراجعة 122.

The raw threshold selected on the policy set was 0.588195, which mapped to a calibrated threshold of 0.17331. It flagged 68 policy applications against a capacity of 70. The diagnostic review band extended 0.02 on either side of the calibrated threshold, approximately [0.15331, 0.19331]. On evaluation, the union of risk flags and near-threshold requests contained 109 applications in 2024Q3 against a capacity of 100, and 122 applications in 2024Q4 against a capacity of 107. Overlapping requests were counted once. Even without the band, Q4 had 109 risk flags, already exceeding its capacity of 107. The resulting status is CAPACITY_REVIEW_REQUIRED. I would revisit the operating policy using designated policy or development data, then assess any revised threshold or review band on a new held-out period. I would not retune using the current evaluation set.

عند تجاوز السعة، وثّق الحاجة إلى تصميم سياسة جديدة على بيانات تطوير وتقييمها بدليل جديد. لا ترفع السقف ولا تقص الحالات بعد رؤية النتيجة. التفسير ليس سببية أو شهادة عدالة، والخسارة وحدات تعليمية لا رسوم أو خصم درجات. لا يستخدم هذا التمرين لتمويل حقيقي.
