# بطاقة قرارك — Tamweel Lite

**الحالة:** جاهزة للمراجعة؛ لا تعني اعتمادًا أو درجة
**مصدر الأرقام:** LIVE · **الاستراتيجية:** weighted · **الصفوف:** 5,039 OOF

**المهمة:** الفئة الموجبة `default_within_90d=1` تعني حدث تعثر اصطناعي خلال90 يومًا بعد الطلب. كل طية تحقق طلبات لاحقة، وتستبعد عملاءها من التدريب وتشترط نضج نتيجة التدريب قبل بدايتها. المعرّفات والتاريخ خارج المدخلات.

| الدليل | القيمة |
|---|---:|
| العتبة المقيدة، بالقيمة الكاملة | 0.6583471436014694 |
| عتبة أقل خسارة دون قيد | 0.44863935722081005 |
| Recall | 40.89% |
| Precision | 29.85% |
| AP مجمع منOOF | 0.3100 |
| الإشارات | 526 من 5,039 |
| FN / FP | 227 / 369 |
| الخسارة التعليمية | 2639 وحدة |
| خسارة0.5 | 2403 وحدة؛ ضمن السعة: False |
| التغير عن0.5 | +236 وحدة؛ الموجب زيادة |
| خسارة لكل10,000 طلب، تطبيع حسابي | 5237.15 وحدة |
| فجوة معدل الإنذار الخاطئ بين المناطق | 0.648 نقطة مئوية |

**القاعدة:** درجة ≥ 0.6583471436014694 تعني إشارة مراجعة داخل التمرين؛ غير ذلك بلا إشارة. لا تتخذ موافقة أو رفض تمويل حقيقي. احفظ الدقة الكاملة؛ تقريب العتبة قد يغيّر حجم الطابور.

**السياسة:** FN=10 وFP=1 وحدات تعليمية، وسعة 12% لكل فترة بعد التقريب لأسفل. ليست ريالات فعلية أو رسوم أدوات أو خصمًا من الدرجة.

**دليل السعة:** الفترة 1: 137/195, الفترة 2: 183/200, الفترة 3: 206/207.

## لماذا اخترت هذه العتبة؟
I selected a threshold of 0.6583471436014694 for the weighted model, with applications flagged when their score is greater than or equal to this threshold. It gives the lowest observed loss among the tested thresholds that satisfy the 12% review limit in every period. It flags 526 of 5,039 applications (10.44%), catches 157 defaults, and produces 369 false positives and 227 missed defaults. Using loss = 10 * FN + FP, the total loss is 2,639 teaching units.

## الخسارة والسعة
The policy assigns 10 loss units to a missed default and 1 unit to a false positive. Without the capacity limit, the threshold of about 0.4486 gives the lowest loss of 2,275 units and 64.06% recall, but flags 1,141 applications and exceeds capacity. The selected threshold reduces flags to 526 and recall to 40.89%, increasing loss by 364 units. It flags 137, 183, and 206 applications against period capacities of 195, 200, and 207, respectively. The default threshold of 0.5 also exceeds capacity: it flags 1,005 applications with a loss of 2,403 units. I therefore accept a higher observed loss to keep the review workload within the stated limits.

## فرق المناطق وما يحتاج إلى مراجعة
At the selected shared threshold, the false-positive rates are 8.09% in central, 8.29% in western, 7.72% in eastern, and 7.64% in other. The highest-to-lowest gap is 0.6484 percentage points, between western and other. These rates measure the proportion of non-defaulting applications incorrectly flagged in each region. The gap is descriptive and has no confidence interval, so it does not establish fairness or equivalent performance across regions. Recall also varies, from 34.62% in western to 48.15% in other.

## حدود النتيجة
The experiment uses synthetic applications and assumed teaching costs, so it cannot establish performance or financial savings on real loans. The threshold was selected and assessed using the same OOF development predictions, which can make its reported loss optimistic. The frozen threshold needs evaluation on untouched later applications. The weighted scores have not been shown to be calibrated probabilities, so a score of 0.6583 should not be interpreted as a 65.83% default probability. Historical capacity compliance also does not guarantee compliance if future application volumes or score distributions change.

OOF تغطي 50.39% من التدريب و100% من الصفوف المؤهلة؛ 4,961 صفًا تمهيديًا بلا تنبؤ. اختيار العتبة وتقدير خسارتها هنا يستخدمان أهدافOOF نفسها؛ هذه نتيجة تطوير لا اختبار نهائي. لم نستخدم التحدي. المقارنة الجغرافية وصفية وليست شهادة عدالة، والأوزان لا تضمن معايرة الدرجات.

## سؤالك الأول: لماذا قد تخدعكAccuracy؟
Only 384 of the 5,039 evaluated applications defaulted, a rate of 7.62%. Predicting no defaults would achieve 92.38% accuracy while missing every default, giving 0% recall and a loss of 3,840 units. Accuracy is dominated by the non-default majority and treats both types of error equally, although this policy assigns missed defaults 10 times the cost of false positives. I therefore assess ranking with average precision and assess the chosen threshold with recall, precision, decision loss, and review capacity.

## سؤالك الثاني: لماذا تختار علىOOF؟
Out-of-fold predictions score each application using a model that was not trained on its validation fold. In this experiment, the three forward folds also exclude validation customers from training and use only training applications whose 90-day outcomes were known before validation began. This reduces optimism caused by evaluating training examples, sharing customers, or using future information. All 5,039 eligible validation applications receive OOF predictions; the 4,961 earlier warm-up applications do not. OOF predictions support threshold development, but choosing the threshold using their labels still requires a separate final evaluation.

أدلتك في `artifacts/threshold_metrics.json` و`day3_period_capacity.csv` و`day3_region_audit.csv` و`day3_cost_sensitivity.csv` و`cost_curve.png`. الحساسية سيناريوهات ±20% لخسارةFN، وليست فترات ثقة. راجع السعة والمعايرة عند تغير البيانات؛ لا تفترض ثباتهما مستقبلًا.
