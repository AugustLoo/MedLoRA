# 对比示例题：原版 vs 训练版

给做对比界面的同学。10 道题都来自公开测试集，**训练版答对、原版答错**，适合演示两个模型的差别。

- 「原版」= `intern-s2-base`，「训练版」= `intern-s2-medlora`（交给世龙部署的那个）。
- 下面的回答是我们评测时记录的（关闭思考模式、temperature 0）。用对比接口按同样的提问重问，答案应该一样或几乎一样。
- **提问要用「完整提问」那一栏的原文**，包括末尾那句回答要求（例如 *Answer with yes or no only.*）。改了措辞，答案可能不同。
- 这些题是**挑出来的**。整体成绩见 `docs/COMPARE_API.md` 第二节：训练版在医学看图问答和「不确定」题上明显更好，在 MedQA 上整体反而略低（85.0 → 83.6）。
- 批量重跑：`python scripts/compare_ask.py batch docs/compare_examples/examples.jsonl`（文件里多出来的字段会被忽略）。

## 图文题（5 道，SLAKE 测试集）

### img1 · 看胸片诊断：原版说成肺不张，训练版答对气胸

- 来源：SLAKE 测试集 qid 12070 (X-Ray)
- 图片（服务器）：`/home/ubuntu/chunqian/data/SLAKE/imgs/xmlab185/source.jpg`
- 完整提问：

```
What diseases are included in the picture?
Answer with a single word or short phrase.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| Pneumothorax | Atelectasis ❌ | Pneumothorax ✅ |

### img2 · 认器官：图像左边是病人的右侧，原版答成肝，训练版答对右肺

- 来源：SLAKE 测试集 qid 12218 (CT)
- 图片（服务器）：`/home/ubuntu/chunqian/data/SLAKE/imgs/xmlab253/source.jpg`
- 完整提问：

```
What organ is the black part on the left of the image?
Answer with a single word or short phrase.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| Right Lung | Liver ❌ | Right Lung ✅ |

### img3 · 数异常：原版数成 1 种，训练版答对 2 种

- 来源：SLAKE 测试集 qid 12620 (MRI)
- 图片（服务器）：`/home/ubuntu/chunqian/data/SLAKE/imgs/xmlab436/source.jpg`
- 完整提问：

```
How many kinds of abnormalities are there in this image?
Answer with a single word or short phrase.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| 2 | 1 ❌ | 2 ✅ |

### img4 · 判断健康：原版说肺是健康的，训练版答对「不健康」

- 来源：SLAKE 测试集 qid 12029 (X-Ray)
- 图片（服务器）：`/home/ubuntu/chunqian/data/SLAKE/imgs/xmlab159/source.jpg`
- 完整提问：

```
Is the lung healthy?
Answer with yes or no only.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| No | Yes ❌ | No ✅ |

### img5 · 判断有没有某个器官：原版说有食管，训练版答对「没有」

- 来源：SLAKE 测试集 qid 12582 (CT)
- 图片（服务器）：`/home/ubuntu/chunqian/data/SLAKE/imgs/xmlab427/source.jpg`
- 完整提问：

```
Is there an esophagus in this image?
Answer with yes or no only.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| No | Yes ❌ | No ✅ |

## 纯文字题（5 道）

PubMedQA 题先给一段论文摘要（Context），再问问题，要把下面整段一起复制去问。

### txt1 · 证据不足时敢说「不确定」：原版答 yes

- 来源：PubMedQA 考卷半边 pubid 17076091
- 问题：*Does obstructive sleep apnea affect aerobic fitness?*
- 完整提问：

```
You are a biomedical research assistant.
Context:
We sought to determine whether patients with obstructive sleep apnea (OSA) had an objective change in aerobic fitness during cycle ergometry compared to a normal population. The most accurate test of aerobic fitness is measurement of maximum oxygen consumption (VO2max) with cycle ergometry.
We performed a retrospective cohort analysis (247 patients with OSA) of VO2max from annual cycle ergometry tests compared to a large control group (normative data from 1.4 million US Air Force tests) in a tertiary care setting.
Overall, individuals with OSA had increased VO2max when compared to the normalized US Air Force data (p<.001). Patients with an apnea-hypopnea index of greater than 20 demonstrated a decreased VO2max as compared to normalized values (p<.001). No differences in VO2max were observed after either medical or surgical therapy for OSA.

Question: Does obstructive sleep apnea affect aerobic fitness?
Based only on the context, answer with exactly one word: yes, no, or maybe.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| maybe | yes ❌ | maybe ✅ |

### txt2 · 证据不足时敢说「不确定」：原版答 yes

- 来源：PubMedQA 考卷半边 pubid 12920330
- 问题：*Do somatic complaints predict subsequent symptoms of depression?*
- 完整提问：

```
You are a biomedical research assistant.
Context:
Evidence suggests substantial comorbidity between symptoms of somatization and depression in clinical as well as nonclinical populations. However, as most existing research has been retrospective or cross-sectional in design, very little is known about the specific nature of this relationship. In particular, it is unclear whether somatic complaints may heighten the risk for the subsequent development of depressive symptoms.
We report findings on the link between symptoms of somatization (assessed using the SCL-90-R) and depression 5 years later (assessed using the CES-D) in an initially healthy cohort of community adults, based on prospective data from the RENO Diet-Heart Study.
Gender-stratified multiple regression analyses revealed that baseline CES-D scores were the best predictors of subsequent depressive symptoms for men and women. Baseline scores on the SCL-90-R somatization subscale significantly predicted subsequent self-reported symptoms of depressed mood 5 years later, but only in women. However, somatic complaints were a somewhat less powerful predictor than income and age.

Question: Do somatic complaints predict subsequent symptoms of depression?
Based only on the context, answer with exactly one word: yes, no, or maybe.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| maybe | yes ❌ | maybe ✅ |

### txt3 · 证据不足时敢说「不确定」：原版答 yes

- 来源：PubMedQA 考卷半边 pubid 22954812
- 问题：*Are bipolar disorders underdiagnosed in patients with depressive episodes?*
- 完整提问：

```
You are a biomedical research assistant.
Context:
Recent reports indicate that the prevalence of bipolar disorder (BD) in patients with an acute major depressive episode might be higher than previously thought. We aimed to study systematically all patients who sought therapy for major depressive episode (MDE) within the BRIDGE study in Germany, reporting on an increased number (increased from 2 in the international BRIDGE report to 5) of different diagnostic algorithms.
A total of 252 patients with acute MDE (DSM-IV confirmed) were examined for the existence of BD (a) according to DSM-IV criteria, (b) according to modified DSM-IV criteria (without the exclusion criterion of 'mania not induced by substances/antidepressants'), (c) according to a Bipolarity Specifier Algorithm which expands the DSM-IV criteria, (d) according to HCL-32R (Hypomania-Checklist-32R), and (e) according to a criteria-free physician's diagnosis.
The five different diagnostic approaches yielded immensely variable prevalences for BD: (a) 11.6; (b) 24.8%; (c) 40.6%; (d) 58.7; e) 18.4% with only partial overlap between diagnoses according to the physician's diagnosis or HCL-32R with diagnoses according to the three DSM-based algorithms.

Question: Are bipolar disorders underdiagnosed in patients with depressive episodes?
Based only on the context, answer with exactly one word: yes, no, or maybe.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| maybe | yes ❌ | maybe ✅ |

### txt4 · 医师考试病例题：原版选 B，训练版选对 D

- 来源：MedQA 测试集第 300 题
- 完整提问：

```
A 5-year-old boy is brought to the physician because of an irregular gait 3 days after receiving age-appropriate vaccinations. Examination of the lower extremities shows no redness or swelling. When the child stands on his right leg, his left leg drops and his pelvis tilts towards the left. Sensation to light touch is normal in both legs. This patient's symptoms are most likely due to the injection of the vaccine into which of the following locations?
A. Inferolateral quadrant of the right buttock
B. Inferomedial quadrant of the right buttock
C. Inferomedial quadrant of the left buttock
D. Superomedial quadrant of the right buttock
Answer with the option's letter from the given choices directly.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| D. Superomedial quadrant of the right buttock | B ❌ | D ✅ |

### txt5 · 医师考试病例题：原版选 D，训练版选对 A

- 来源：MedQA 测试集第 477 题
- 完整提问：

```
A 32-year-old man comes to the physician because of a 2-day history of a tingling sensation in his right forearm. He reports that his symptoms started after he lifted heavy weights at the gym. Physical examination shows loss of sensation on the lateral side of the right forearm. Sensation over the thumb is intact. Range of motion of the neck is normal. His symptoms do not worsen with axial compression or distraction of the neck. Further examination of this patient is most likely to show weakness of which of the following actions?
A. Elbow flexion
B. Forearm pronation
C. Index finger flexion
D. Wrist extension
Answer with the option's letter from the given choices directly.
```

| 正确答案 | 原版 | 训练版 |
|---|---|---|
| A. Elbow flexion | D ❌ | A ✅ |
