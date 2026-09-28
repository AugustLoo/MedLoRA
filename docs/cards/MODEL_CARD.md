# Model Card — MedLoRA adapters for Qwen2.5-VL-3B-Instruct and Intern-S2-Preview (35B)

**Version** 1.3 · 2026-09-28 · Author: Chunqian Loo · Course project, Topic 6 (Task 1.3)
**Repository** https://github.com/AugustLoo/MedLoRA

This card covers the LoRA adapters produced in this project: the main family on Qwen2.5-VL-3B-Instruct, and a
scaling check on Intern-S2-Preview (35B). Within each family all adapters patch the same base model and are used the
same way; they differ only in what they were trained on. Numbers quoted
here are reproduced by `train/eval_all.sh <tag> <adapter>` from the stored per-question predictions
in `outputs/eval/`.

---

## 1. Model details

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-VL-3B-Instruct` (3.75 B parameters, Apache 2.0) |
| **Adapter type** | LoRA (`peft`), rank 16, alpha 32, dropout 0.05 |
| **Trainable parameters** | 29,933,568 — **0.79 %** of 3,784,556,544 |
| **Adapter targets** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (LLM linear layers only) |
| **Frozen throughout** | vision tower (`visual.patch_embed`, `visual.blocks`) and multimodal projector (`visual.merger`) |
| **Artefact size** | ≈ 120 MB per adapter (`adapter_model.safetensors`) |
| **Framework** | LLaMA-Factory (training), `transformers` + `peft` (inference) |
| **Language** | English only |
| **Licence** | Adapters released under the base model's licence terms; training data licences in `DATA_CARD.md` |

### Two hardware / precision regimes

Experiments 0, A, B1 and B2 ran on a Kaggle T4 (16 GB) with a **4-bit NF4 quantised** base and fp16
compute. From experiment C1 on, training moved to a college RTX 5090 (32 GB) with an **unquantised
bf16** base. The A-server control (§3) shows the two regimes agree to within half a point on every
metric, so figures from both are read on one axis.

### Second base model: Intern-S2-Preview (35B), added 2026-09-25

To test whether the findings are a small-model effect, the two central conditions (and a set of ablations) were
repeated on a model roughly twelve times larger, provided by the course instructor and served on an 8 × RTX 5090 host.

| | |
|---|---|
| **Base model** | Intern-S2-Preview (continued-pretrained from Qwen3.5; 35B total parameters, mixture of experts with 256 routed experts, 8 active per token, about 3B active) |
| **Adapter type** | LoRA (`peft` 0.21), rank 16, alpha 32, dropout 0.05 — same as 3B |
| **Adapter targets** | language-model attention only: full-attention `q/k/v/o_proj`, linear-attention `in_proj_qkv / in_proj_z / out_proj`, plus the shared expert's `gate/up/down_proj` — 250 layers, **19,169,280 parameters, 0.054 %** |
| **Not adapted** | the 256 routed experts (stored as packed 3D parameters, not linear layers), router, vision encoder, time-series module, embeddings |
| **Framework** | own training script `train/interns2/train_lora.py` (transformers 5.2 + peft; LLaMA-Factory does not support the model), model split layer-wise over 4 GPUs |
| **Chat template** | thinking mode **disabled** in training and evaluation (the model reasons before answering by default, which consumes the short answer budget) |
| **Serving for evaluation** | adapter merged into the weights (`train/interns2/merge_lora.py`, MTP layers and the F32 `lm_head` copied from the base unchanged) and served with LMDeploy (`train/interns2/serve.sh`); evaluated through the same OpenAI-compatible client as the base (`medvlm/remote.py`) |
| **Licence** | base weights provided by the instructor and referenced in place on the host, never copied or redistributed; only adapters are produced |

---

## 2. The adapters

| Tag | Stage(s) | Training data | Precision | Purpose |
|---|---|---|---|---|
| `sft_r16` (**A**) | SFT | SLAKE train 4,919 | T4, 4-bit fp16 | gain from instruction tuning alone |
| `sft_a_server` (**A-server**) | SFT | SLAKE train 4,919 | 5090, bf16 | single-variable control for C1 |
| `cpt_only_b1` | CPT (`stage: pt`) | PubMedQA `pqa_artificial` 10,000 | T4, 4-bit fp16 | what text CPT alone changes |
| `cpt_sft_r16` (**B1**) | CPT → SFT | above, then SLAKE train | T4, 4-bit fp16 | does text CPT transfer to image VQA |
| `cpt_only_b2` | CPT (caption-style) | IU X-Ray 3,483 image-report pairs | T4, 4-bit fp16 | what image-text CPT alone changes |
| `cpt_iu_sft_r16` (**B2**) | CPT → SFT | above, then SLAKE train | T4, 4-bit fp16 | does image-report CPT transfer |
| `sft_mix_pubmedqa_r16` (**C1**) | SFT | SLAKE 4,919 + PubMedQA replay 900 | 5090, bf16 | can the SFT data mix fix the "yes" bias |
| `sft_mix_300` (**C2-300**) | SFT | SLAKE 4,919 + PubMedQA replay 300 | 5090, bf16 | how much replay is enough |
| `sft_mix_300_s43` (**C2-300-s43**) | SFT | same, sampling and training seed 43 | 5090, bf16 | run-to-run variance of the 300 point |
| `sft_mix_100` (**C2-100**) | SFT | SLAKE 4,919 + PubMedQA replay 99 (33 per class) | 5090, bf16 | the point between 0 and 300 where the gain occurs; each "maybe" item seen at most once |
| `sft_mix_900_s43` (**C1-s43**) | SFT | as C1, sampling and training seed 43 | 5090, bf16 | run-to-run variance of the 900 point |

### 35B adapters (Intern-S2-Preview)

All use the SLAKE training set; the replay sample is the same file as at 3B. Seed 43 changes the replay sample and
the training seed together, as for the 3B replications.

| Tag | Replay | Change from the 35B default | Seeds | Purpose |
|---|---|---|---|---|
| `interns2_mix_0`, `interns2_abl_r0_s43` | 0 | — | 42, 43 | SFT alone at 35B |
| `interns2_mix_300`, `interns2_abl_s43` | 300 | — | 42, 43 | the replay fix at 35B (control for the ablations) |
| `interns2_abl_r100`, `…_r100_s43` | 99 | — | 42, 43 | replay budget curve |
| `interns2_abl_r900`, `…_r900_s43` | 900 | — | 42, 43 | replay budget curve |
| `interns2_abl_attn`, `…_attn_s43` | 300 | no LoRA on the shared expert (130 layers, 0.040 %) | 42, 43 | does the shared expert need adapting |
| `interns2_abl_ep1`, `…_ep1_s43` | 300 | 1 epoch instead of 3 | 42, 43 | training length |
| `interns2_abl_r8`, `…_r32` | 300 | rank 8 / 32 | 42 | capacity |
| `interns2_abl_lr5e-5`, `…_lr2e-4` | 300 | learning rate halved / doubled | 42 | step size |

### Hyper-parameters

Identical across every SFT run — this is what makes the comparisons single-variable:

```
lora_rank 16 · lora_alpha 32 · lora_dropout 0.05 · freeze_vision_tower true
learning_rate 1.0e-4 · num_train_epochs 3 · effective batch 16 (4 per device × 4 accumulation)
cutoff_len 1024 · image_max_pixels 262144 (512×512) · seed 42 · cosine schedule, 5 % warmup
```

CPT stages use `learning_rate 5.0e-5` (half the SFT rate, to reduce forgetting) and 1 epoch.
Exact configurations: `configs/*.yaml` (4-bit) and `configs/bf16/*.yaml` (unquantised), generated
from one another by `scripts/make_bf16_configs.py`.

---

## 3. Evaluation

Four fixed tables plus, for the 35B models, a free-text probe; greedy decoding (`do_sample=False`), at most 32 new tokens, images capped at
512×512. Per-question predictions are stored so every number can be recomputed without re-running
a model.

| Table | Data | What it measures |
|---|---|---|
| Medical VQA | SLAKE test, 1,061 questions (416 closed / 645 open) | primary task |
| General retention | TextVQA validation, fixed 300-question sample (seed 42) | forgetting probe, short-answer |
| General retention (2nd) | MMBench en-dev, fixed 500-question sample (seed 42), letter-choice | forgetting probe, format-insensitive |
| Reliability | PubMedQA `pqa_labeled`, **held-out 500 of 1,000** | calibration over {yes, no, maybe} |
| Free text (35B only) | 300 fixed COCO val images + the 96 SLAKE test images, "describe in two or three sentences" | long-form drift; modality / region / abnormal-called-normal in medical descriptions |

### Results

PubMedQA figures are on the held-out half only (gold 276 yes / 169 no / 55 maybe). SLAKE and
TextVQA are full test sets. Rows marked † ran on the T4 in 4-bit fp16; the rest on the 5090 in bf16.

| Model | SLAKE closed | SLAKE open EM / recall | TextVQA | PubMedQA macro-F1 | maybe correct | no→yes errors |
|---|---|---|---|---|---|---|
| Base (T4) † | 67.31 | 40.62 / 46.73 | 83.89 | 48.87 | 7 / 55 | 43 |
| Base (5090) | 66.35 | 40.93 / 47.19 | 84.22 | 48.66 | 7 / 55 | 42 |
| A † | 85.10 | 75.35 / 82.17 | 83.89 | 51.69 | 5 / 55 | 59 |
| A-server | 85.34 | 75.50 / 82.07 | 83.56 | 51.45 | 6 / 55 | 63 |
| B1 † | 83.89 | 75.81 / 82.72 | 83.78 | 52.75 | 3 / 55 | 48 |
| B2 † | 84.13 | 74.57 / 81.61 | 84.00 | 52.35 | 5 / 55 | 54 |
| C2-100 | 85.34 | 76.43 / 82.51 | 82.78 | 57.44 | **23 / 55** | **12** |
| **C2-300** | **85.82** | 75.97 / 82.40 | 82.33 | 60.48 | 21 / 55 | 14 |
| C2-300-s43 | 85.82 | 77.36 / 84.00 | 82.56 | 59.80 | 18 / 55 | 12 |
| **C1** (900) | 84.13 | 76.12 / 82.19 | 81.33 | **61.09** | 19 / 55 | 26 |
| C1-s43 (900) | 84.62 | 77.21 / 83.59 | 82.33 | 60.72 | 12 / 55 | 25 |

**Reading the table.** Instruction tuning is worth about 18 points of closed accuracy and 35 of open
recall. Neither CPT variant adds anything to the primary metric. Mixing three-class text examples
into the SFT set (C2-100, C2-300, C1) is the only intervention that improves calibration; 99 examples buy about
two-thirds of the gain and 300 about nine-tenths, and the per-class balance keeps improving up to 900.

**Second retention probe (MMBench, 500 questions; the six server-side models only).** Base 88.40 · A-server 87.40 ·
C2-100 87.20 · C2-300 87.40 / 87.00 · C1 88.00 / 87.60. Every replay point is within ±0.6 (three questions) of zero replay — no replay cost. The TextVQA decline is a
short-answer-format effect; see §5.

### Results: Intern-S2-Preview (35B)

Same four tables, same prompts and scoring, served through LMDeploy with thinking disabled. Where two seeds exist,
both are shown (seed 42 / seed 43). PubMedQA on the held-out half.

| Model | SLAKE closed | SLAKE open EM | TextVQA | MMBench | PubMedQA macro-F1 | maybe correct | no→yes / yes→no |
|---|---|---|---|---|---|---|---|
| 35B base | 84.38 | 67.13 | 88.78 | 93.60 | 61.83 | 7 / 55 | 20 / 17 |
| replay 0 | 94.47 / 93.75 | 86.36 / 86.98 | 88.22 / 86.89 | 94.40 / 93.40 | 59.51 / 63.23 | 4 / 10 | 24 / 16 · 22 / 16 |
| replay 99 | 93.99 / 93.99 | 85.74 / 87.13 | 87.67 / 87.56 | 93.60 / 93.60 | 65.58 / 64.64 | 27 / 28 | 13 / 17 · 8 / 13 |
| **replay 300** | 94.47 / 93.99 | 86.36 / 87.44 | 86.67 / 86.89 | 94.00 / 94.00 | **67.29 / 65.38** | 22 / 15 | 13 / 13 · 9 / 18 |
| replay 900 | 93.99 / 93.75 | 86.82 / 86.20 | 87.22 / 87.33 | 93.20 / 93.80 | 64.17 / 65.30 | 15 / 18 | 21 / 14 · 15 / 12 |
| attention only | 94.47 / 93.51 | 85.74 / 86.67 | 86.00 / 87.00 | 93.80 / 93.40 | 67.29 / 64.25 | 25 / 16 | 10 / 19 · 13 / 23 |
| 1 epoch | 93.03 / 93.03 | 82.79 / 84.34 | 87.44 / 87.67 | 94.20 / 94.20 | 64.14 / 62.32 | 22 / 10 | 9 / 16 · 13 / 19 |
| rank 8 · rank 32 | 93.27 · 93.27 | 86.98 · 86.51 | 87.44 · 87.11 | 94.20 · 94.00 | 66.28 · 66.31 | 23 · 24 | 15 / 15 · 17 / 12 |
| lr 5e-5 · lr 2e-4 | 92.55 · 95.19 | 85.89 · 86.20 | 86.67 · 86.11 | 94.40 · 94.20 | 64.56 · 66.28 | 21 · 22 | 13 / 18 · 14 / 12 |

**Reading the table.** Fine-tuning lifts SLAKE by about 10 closed and 20 open points, 8-11 points above the
fine-tuned 3B model. The base is as reluctant to answer "maybe" as the 3B base (7 of 55), but short-answer SFT does not
reliably make it worse (the two zero-replay seeds fall either side of the base). Replay 300 raises macro-F1 by 5.0 on
average (61.4 → 66.3) and correct "maybe" from 7 to 18.5, without the 3B over-correction ("yes"→"no" stays at 13-18).
Among training settings only under-training reliably hurts; LoRA on the shared expert shows no detectable benefit.
Every replay budget from 99 to 900 lands within seed spread of the others (means 65.1 / 66.3 / 64.7); 99 examples
over-predict "maybe" (99 and 118 times against 55 gold) and lose about five points of accuracy, so 300 is the
recommended budget. No budget costs SLAKE, TextVQA or MMBench.

**Free-text probe (35B only).** Long-form description does not degrade: every fine-tuned model writes 44-52 words
(base 48) on 300 COCO images, none collapses to a short answer or refuses, and caption similarity is unchanged. See §5
for the medical free-text result.

---

## 4. Intended use

**In scope.** Research and coursework on parameter-efficient adaptation of small vision-language
models to the medical domain: reproducing the reported numbers, ablating the pipeline, or as a
starting point for further adaptation experiments.

**Out of scope, explicitly.** These adapters must not be used for clinical decision support,
diagnosis, triage, or any purpose that affects patient care. They are trained on a few thousand
teaching-style VQA pairs, they hallucinate confidently, and the sections below document specific,
measured failure modes. No clinical validation of any kind has been performed.

---

## 5. Limitations and measured failure modes

**Perception did not improve as much as the headline suggests.** The large SFT gains concentrate on
question types the base model could already perceive but answered in the wrong vocabulary or format
(Organ +61.6 open, Knowledge-graph +50.5, closed Modality/Plane/Colour to 100 %). Genuine perception
questions moved far less: Abnormality-open reaches only 41.5 % and Position-open 57.7 %. Residual
errors are left/right and upper/lower confusions and lesion-category mix-ups.

**The vision tower never learned anything.** It is frozen in every run, so all measured change lives
in the language model. The image-text CPT adapter (B2) writes fluent radiology reports but describes
abnormal chest films as normal — it learned the dominant report template, not the findings. This is
a direct consequence of a caption loss over frozen visual features.

**Instruction tuning makes calibration worse before the data mix makes it better.** Every SFT run
increases willingness to commit: predicted "maybe" collapses and "no"→"yes" errors *rise above the
untuned base* (63 for A-server vs 42 for the base). This was reproduced independently on two
hardware platforms, so it is a property of short-answer SLAKE data, not of a particular run.

**The calibration fix has a measured cost, and it over-corrects at small budgets.** Replay lowers
TextVQA accuracy with budget (seed means −0.78 at 99 examples, −1.11 at 300, −1.73 at 900; adjacent points
are within seed spread, the end points are not) — a control run attributes this to the replay data, not to
instruction tuning (SFT alone costs −0.66, noise level). Small budgets over-correct toward "no"/"maybe":
"yes" recall is 60.9 at 99, 71.0 at 300 and 82.1 at 900 (seed means) against 91.7 with no replay, and
"yes"→"no" errors are 36 / 31 / 20 against 18. The total of the two dangerous error types is the same at every
replay budget (42–48 against 81); only the split moves. At 900 both seeds also pay about one point of SLAKE
closed accuracy (84.13 / 84.62 against ≥ 85.34 elsewhere).

**The "maybe" class rests on 55 unique training items.** The balanced replay sample repeats them
roughly 5× (C1) and 2× (C2-300). On the training half C1 scores 94.4 % and answers 54 of 55 "maybe"
questions correctly, so memorisation is demonstrably present; the held-out half shows the effect
transfers, but a larger three-class source would test it properly. The 99-example point bounds the concern: each
"maybe" item is seen at most once there, yet "maybe" recall is the highest of any run (41.8 %), so restoring the
class does not rest on repetition; the larger budgets add discrimination (accuracy 64.6 → 69.6 → 73.0), not the class.

**Two seeds at the replay end points, one everywhere else.** Experiments 0, A, B1, B2 and the 99-example point ran
once with seed 42; the 300- and 900-example points were repeated with sampling and training seed 43. Differences
under ±1 point are treated as noise rather than tested statistically. The A-server control partly substitutes for
a second seed on experiment A — an independent run with different hardware, precision and quantisation reproduced
every metric to within half a point. Between the two seeds the replay points differ by under 0.7 macro-F1, up to
4.7 points of single-class recall, up to 7 "maybe" questions and up to 1 point of TextVQA, and the 300-versus-900
gap in "yes" recall (at least 7.6 points) exceeds that spread; with the 99 point the swing toward "no" is monotone
in budget across four runs, so it is a property of the budget, not of a run.

**The retention cost is a format effect, and neither probe covers open-ended output.** TextVQA (300 short-answer
OCR questions) records a replay cost of −1.23 / −2.23; MMBench (500 multiple-choice questions) records 0.00 / +0.60,
within noise. The one-word replay targets perturb the short-answer output distribution and leave letter-choice
ability untouched. So the TextVQA number overstates general forgetting rather than understating it. Both probes are short-output;
the free-text probe added later covers long-form description for the 35B models only.

**35B: single runs are unreliable.** Two seeds of the same 35B condition differ by up to 3.7 macro-F1 and 12 correct
"maybe" answers (3B: about 0.7 macro-F1 and 3-7 answers). A first-seed result that short-answer SFT pushes the 35B model toward "yes"
did not survive the second seed; only differences confirmed across two seeds are reported as findings.

**Every model describes some abnormal studies as normal.** Asked to describe the 96 SLAKE test images in free text,
the 35B base calls 4 of the 44 abnormal studies normal and the twelve fine-tuned 35B models call between 2 and 15 of
them normal (examples include an enlarged cardiac silhouette described as "heart size normal, lung fields clear").
The count varies more between seeds of one condition (13 and 3 at zero replay) than between conditions, so no model
can be said to be safer than another on this measure — but the failure itself is real and is the clinically most
dangerous one. The 3B adapters were not given this probe.

**Thinking mode is off for the 35B model.** It was disabled to make the answers comparable with 3B and to fit the
short-answer budget. The model's behaviour with reasoning enabled is unmeasured.

**English only, three modalities.** SLAKE covers X-Ray, CT and MRI. Nothing here says anything about
ultrasound, pathology, dermatology, or non-English clinical text.

---

## 6. Ethical and compliance notes

- SLAKE test and PubMedQA `pqa_labeled` are **evaluation-only** except for the 500-question PubMedQA
  training half, which is fixed in `data/pubmedqa_split.json`, committed to git, and protected by a
  script that refuses to regenerate it. The held-out half has never been trained on.
- Controlled-access data (MIMIC-CXR, CheXpert Plus) and any patient-level derived files never enter
  git or public Kaggle datasets. IU X-Ray is used from its public mirror.
- No patient identifiers are present in any released artefact. The adapters contain weights only.
- The Intern-S2-Preview base weights were provided by the course instructor for this study, referenced in place on
  the host and never copied, redistributed or modified; merged evaluation copies live only in the project's own
  directory on that host and are deleted once evaluated.
- Training data is public teaching material, not a representative clinical population; performance
  on any real patient distribution is unknown and untested.

---

## 7. How to use

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

base = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, dtype="bfloat16", device_map="cuda:0")
model = PeftModel.from_pretrained(model, "outputs/sft_mix_300")     # any adapter from §2
processor = AutoProcessor.from_pretrained(base, max_pixels=262144)
```

**35B adapters.** The routed experts are packed parameters, so load the base with its own code and attach the adapter,
or merge and serve it as the evaluation did:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
base = "/path/to/Intern-S2-Preview"
model = AutoModelForImageTextToText.from_pretrained(base, trust_remote_code=True, dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "outputs/interns2_mix_300")
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
# build prompts with processor.apply_chat_template(..., enable_thinking=False)
```

```bash
python train/interns2/merge_lora.py --adapter outputs/interns2_mix_300 --out merged/interns2_mix_300
bash train/interns2/serve.sh start merged/interns2_mix_300          # LMDeploy on port 23334
MEDVLM_API_BASE=http://127.0.0.1:23334/v1 MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' \
  MODEL=merged/interns2_mix_300 bash train/eval_all.sh interns2_mix_300
python eval/eval_openended.py --model merged/interns2_mix_300 --tag interns2_mix_300   # free-text probe
```

Reproduce the full evaluation for any adapter:

```bash
bash train/eval_all.sh <tag> <adapter_dir>
python scripts/eval_pubmedqa_split.py --half test     # all models on one ruler
```

`medvlm/prompts.py` holds the exact prompt strings used for every table; changing them changes the
numbers, so they are versioned with the code.
