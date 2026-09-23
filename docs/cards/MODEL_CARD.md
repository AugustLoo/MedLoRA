# Model Card — MedLoRA adapters for Qwen2.5-VL-3B-Instruct

**Version** 1.0 · 2026-09-22 · Author: Chunqian Loo · Course project, Topic 6 (Task 1.3)
**Repository** https://github.com/AugustLoo/MedLoRA

This card covers the family of LoRA adapters released in this project. All of them patch the same
base model and are used the same way; they differ only in what they were trained on. Numbers quoted
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

Three fixed tables, greedy decoding (`do_sample=False`), at most 32 new tokens, images capped at
512×512. Per-question predictions are stored so every number can be recomputed without re-running
a model.

| Table | Data | What it measures |
|---|---|---|
| Medical VQA | SLAKE test, 1,061 questions (416 closed / 645 open) | primary task |
| General retention | TextVQA validation, fixed 300-question sample (seed 42) | forgetting probe, short-answer |
| General retention (2nd) | MMBench en-dev, fixed 500-question sample (seed 42), letter-choice | forgetting probe, format-insensitive |
| Reliability | PubMedQA `pqa_labeled`, **held-out 500 of 1,000** | calibration over {yes, no, maybe} |

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
| **C1** | 84.13 | 76.12 / 82.19 | 81.33 | **61.09** | 19 / 55 | 26 |
| **C2-300** | **85.82** | 75.97 / 82.40 | 82.33 | 60.48 | **21 / 55** | **14** |
| C2-300-s43 | 85.82 | 77.36 / 84.00 | 82.56 | 59.80 | 18 / 55 | 12 |

**Reading the table.** Instruction tuning is worth about 18 points of closed accuracy and 35 of open
recall. Neither CPT variant adds anything to the primary metric. Mixing three-class text examples
into the SFT set (C1, C2-300) is the only intervention that improves calibration, and the gain
saturates by 300 examples.

**Second retention probe (MMBench, 500 questions; the four server-side models only).** Base 88.40 · A-server 87.40 ·
C2-300 87.40 · C1 88.00. Relative to zero replay the curve is 0.00 / +0.60 — no replay cost. The TextVQA decline is a
short-answer-format effect; see §5.

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
general VQA accuracy monotonically (−1.23 at 300 examples, −2.23 at 900) — a control run attributes
this to the replay data, not to instruction tuning (SFT alone costs −0.66, noise level). At 300
examples the model over-corrects toward "no"/"maybe": "no"→"yes" errors fall 63→14 but "yes"→"no"
errors rise 18→32, so total dangerous flips are unchanged relative to 900 examples (46 vs 45).

**The "maybe" class rests on 55 unique training items.** The balanced replay sample repeats them
roughly 5× (C1) and 2× (C2-300). On the training half C1 scores 94.4 % and answers 54 of 55 "maybe"
questions correctly, so memorisation is demonstrably present; the held-out half shows the effect
transfers, but a larger three-class source would test it properly.

**Mostly single seed.** Every experiment ran once with seed 42, except the 300-example replay point, which was repeated with seed 43. Differences under ±1 point are treated as
noise rather than tested statistically. The A-server control partly substitutes for a second seed on
experiment A — an independent run with different hardware, precision and quantisation reproduced
every metric to within half a point — but the replay-budget comparison (300 vs 900) shows a
swing in "yes" recall between 300 and 900 that needed a second seed. That seed (C2-300-s43) reproduces the
300-example point to within 0.7 macro-F1 and 2.2 points of "yes" recall, so the 8.7-point recall gap to 900 and the
reversed direction of the two dangerous-error counts are properties of the budget, not of a run. The 900-example
point and the zero point remain single runs.

**The retention cost is a format effect, and neither probe covers open-ended output.** TextVQA (300 short-answer
OCR questions) records a replay cost of −1.23 / −2.23; MMBench (500 multiple-choice questions) records 0.00 / +0.60,
within noise. The one-word replay targets perturb the short-answer output distribution and leave letter-choice
ability untouched. So the TextVQA number overstates general forgetting rather than understating it — but both
probes are short-output, and drift in open-ended description remains unmeasured.

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

Reproduce the full evaluation for any adapter:

```bash
bash train/eval_all.sh <tag> <adapter_dir>
python scripts/eval_pubmedqa_split.py --half test     # all models on one ruler
```

`medvlm/prompts.py` holds the exact prompt strings used for every table; changing them changes the
numbers, so they are versioned with the code.
