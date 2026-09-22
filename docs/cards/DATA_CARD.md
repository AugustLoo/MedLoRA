# Data Card — MedLoRA

**Version** 1.0 · 2026-09-22 · Author: Chunqian Loo · Course project, Topic 6 (Task 1.3)
**Repository** https://github.com/AugustLoo/MedLoRA

Every dataset used for training or evaluation in this project, how it was obtained, what was done to
it, and the rules that govern it. Each row of the pipeline is reproducible from a script in `data/`.

---

## 1. Overview

| Dataset | Role | Size used | Licence / access |
|---|---|---|---|
| SLAKE (English) | SFT training; validation loss; **medical VQA test set** | 4,919 / 1,053 / 1,061 | CC BY 4.0 |
| PubMedQA `pqa_artificial` | text CPT corpus | 10,000 documents | MIT |
| PubMedQA `pqa_labeled` | **reliability evaluation**; from C1 on, also a three-class SFT source | 1,000, split 500 / 500 | MIT |
| TextVQA validation | **general-ability retention probe** (short-answer) | fixed 300-question sample | CC BY 4.0 |
| MMBench (en, dev) | **general-ability retention probe** (multiple-choice) | fixed 500-question sample | see dataset card |
| IU X-Ray (OpenI, Kaggle mirror) | image-text CPT prototype | ≈3,483 frontal image-report pairs | public |
| CheXpert Plus | image-text CPT at scale (planned, B3) | 5k → 20k studies | registered; download pending |

**Two hard rules, enforced in code and reviewed before every run:**

1. **SLAKE test and the PubMedQA held-out half are never trained on.** The PubMedQA split is a
   committed file with a regeneration guard; the SLAKE test split is the dataset's own.
2. **Controlled-access data and patient-level derived files never enter git or a public Kaggle
   dataset.** This governs MIMIC-CXR and CheXpert Plus. Only aggregate metrics leave the machine.

---

## 2. SLAKE (English subset)

**Source** Bo Liu et al., *SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical
Visual Question Answering*. Downloaded by `data/download_slake.py`; converted by
`data/convert_slake_sharegpt.py`.

**Content** Radiology images (X-Ray, CT, MRI) with English question-answer pairs, annotated by
question type (Organ, Abnormality, Position, Modality, Knowledge-graph, Quantity, Plane, Colour,
Size) and by CLOSED / OPEN answer format.

**Splits used** train 4,919 (SFT) · validation 1,053 (validation loss during training) · test 1,061
(the medical VQA table). The test split is the dataset's own; we never re-partitioned it.

**Preprocessing** Converted to ShareGPT message format, one image per question, images capped at
512×512 (`image_max_pixels: 262144`) to match the evaluation-time cap. Text truncated at 1,024
tokens. No answer normalisation at training time; normalisation happens only in scoring.

**Composition of the test split** 416 closed / 645 open. **61 of the 416 closed questions do not
have a yes/no gold answer** — they have a closed-vocabulary answer such as *Lung*, *Liver*, *T2* or
*Coronal Plane*. This caused a scoring bug that inflated every instruction-tuned adapter by 4–5
points until it was found and corrected on 2026-09-19; see §6.

**Known limitations** Teaching-style questions with short answers; MRI-heavy training distribution
(which is why MRI improves most after fine-tuning); no free-text reporting; three modalities only.

---

## 3. PubMedQA

Two configurations are used for two completely different purposes. Keeping them straight matters.

### 3a. `pqa_artificial` — text CPT corpus

**Source** Jin et al., PubMedQA. Downloaded by `data/download_pubmedqa.py`, converted by
`data/convert_pubmedqa_cpt.py --max 10000`.

**Content** Biomedical abstracts with machine-generated labels. **Only the context text is used** —
the labels are discarded. The CPT stage is plain next-token prediction (`stage: pt`) over 10,000
abstract-level documents, packed to 1,024 tokens.

**Why abstracts** This is the cheapest available in-domain text corpus. Its failure to transfer to
image questions (experiment B1) is itself a reported result, not an oversight.

### 3b. `pqa_labeled` — reliability evaluation, and (from C1) a training source

**Source** Same, expert-labelled subset: 1,000 questions with gold labels in {yes, no, maybe} and a
context passage. Label distribution 552 yes / 338 no / 110 maybe.

**The split protocol.** Experiment C1 needed three-class text examples, and PubMedQA is the only
source of them in this project — but it is also the reliability evaluation set. Training on the
evaluation set would make the scores meaningless, so the set was split before it was used for
anything else:

- `data/split_pubmedqa.py` draws a **label-stratified 500 / 500 split with seed 42**
- the result is written to `data/pubmedqa_split.json` and **committed to git**
- the script **refuses to run if that file already exists**, so the split can never drift
- training half: 276 yes / 169 no / **55 maybe** · held-out half: the same distribution

| Half | yes | no | maybe | Use |
|---|---|---|---|---|
| train | 276 | 169 | 55 | source for the C1 / C2 replay samples |
| test | 276 | 169 | 55 | reliability table — **never trained on** |

**Comparability across experiments.** Models trained before the split existed were scored on all
1,000 questions. Rather than re-running them, `scripts/eval_pubmedqa_split.py` filters each model's
stored per-question predictions to the held-out half and recomputes the metrics, so every model is
compared on one ruler at zero additional GPU cost.

### 3c. The replay samples (C1, C2)

Built by `data/convert_pubmedqa_sft.py --per-class N --tag <name>` from the **training half only**,
using the identical prompt string that evaluation uses (`medvlm.prompts.PUBMEDQA`), with the gold
label as the target. Class-balanced by sampling with replacement:

| Sample | `--per-class` | Total | maybe repetition | Used by |
|---|---|---|---|---|
| `pubmedqa_sft_train_300` | 100 | 300 | ≈1.8× | C2-300 |
| `pubmedqa_sft_train` | 300 | 900 | ≈5.5× | C1 |
| (not built) | 600 | 1,800 | ≈10.9× | C2-1800, deprioritised |

**This repetition is the main internal limitation of the replay experiments.** With only 55 unique
"maybe" items, a balanced sample necessarily repeats them. On the training half C1 reaches 94.4 %
accuracy and answers 54 of 55 "maybe" questions correctly, which confirms memorisation is present.
The held-out half shows the calibration effect transfers, but any conclusion about *how far* the
replay budget can be pushed is bounded by this, and is why the 1,800-example point was judged
uninformative rather than simply expensive.

---

## 4. TextVQA (retention probe)

**Source** Singh et al., TextVQA validation split. A **fixed 300-question sample drawn with seed
42** and stored, so every model sees exactly the same questions.

**Role** The general-ability probe: natural images with text in them, scored with the standard VQA
accuracy `min(#matching annotators / 3, 1)`.

**Known weakness, stated plainly.** Short-answer OCR questions are close to the SFT output format,
so this probe under-reports drift in longer-form ability. 300 questions also means one accuracy
point is three questions. It was sensitive enough to detect the replay cost (−2.23 for C1, clearly
attributed by a control run) but is not sensitive enough to bound that cost tightly. Replacing it
with an MMBench subset or an open-ended description task was the main outstanding measurement gap; the MMBench
probe below now closes half of it.

## 4b. MMBench (second retention probe, added 2026-09-23)

**Source** `lmms-lab/MMBench`, config `en`, split `dev` (the test split has no answers). A **fixed 500-question
sample drawn with seed 42** by `eval/eval_mmbench.py`; every model sees the same questions.

**Role** A format-insensitive general-ability probe: four-way (sometimes two- or three-way) multiple choice over
20 ability dimensions. Scored on the first A–D letter in the output; no circular evaluation (option rotation
would quadruple inference and is unnecessary for a differential probe where every model sees identical inputs).
Per-category accuracy is stored in the summary JSON. Zero unparsed answers across the four models evaluated.

**What it showed** Replay cost 0.00 / +0.60 at 300 / 900 examples, against −1.23 / −2.23 on TextVQA. The
TextVQA cost is therefore a short-answer-format perturbation, not general forgetting. Neither probe measures
open-ended generation, which remains the unmeasured case.

**Licence / access** Public on the Hugging Face Hub; fetched through the mirror on the GPU server. Licence per
the dataset card — verify before any redistribution of the sampled subset (none is redistributed here; only
per-question predictions and scores are stored).

---

## 5. IU X-Ray (image-text CPT prototype)

**Source** Indiana University chest X-ray collection (OpenI), public Kaggle mirror. Converted by
`data/convert_iu_xray.py`.

**Content used** ≈3,483 frontal images paired with their non-empty radiology reports (Findings +
Impression). Split 95/5 **by report id**, so the two views of one study never straddle the split.

**Format** Caption-style pairs trained with `stage: sft` (an image plus an instruction to write the
report). This is a caption objective, not next-token prediction, which is why the config says `sft`
despite the stage being conceptually CPT — the comment in the config says so.

**Composition bias, and why it matters.** **36 % of the training reports describe a normal study**,
and abnormal reports are written largely as negations ("no pleural effusion", "no pneumothorax").
A caption loss over a frozen vision tower therefore rewards reproducing the dominant template. The
resulting adapter writes fluent, well-structured reports that describe abnormal films as normal —
this is a measured finding in the report, not a bug in the conversion.

**Licence / access** Public collection, used from its Kaggle mirror. No patient identifiers are
present in the released artefacts; only the derived training JSON (report text + image paths) is
used, and it stays out of git.

---

## 6. Scoring code and one correction

Metrics live in `medvlm/metrics.py`; prompts in `medvlm/prompts.py`. Both are versioned because
changing either changes the numbers.

| Table | Metric |
|---|---|
| SLAKE closed | yes/no agreement when the gold is yes/no, **exact match otherwise** (`closed_score`) |
| SLAKE open | exact match, token recall (LLaVA-Med convention), token F1 — after lower-casing and stripping punctuation and articles |
| TextVQA | VQA accuracy, `min(#matching annotators / 3, 1)` |
| PubMedQA | accuracy, macro-F1 over {yes, no, maybe}, predicted-label distribution vs gold |

**Correction of 2026-09-19.** The first implementation folded both prediction and gold through a
yes/no/other mapping. On the 61 closed questions whose gold is a content word, any non-yes/no answer
collapsed to "other" and matched the gold — scoring as correct for free. This inflated every
instruction-tuned adapter (which had learned to answer those questions with a content word) by 4–5
points; the zero-shot base was unaffected because it forced yes/no answers there. `closed_score`
fixes it, and `scripts/rescore_slake.py` recomputed every historical result from the stored
per-question predictions without re-running a model. Experiment A moved 89.18 → 85.10, B1
88.70 → 83.89, B2 88.70 → 84.13. **No conclusion changed.** All numbers in the report and in
`MODEL_CARD.md` use the corrected scorer.

---

## 7. Reproducing the data pipeline

```bash
python data/download_slake.py    && python data/convert_slake_sharegpt.py
python data/download_pubmedqa.py && python data/convert_pubmedqa_cpt.py --max 10000
python data/convert_iu_xray.py
# the split already exists in git and will refuse to regenerate:
python data/split_pubmedqa.py
python data/convert_pubmedqa_sft.py --per-class 100 --tag 300     # C2-300 replay sample
python data/convert_pubmedqa_sft.py --per-class 300               # C1 replay sample
```

Processed files land in `data/processed/` and register themselves in `dataset_info.json` for
LLaMA-Factory. Raw downloads go to `data/raw/`, which is git-ignored; on the GPU server it is a
symlink to a workspace directory so it does not consume the project quota.

---

## 8. Planned: CheXpert Plus (experiment B3)

Registered but not yet downloaded. It replaces MIMIC-CXR from the original task description because
it offers the same image-report structure with a lighter access process (registration rather than
PhysioNet credentialing), and because it is the dataset shared with the Topic 1 teammate whose
image-text alignment model will supply the data-quality filtering scores for B3.

**Compliance, decided in advance.** CheXpert Plus is controlled-access. It will be downloaded only
to the college GPU server, never to a laptop or a Kaggle dataset; no image, report, or patient-level
derived file will enter git; only aggregate metrics and trained adapter weights leave the machine.
The 36 %-normal composition problem found in IU X-Ray (§5) is the reason B3 is specified as
*sampled by finding* rather than taken as-is.
