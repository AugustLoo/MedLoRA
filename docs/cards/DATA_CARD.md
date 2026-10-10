# Data Card — MedLoRA

**Version** 1.6 · 2026-10-10 · Author: Chunqian Loo · Course project, Topic 6 (Task 1.3)
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
| COCO Caption 2017 (val) | **free-text probe** (35B models) | fixed 300-image sample | CC BY 4.0 annotations |
| SLAKE test images | **free-text medical probe** (35B models) | all 96 test images | CC BY 4.0 |
| VQA-RAD (test; train split §4e) | **external medical VQA test** (added 2026-09-30); train split for `interns2_6ds` | 451 test questions; 734 train | CC0 1.0 |
| PathVQA (test; train split §4e) | **external pathology VQA test** (added 2026-09-30); train split for `interns2_6ds` | 6,719 test questions; 5,000 train | MIT |
| MedQA-USMLE, 4 options (test; train split §4e) | **external medical-knowledge test** (added 2026-09-30); train split for `interns2_6ds` | 1,273 test questions; 3,000 train | CC BY 4.0 |
| PneumoniaMNIST 224 (test; train split §4e) | **missed-abnormality probe** (added 2026-09-30); train split for `interns2_6ds` | 624 test images (390 pneumonia / 234 normal); 2,000 train | CC BY 4.0 |
| IU X-Ray (OpenI, Kaggle mirror) | image-text CPT prototype | ≈3,483 frontal image-report pairs | public |
| CheXpert Plus | image-text CPT (B3) | 10,000-study pool, 5,000 per arm | registered; college server only |

**Two hard rules, enforced in code and reviewed before every run:**

1. **SLAKE test and the PubMedQA held-out half are never trained on.** The PubMedQA split is a
   committed file with a regeneration guard; the SLAKE test split is the dataset's own. The four external
   test sets added on 2026-09-30 (Section 4d) are evaluation-only as well. Their train splits are used only by
   `interns2_6ds` (Section 4e), after removing every training item that repeats a test item and, for VQA-RAD,
   every training question on a test-split image.
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
| `pubmedqa_sft_train_100` | 33 | 99 | ≈0.6× (each item at most once) | C2-100 |
| `pubmedqa_sft_train_300` | 100 | 300 | ≈1.8× | C2-300 |
| `pubmedqa_sft_train_300s43` | 100 (seed 43) | 300 | ≈1.8× | C2-300-s43, second seed |
| `pubmedqa_sft_train` | 300 | 900 | ≈5.5× | C1 |
| `pubmedqa_sft_train_900s43` | 300 (seed 43) | 900 | ≈5.5× | C1-s43, second seed |
| (not built) | 600 | 1,800 | ≈10.9× | C2-1800, deprioritised |

**This repetition is the main internal limitation of the replay experiments.** With only 55 unique
"maybe" items, a balanced sample necessarily repeats them. On the training half C1 reaches 94.4 %
accuracy and answers 54 of 55 "maybe" questions correctly, which confirms memorisation is present.
The held-out half shows the calibration effect transfers, but any conclusion about *how far* the
replay budget can be pushed is bounded by this, and is why the 1,800-example point was judged
uninformative rather than simply expensive. The 99-example sample bounds the concern from below: it uses 33 of
the 55 "maybe" items once each, and the resulting model has the highest "maybe" recall of any run (41.8 %), so
restoring the class does not depend on repetition; what the larger budgets add is discrimination between the
classes (accuracy 64.6 → 69.6 → 73.0 at 99 / 300 / 900).

---

## 4. TextVQA (retention probe)

**Source** Singh et al., TextVQA validation split. A **fixed 300-question sample drawn with seed
42** and stored, so every model sees exactly the same questions.

**Role** The general-ability probe: natural images with text in them, scored with the standard VQA
accuracy `min(#matching annotators / 3, 1)`.

**Known weakness, stated plainly.** Short-answer OCR questions are close to the SFT output format,
so it was expected to under-report drift in longer-form ability; the MMBench probe (4b) showed the
reverse for replay — TextVQA over-reports, because the one-word replay targets perturb exactly this
short-answer format. 300 questions also means one accuracy point is three questions. It was sensitive enough to detect the replay cost (−2.23 for C1, clearly
attributed by a control run) but is not sensitive enough to bound that cost tightly. Replacing it
with an MMBench subset or an open-ended description task was the main outstanding measurement gap; the MMBench
probe below now closes half of it.

## 4b. MMBench (second retention probe, added 2026-09-23)

**Source** `lmms-lab/MMBench`, config `en`, split `dev` (the test split has no answers). A **fixed 500-question
sample drawn with seed 42** by `eval/eval_mmbench.py`; every model sees the same questions.

**Role** A format-insensitive general-ability probe: four-way (sometimes two- or three-way) multiple choice over
20 ability dimensions. Scored on the first A–D letter in the output; no circular evaluation (option rotation
would quadruple inference and is unnecessary for a differential probe where every model sees identical inputs).
Per-category accuracy is stored in the summary JSON. Zero unparsed answers across the six models evaluated.

**What it showed** Replay cost −0.20 / −0.20 / +0.40 at 99 / 300 / 900 examples (seed means; all within ±0.6, three questions), against −0.78 / −1.11 / −1.73 on TextVQA. The
TextVQA cost is therefore a short-answer-format perturbation, not general forgetting. Neither probe measures
open-ended generation, which remains the unmeasured case.

**Licence / access** Public on the Hugging Face Hub; fetched through the mirror on the GPU server. Licence per
the dataset card — verify before any redistribution of the sampled subset (none is redistributed here; only
per-question predictions and scores are stored).

## 4c. Free-text probe data (added 2026-09-27, 35B models)

`eval/eval_openended.py` asks for two-to-three-sentence descriptions to measure drift in long-form output, which the
three short-output probes cannot see.

**COCO captions.** `lmms-lab/COCO-Caption2017`, split `val` only (two parquet files, 815 MB; the 6.6 GB `test` split
is never downloaded). A **fixed 300-image sample drawn with seed 42** is saved once to
`$HF_HOME/medlora_coco_val_n300_seed42` and read from there afterwards, so offline runs do not depend on the `datasets`
cache. Each image has five to seven human captions. Scores: CIDEr **without** the CIDEr-D length penalty (the penalty
zeroes out 50-word answers against 10-word references), ROUGE-L, recall of reference content words, mean length,
share of answers under five words, refusal rate. Licence: COCO annotations CC BY 4.0; images under their Flickr terms.

**SLAKE free-text reports.** All **96 SLAKE test images**, each asked for modality, body region and abnormal
findings. No reference text is needed: modality and region come from the image's own annotation. Abnormality is
derived from the same image's closed questions — a "no" to "Is the … healthy / normal?" or a "yes" to "Are there
abnormalities?" marks it abnormal, the opposite answers mark it normal. This yields **44 abnormal, 12 normal and 40
undetermined** images. A description counts as "abnormal called normal" if it claims normality and, after negated
phrases ("no evidence of consolidation or nodules") are removed, mentions no abnormal finding.

**Known weakness, stated plainly.** Over 44 abnormal images and one greedy decoding the false-normal count is too
unstable to compare models: the same zero-replay condition gave 13 and 3 across two seeds, and twelve fine-tuned
models spread from 2 to 15 without relation to calibration. The measure documents that the failure exists; comparing
models on it would need a larger annotated set (for example CheXpert Plus reports), several seeds and human reading.
The keyword judge was checked by hand only on one model's 13 flagged cases. **A later hand check (2026-10-01,
Section 4d) found that this "claims normality" judge misses phrasings such as "no apparent abnormalities" and that the
miss rate differs between models**, so the SLAKE counts above are also undercounts of unknown size.

## 4d. External test sets (added 2026-09-30, approved by the instructor; evaluation only)

Added to answer three questions the in-domain tables cannot: does medical fine-tuning transfer to another
dataset, does the yes/no bias appear on image questions in another domain, and is text-only medical knowledge
kept. All are fetched from the Hugging Face Hub (through the mirror on the server) with **only the test shards
downloaded** (`data_files` pattern plus `verification_mode="no_checks"`); nothing from them is trained on.

| Set | Source | Test size | Format and scoring |
|---|---|---|---|
| VQA-RAD | `flaviagiammarino/vqa-rad` | 451 (251 yes/no, 200 open) | radiology VQA; yes/no gold → closed, same prompt and scoring as SLAKE; open → exact match / token recall / F1 |
| PathVQA | `flaviagiammarino/path-vqa` | 6,719 (3,362 yes/no, 3,357 open) | pathology VQA; as VQA-RAD; closed questions also report predicted-yes rate and yes→no / no→yes counts |
| MedQA-USMLE | `GBaker/MedQA-USMLE-4-options`, file `phrases_no_exclude_test.jsonl` | 1,273 | four-option text questions; first A–D letter in the output, as MMBench |
| PneumoniaMNIST | MedMNIST v2, 224×224 test split | 624 (390 pneumonia / 234 normal) | (1) "Does this chest X-ray show pneumonia? Answer with yes or no only." → sensitivity, specificity, missed pneumonia; (2) free description → does it report an abnormality |

VQA-RAD and PathVQA carry no answer-type field in these Hub versions, so yes/no gold answers define the closed
subset (the LLaVA-Med convention). PneumoniaMNIST images are paediatric and low-resolution; scores are compared
only before versus after fine-tuning, never against other datasets.

**PneumoniaMNIST provenance.** The test split was first extracted from the official `pneumoniamnist_224.npz`
(received from a classmate). Because uploading to the server ran at a few KB/s and Zenodo was equally slow from
the server, the server copy was taken from the Hub mirror `danjacobellis/pneumoniamnist_224` and checked against the
official file: identical image array md5 (`5dbcb024d33649403ec6a610cb0fcc02`) and label md5. Stored as
`data/raw/medmnist/pneumoniamnist_224_test.npz` (not in git).

**Free-text judge, corrected on 2026-10-01.** The first version counted "pneumonia described as normal" with the
`says_normal` judge from Section 4c. A hand check found it misses phrasings such as "no apparent abnormalities" and
"absence of major pulmonary abnormalities", and that the number of misses differs between models (21–99 of 624),
which distorts comparisons. The judge now asks the opposite question — after negated phrases are removed, does the
description still report an abnormality — in two forms (any abnormality; lung abnormality, the primary one because
some descriptions report only an enlarged heart). A hand check of 50 descriptions (25 per verdict, across five
models) found all 50 judged correctly. `scripts/rescore_pneumonia.py` recomputes the metric from stored descriptions.

**Licences / access** (checked on the Hub dataset cards, 2026-10-01). VQA-RAD CC0 1.0; PathVQA MIT; MedQA CC BY 4.0;
MedMNIST CC BY 4.0 (the Hub mirror used on the server states no licence; the official MedMNIST terms apply). Only per-question predictions and scores are stored; no data is redistributed.

---

## 4e. Train splits of the four external sets (S2-6datasets, added 2026-10-08)

At the instructor's request one 35B model (`interns2_6ds`, "S2-6datasets") is trained on six sources: the two used
so far plus the **train splits** of the four sets in §4d. Their **test splits stay evaluation-only** and are used here
only to remove overlapping training examples. Built by `data/convert_sft6.py` (unit tests in `tests/test_convert_sft6.py`)
on the 5090 host, seed 42; prompts are verbatim the evaluation prompts, so a training example and a test question
differ only in content. Counts below are from `results/interns2_6ds_data_2026-10-10.json`.

| Source | Train split | Duplicates removed | Same item as a test item | Rows on test-split images removed | Used |
|---|---|---|---|---|---|
| SLAKE (existing `slake_train.json`) | 4,919 | — | — | — | 4,919 |
| PubMedQA replay (existing, §3c) | 300 | — | — | — | 300 |
| VQA-RAD `data/train-*.parquet` | 1,793 | 0 | 0 | **1,059** | 734 (400 yes/no) |
| PathVQA `data/train-*.parquet` | 19,654 | 1,659 | 0 | 0 | 5,000 sampled (2,712 yes/no) |
| MedQA `phrases_no_exclude_train.jsonl` | 10,178 | 2 | 0 | — | 3,000 sampled |
| PneumoniaMNIST 224, train split | 4,708 (3,477 pneumonia / 1,213 normal) | 18 | 0 | — | 2,000 (1,000 each) |
| **Total** | | | | | **15,953** |

- **Overlap checks.** Images are matched by a pixel hash (same pixels in any mode give the same key), questions by
  lower-cased whitespace-normalised text. "Same item" means same image and question (VQA), same stem (MedQA) or same
  image (PneumoniaMNIST); none were found. Exact duplicates inside a train split are kept once.
- **VQA-RAD shares images between its splits.** Its official split is by question: 202 of the 203 test images also
  appear in the train split, under 1,059 of the 1,793 training questions. Training on those would let the model see
  almost every test image before the test, so all training questions on test-split images are dropped and 734 remain
  (111 images). VQA-RAD therefore stays an unseen-image test, though no longer an unseen-dataset test, for
  `interns2_6ds`. PathVQA's splits share no images.
- **Sampling.** PathVQA and MedQA are capped (5,000 and 3,000) so that, together with PneumoniaMNIST, they do not
  outweigh the 4,919 SLAKE examples, and to keep training to one night; the caps are command-line arguments and can be
  raised. PneumoniaMNIST is class-balanced (the train split is 74 % pneumonia).
- **Storage.** Training files `data/processed/{vqarad,pathvqa,medqa,pneumonia}_sft_train.json` and images
  `data/sft6_images/` on the host, outside git.

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
python data/convert_pubmedqa_sft.py --per-class 33 --tag 100      # C2-100 replay sample
python data/convert_pubmedqa_sft.py --per-class 100 --tag 300s43 --seed 43   # second-seed 300 sample (3B and 35B)
python data/convert_pubmedqa_sft.py --per-class 300 --tag 900s43 --seed 43   # second-seed 900 sample
```

Processed files land in `data/processed/` and register themselves in `dataset_info.json` for
LLaMA-Factory. Raw downloads go to `data/raw/`, which is git-ignored; on the GPU server it is a
symlink to a workspace directory so it does not consume the project quota.

**35B host.** The 35B experiments ran on a separate 8-GPU host. The processed training files and SLAKE images were
copied there unchanged; image paths inside the training files are rewritten at load time (`--path-map`) rather than
edited. The Hugging Face caches for PubMedQA, TextVQA and MMBench were copied from the college server, so every model
on both hosts is scored on byte-identical samples.

---

## 8. CheXpert Plus (experiment B3, used from 2026-10-02)

Used only on the college GPU server (user0 container). It replaces MIMIC-CXR from the original task description:
same image-report structure, lighter access (registration rather than PhysioNet credentialing), and it is the
dataset shared with the Topic 1 teammate whose image-text alignment model supplies the B3 quality scores.

**Source.** `/home/share/chexpert_plus/` (DICOM plus `df_chexpert_plus_240401.csv`, 223,462 rows). The teammate's
15,000-pair subset (`/workspace/rafi/chexpert_plus_subset`, patient-level 10,293 / 1,518 / 3,189) and his code and
checkpoint are used **read-only**; nothing is written under his folder or under `/home/share`.

**B3 candidate pool.** 10,000 studies from 10,000 patients outside the teammate's subset (so none of his test
patients): frontal images only, report = `section_findings` + newline + `section_impression` (his rule), first frontal
image per study, one study per patient, seed 42 (`scripts/build_b3_pool.py`). DICOMs converted with his Chambon
Appendix A pre-processing (his `load_dicom_rgb`, JPEG quality 95, opencv-python-headless 5.0.0.93, pydicom 3.0.2 —
the versions in his environment); 20 of his cached images re-converted this way were pixel-identical. Scored with his
`plus_g_peft_001` model (`scripts/score_chexpert_pairs.py`); B3-top = 5,000 highest-scoring pairs, B3-rand = 5,000 at
random (`data/convert_chexpert_b3.py`), images downscaled to a 1,024-pixel long side for training.

**Compliance.** Manifests, per-pair score files, training JSONs and images contain patient identifiers or report text
and stay on the server, outside git. Only aggregate statistics (`results/b3_chexpert_cpt_2026-10-04.json`), evaluation
results on the public test sets, and adapter weights leave it. Demographic columns of the source table (age, sex,
race, insurance, …) are never read into any derived file.

**Test split for missed abnormalities (Section 5.10 of the report, 2026-10-06/07).** The 3,189 films of the teammate's
`our_split=test` are used for evaluation only, read-only and with his agreement (he keeps the split sealed for his own
model selection; our use does not touch his models). Groups from the CheXbert labels in `pairs_with_text_labels.csv`:
abnormal = any positive finding other than Support Devices (2,733), normal = No Finding positive and nothing else
positive (231), uncertain-only (225). A downscaled copy of the test images (1,024-pixel long side) is kept in
`/workspace/chunqian/data/chexpert_test_1024/` for the evaluation; images go from the container to the model served on the
GPU host only through the machine's internal network and are not stored on the host. Per-image predictions contain
report-derived labels and stay on the server; only aggregate rates (`results/interns2_chexpert_2026-10-07.json`) leave it.
