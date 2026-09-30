# Diabetic retinopathy stage detection

Classifies retinal fundus photographs into the five clinical stages of diabetic
retinopathy using a convolutional neural network with transfer learning.

Computer Vision coursework 1, BSc (Hons) Computer Science, NIBM.

**Live demo:** https://huggingface.co/spaces/gimz12/dr-screening

| Grade | Stage |
|---|---|
| 0 | No diabetic retinopathy |
| 1 | Mild non-proliferative |
| 2 | Moderate non-proliferative |
| 3 | Severe non-proliferative |
| 4 | Proliferative |

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# unzip the Kaggle download anywhere under data/raw/aptos/
.venv/bin/python prepare_data.py            # normalise whatever layout you got
.venv/bin/jupyter notebook notebook.ipynb   # run every cell in order
.venv/bin/python app.py                     # the demo, after training
```

## Dataset

**APTOS 2019 Blindness Detection**: 3,662 real fundus photographs from the
Aravind Eye Hospital in India, each graded 0 to 4 by a clinician.

Download `mariaherrerot/aptos2019` from Kaggle (9 GB, needs only a free account)
and unzip it anywhere under `data/raw/aptos/`.

Kaggle mirrors are packaged differently from one another. That one ships three
label CSVs and three image folders; others ship a single CSV, and some ship no
CSV at all and encode the grade in folder names. `prepare_data.py` reads any of
those and writes one uniform layout:

```
data/raw/aptos/
├── train.csv       # id_code, diagnosis   (every labelled image, merged)
└── all_images/     # symlinks to the real files, one per id_code
```

Symlinks rather than copies, so the 9 GB is not duplicated on disk.

```bash
.venv/bin/python prepare_data.py
```

Optionally also download `mariaherrerot/ddrdataset` into `data/raw/ddr/` and run
`prepare_data.py --root data/raw/ddr` for the external validation section, which
tests whether the model generalises to a different country, population and set
of cameras.

## Trained model (download separately)

Checkpoints are not stored in git (each is about 95 MB). The final model,
`runs/best/best.pt` (ResNet50, pipeline v3, 448 px, trained on cleaned APTOS + DDR),
is hosted with the live demo. Download it into place, then start the demo:

```bash
curl -L -o runs/best/best.pt https://huggingface.co/spaces/gimz12/dr-screening/resolve/main/runs/best/best.pt
```


```bash
.venv/bin/python app.py            # opens http://127.0.0.1:7860
```

## What is in this repository

- `dr/`, `*.py` — the code (each module has a docstring and a runnable self-check)
- `notebook.ipynb` — the executed narrative notebook: every figure and table of the report, in report order
- `runs/` — logs, metrics, figures and per-run `config.json` for every experiment (weights excluded)
- `data/clean`, `data/splits*` — the cleaned label files and the frozen 70/15/15 splits (seed 42); images are not included
- `sample_images/`, `demo_images/` — example photos for the prototype (from the sealed test split)
- `viva_figures/` — figures used in the presentation

## What each file does

| File | Purpose |
|---|---|
| `dr/preprocess.py` | clean a raw photo: circle crop, resize, Ben Graham, CLAHE, denoise |
| `dr/data.py` | stratified splits, dataset class, augmentation, class balancing |
| `dr/model.py` | EfficientNet-B0 or ResNet50 with a new 5-class head, freeze and unfreeze |
| `dr/train.py` | two-phase training loop, early stopping, LR scheduling, checkpoints |
| `dr/evaluate.py` | metrics, confusion matrix, curves, Grad-CAM |
| `notebook.ipynb` | the full narrative and every figure for the report |
| `app.py` | Gradio demo, the prototype to record for the video |
| `prepare_data.py` | normalises any Kaggle mirror layout into one uniform folder |
| `run_training.py` | the seven APTOS training experiments |
| `evaluate_best.py` | picks the APTOS-only winner, TTA and cut points on validation, scores test and DDR |
| `local_calibration.py` | recalibrates the APTOS-only model with a few local DDR photos |
| `domain_experiments.py` | fine-tune and multi-source experiments, ships the final model |
| `external_test.py` | scores every model on a dataset none of them saw (IDRiD, Messidor-2) |

Each module has a runnable self-check:

```bash
.venv/bin/python -m dr.preprocess
.venv/bin/python -m dr.data
.venv/bin/python -m dr.model
.venv/bin/python -m dr.train
.venv/bin/python -m dr.evaluate
.venv/bin/python prepare_data.py --self-check
```

## Method

**Preprocessing.** Images come from different clinics, cameras and years.
Five steps make them comparable: crop away the black border, resize to 256 px,
subtract a blurred copy to remove uneven lighting and sharpen edges (Ben Graham's
method), apply CLAHE to the green channel where blood shows best, and median
blur to remove the speckle the sharpening amplified.

**Augmentation.** Training images are randomly flipped, rotated a full 180
degrees, zoomed 90 to 100 percent and jittered in brightness. A retina is the
same retina upside down, so the grade never changes. Validation and test images
are only resized.

**Class balance.** Grade 0 is about half the dataset and grade 3 about five
percent. A weighted sampler draws rare grades more often; class weights in the
loss are implemented as the alternative and compared in the tuning table.

**Transfer learning.** ImageNet-pretrained backbone with its 1,000-class layer
replaced by a 5-class head. EfficientNet-B0 was the initial choice; ResNet50 was
run as the comparison and won (validation kappa 0.895 vs 0.857), so it is the
model shipped in the demo. Both are built by the same code path. Phase 1 freezes the backbone and
trains the head alone. Phase 2 unfreezes the deepest three stages at a learning
rate ten times smaller than the head's. Batch-norm layers stay frozen, since
batches of 32 fundus images cannot re-estimate statistics computed over 1.2
million photographs.

**Training.** AdamW, cosine annealing, early stopping with patience 5 on
validation quadratic kappa, gradient clipping, dropout 0.3, label smoothing 0.1,
seed 42. The best epoch is checkpointed, not the last.

**Evaluation.** Accuracy alone is misleading here, since always answering
"No DR" scores about 50 percent. Reported instead: per-class precision, recall
and F1; quadratic weighted kappa (the official APTOS metric, which understands
that grades are ordered); a confusion matrix; binary sensitivity and specificity
for the screening use case; Grad-CAM heatmaps; and an external test on DDR.

## Results

Seven runs on identical frozen splits, two pretrained backbones. The winner is
chosen on validation, then evaluated once on the held-out test set.

| Run | Backbone | Head | Image | Dropout | Balancing | Val kappa |
|---|---|---|---|---|---|---|
| resnet50_256 | ResNet50 | classification | 256 | 0.3 | sampler | **0.900** |
| resnet50 | ResNet50 | classification | 224 | 0.3 | sampler | 0.895 |
| resnet50_ordinal | ResNet50 | regression | 224 | 0.3 | sampler | 0.882 |
| effnetb0_256 | EfficientNet-B0 | classification | 256 | 0.3 | sampler | 0.881 |
| effnetb0_drop5 | EfficientNet-B0 | classification | 224 | 0.5 | sampler | 0.867 |
| effnetb0_sampler | EfficientNet-B0 | classification | 224 | 0.3 | sampler | 0.857 |
| effnetb0_weights | EfficientNet-B0 | classification | 224 | 0.3 | class weights | 0.813 |

Test-time augmentation raised the winner's validation kappa from 0.900 to 0.903
and is used in the final model. The ordinal model reached 0.899 with cut points
tuned on validation, still below the winner even with that advantage.

**Winner on the 550 test images** (ResNet50, 256 px, four-view TTA):

| Metric | Value |
|---|---|
| Quadratic kappa | 0.868 |
| Accuracy | 76.7 % |
| Macro F1 | 0.626 |
| Screening sensitivity, healthy vs any DR | 96.1 % |
| Screening specificity | 97.1 % |
| Screening ROC-AUC | 0.991 |

Per-grade recall: No DR 97 %, Mild 55 %, Moderate 53 %, Severe 48 %,
Proliferative 77 %.

**External validation on DDR** (all 12,522 images from 147 Chinese hospitals,
never seen in training, same model and settings):

| Metric | APTOS test | DDR |
|---|---|---|
| Quadratic kappa | 0.868 | 0.587 |
| Accuracy | 76.7 % | 60.3 % |
| Screening sensitivity | 96.1 % | 48.4 % |
| Screening specificity | 97.1 % | 97.1 % |
| Screening ROC-AUC | 0.991 | 0.862 |

The model keeps clearing healthy eyes correctly but misses early disease on the
new population: 93 % of DDR mild and 54 % of DDR moderate eyes are called
"No DR". A clear case of domain shift, and the main limitation of a model
trained on one hospital network.

**Can a new clinic fix it with its own photos?** `local_calibration.py` adjusts
only the decision points (no retraining) using a small random sample of labelled
DDR photos, then scores the rest of DDR. Five random samples per size.

| Local photos | Kappa on rest of DDR |
|---|---|
| none (as shipped) | 0.587 |
| 100 | 0.693 |
| 500 | 0.704 |
| 1,000 | 0.707 |

Screening alarm set from 500 local photos:

| Alarm setting | Sick eyes caught | Healthy eyes cleared |
|---|---|---|
| as shipped | 48.5 % | 97.1 % |
| aim for 70 % | 71.3 % | 87.1 % |
| aim for 80 % | 82.2 % | 71.4 % |
| aim for 90 % | 90.7 % | 48.6 % |

Recalibration recovers about 40 % of the lost agreement, and 100 photos are
almost as good as 1,000. But on this population the model cannot catch most
sick eyes without also flagging many healthy ones. Moving the decision points
is a partial fix; a real fix needs retraining on local or multi-source data.

**Retraining fixes, `domain_experiments.py`.** DDR gets its own frozen 70/15/15
split. Every model is scored on the same de-duplicated held-out sets (514 APTOS,
1,858 DDR, see "Duplicate photographs" below), with four-view TTA.

| Model | APTOS test kappa | DDR test kappa | DDR sensitivity | DDR specificity |
|---|---|---|---|---|
| APTOS only | 0.870 | 0.590 | 48.5 % | 98.3 % |
| A: fine-tuned on 1,000 DDR photos | 0.774 | 0.756 | 82.4 % | 86.7 % |
| **B: trained on APTOS + DDR** | **0.877** | **0.864** | **89.6 %** | **87.6 %** |

Fine-tuning (A) fixes DDR but forgets APTOS: APTOS mild recall falls from 53 %
to 0 %. Training on both from the start (B) matches the APTOS-only model at home
and beats every alternative on DDR. Mild recall rises to 69 % on APTOS and 39 %
on DDR. **But B's APTOS proliferative recall falls from 76 % to 58 %.** Three of
every four grade-4 training photos were DDR, and DDR grade 4 (fibrovascular
membranes, hospital referrals) looks different from APTOS grade 4 (haemorrhages,
exudates, laser scars, screening camps), so B learned mostly the DDR picture.
Averaging the two models was tested on validation and rejected: it recovers
little on APTOS and loses most of the DDR gain. The principled fix is a sampler
balanced by dataset as well as grade. **B is the model shipped in the demo** (`runs/best/best.pt`); the
APTOS-only winner is kept at `runs/best_aptos/best.pt`. Because B trained on
DDR, DDR is no longer an unseen hospital for it.

This matches the literature. On the GDRBench benchmark, adding source datasets
raised unseen-domain AUC from 67.1 to 75.9. RETFound, a retina-specific
foundation model, was considered and not used: with thousands of labelled images
it matches ResNet50 on DR, and a natural-image model generalised better across
datasets in a 2025 comparison.

## Truly unseen hospitals: IDRiD and Messidor-2

Neither dataset was used to train or tune any model. Scored on every photo, with
four-view TTA, by `external_test.py`. No photo is byte-identical to a training photo.

| Model | IDRiD kappa (India, other hospital, 455) | Messidor-2 kappa (France, 1,744) |
|---|---|---|
| APTOS only | **0.804** | **0.490** |
| A: fine-tuned on 1,000 DDR | 0.586 | 0.433 |
| B: APTOS + DDR | 0.756 | 0.460 |

Screening on IDRiD: APTOS-only sensitivity 95 % / specificity 75 %, AUC 0.94.
B: sensitivity 100 % / specificity **26 %**, AUC 0.79: it calls most healthy
IDRiD eyes "mild". On Messidor-2 every model has AUC 0.71–0.73, in line with
published CNN results for APTOS → Messidor-2 transfer (RETFound 0.725, DINOv2-L
0.817 in a 2025 comparison).

Reading: adding DDR made the model better on DDR and **worse on hospitals it had
never seen**. With 77 % of training photos from DDR, the pooled model inherited
DDR's decision boundaries. The country-balanced sampler is the obvious next
experiment. Messidor-2 is hard for every model: its mild and moderate eyes show
few lesions at 512 px (the Kaggle copy is downsized from the 1440–2304 px
originals), its colours are rendered differently, and its labels are the
strictest available (three-specialist adjudication). Messidor-2 photos are
distributed by ADCIS under a registration agreement; the Kaggle copy is a
third-party mirror.

## Pipeline v2

A literature review found that
the v1 enhancement chain is neutral-to-harmful with ImageNet-pretrained CNNs
(Huang et al. 2021: Ben Graham −0.5, CLAHE −0.7 kappa), that the median blur
erases 1–3 px microaneurysms, and that resolution is the largest lever
(256→512 px: +6 kappa). Pipeline v2 therefore:

1. **Cleans** every dataset with `dedup.py` (exact + verified near-duplicates,
   pixel-correlation check on the vessel pattern) and `pipeline_v2.py clean`:
   one photo per duplicate group, groups with conflicting grades removed.

   | Dataset | Raw | Conflicting removed | Extra copies removed | Clean |
   |---|---|---|---|---|
   | APTOS | 3,662 | 79 | 228 | 3,355 |
   | DDR | 12,522 | 6 | 154 | 12,362 |
   | IDRiD | 455 | 6 | 3 | 446 |
   | Messidor-2 | 1,744 | 8 | 7 | 1,729 |

   No photo is shared between datasets, so IDRiD and Messidor-2 are genuinely unseen.
2. **Preprocesses** with crop → pad square → resize 512 only (`preprocess_v2`).
3. **Augments** with GDRNet-style colour jitter (0.4/0.4/0.4/0.05), camera
   artefacts (halo, dark patch, spots), blur and sharpness, p = 0.5 each.
4. **Balances** pooled datasets over (dataset, grade) with soft weights (β = 0.5).
5. **Runs four experiments** that each change one thing: v1 chain @256 on the
   cleaned splits → v2 @256 → v2 @448 → v2 @448 pooled APTOS+DDR. All scored
   on all four hospitals (`runs/v2/summary.csv`).

The demo (`app.py`) now refuses to sound certain: below 50 % top probability it
says "uncertain, refer", and any ≥20 % chance of severe/proliferative disease
adds an urgent-referral alert.

### Pipeline v2 results

All runs trained on the cleaned APTOS split (Run 3 adds cleaned DDR), scored on
every hospital's cleaned test set with four-view TTA (`runs/v2/summary.csv`).

| Run | One change from the run above | APTOS | DDR | IDRiD | Messidor-2 |
|---|---|---|---|---|---|
| 0 | v1 chain @256, cleaned data | 0.881 | 0.528 | **0.785** | 0.495 |
| 1b | drop the enhancement | 0.883 | 0.573 | 0.676 | 0.536 |
| 1 | + strong augmentation | **0.901** | 0.573 | 0.722 | 0.499 |
| 2 | + 448 px | 0.884 | 0.625 | 0.751 | 0.584 |
| 3 | + DDR, balanced by dataset and grade | 0.899 | *0.880 (seen)* | 0.745 | **0.634** |

Quadratic kappa. APTOS, IDRiD and Messidor-2 are unseen by every run except as
labelled; DDR is unseen except by Run 3.

What each change did:

- **Dropping the enhancement** (0 → 1b) was neutral at home, helped on DDR and
  Messidor-2 (+0.04 each) and **hurt on IDRiD (−0.11)**. The published finding
  that enhancement does not help held on three of four hospitals, not all.
- **Strong augmentation** (1b → 1) was mixed: +0.02 at home, +0.05 IDRiD, −0.04 Messidor-2.
- **Resolution** (1 → 2) was the most consistent gain on unseen hospitals:
  DDR +0.05, IDRiD +0.03, Messidor-2 +0.09, with screening AUC up on all three.
- **Balanced pooling** (2 → 3) gave the best Messidor-2 score of any model, fixed
  the proliferative regression of the earlier lopsided pooling (APTOS grade-4
  recall 72.5 %, was 58 %), and is the only model that recognises moderate DR in
  Messidor-2 (73 % recall, others ≤ 13 %).

Per-grade trade-offs: APTOS mild recall rose from 51 % (Run 0) to 75 % (Runs 2
and 3). **Severe recall fell from 84 % to about 50 %** in every v2 run (25 test
photos, so noisy, but consistent). On IDRiD, Run 3 clears only 41 % of healthy
eyes: it over-calls them as mild. The old enhancement chain remains best on IDRiD.

### The photo-shape shortcut, and pipeline v3

In APTOS, field-of-view shape is confounded with grade: 95 % of full-circle
training photos are healthy against 22 % of flat-cut ones, and every IDRiD photo
is flat-cut. A controlled test cut the top and bottom off 181 healthy round test
photos, retina untouched: P(healthy) fell by 0.10 (v1), 0.30 (v2 Run 2) and 0.18
(v2 Run 3, 9 % flipped to diseased). v2's padding preserved the exact outline
and made the shortcut easier. **v3** adds one step, `common_band`: every photo is
cut to the same band (height 0.70 × width), so the outline carries no information.

| Model (v3 unless noted) | APTOS | DDR | IDRiD | Messidor-2 |
|---|---|---|---|---|
| ResNet50, APTOS only, **v2** (Run 2) | 0.884 | 0.625 | 0.751 | 0.584 |
| ResNet50, APTOS only | **0.913** | 0.632 | 0.761 | 0.603 |
| ConvNeXt-Tiny, APTOS only | 0.874 | 0.620 | 0.777 | 0.607 |
| ResNet50, APTOS + DDR, **v2** (Run 3) | 0.899 | 0.880* | 0.745 | 0.634 |
| **ResNet50, APTOS + DDR (final)** | 0.892 | 0.864* | **0.811** | 0.629 |
| ConvNeXt-Tiny, APTOS + DDR | 0.906 | **0.905*** | 0.808 | **0.667** |

\* DDR seen in training. The shape fix lifted the pooled model's IDRiD kappa by
0.066 and healthy-eye specificity there from 41 % to 63 %.

**Contrast and edge enhancement on top of v3 (v4, rule fixed before training).**
v4 = v3 + inscribed circle (for the ~4 % of photos framed to their corners) +
Ben Graham blur-subtraction (edge enhancement, sigma = radius/30) + CLAHE on the
green channel (contrast enhancement) + rim re-mask, no median blur. Rule: v4
replaces v3 if it is not worse on the unseen hospitals or on severe and
proliferative recall; ties go to v4 because it matches the brief's examples.

| ResNet50, APTOS + DDR | APTOS | DDR* | IDRiD | Messidor-2 |
|---|---|---|---|---|
| v3 (final) | 0.892 | 0.864 | 0.811 | 0.629 |
| v4 | 0.894 | 0.850 | 0.816 | **0.694** |
| v3 proliferative recall | 72.5 % | 86.8 % | 58.1 % | 42.9 % |
| v4 proliferative recall | 57.5 % | 82.4 % | 38.7 % | 25.7 % |

v4 was more conservative (healthy-eye specificity up at every hospital, e.g.
IDRiD 63 → 73 %, Messidor-2 81 → 92 %) and gave the best Messidor-2 kappa of any
model, but **lost proliferative recall at all four hospitals**. The APTOS-only
v4 run was also worse than v3 on three of four hospitals. v3 therefore stays the
final pipeline. A plausible mechanism: blur-subtraction is a high-pass filter;
it sharpens small lesions but removes large, low-frequency structure such as
fibrovascular tissue and big haemorrhages, which is what proliferative disease
looks like. The enhancement chain is implemented, visualised and evaluated, and
kept out of the final model on that evidence.

**Backbone decision (rule fixed before training).** ConvNeXt replaces ResNet50
only if it wins on validation (0.898 vs 0.868: yes) **and** is not worse on the
unseen hospitals or on severe/proliferative recall. It tied IDRiD (0.808 vs
0.811) and won Messidor-2, but its severe and proliferative recall was lower on
APTOS (40/65 % vs 48/72.5 %), IDRiD (23/52 % vs 22/58 %) and Messidor-2 (9/37 %
vs 21/43 %). Those are the urgent-referral grades, so **ResNet50 v3 is the final
model** and the demo model (`runs/best/best.pt`). Earlier demo models are kept
at `runs/best_v2_multi` and `runs/best_v1_multi`. This matches Fang et al. (2023):
on APTOS, newer ImageNet architectures give little or no gain over ResNet-50.

## Duplicate photographs

A byte-level hash of every file finds exact duplicates.

| | APTOS | DDR |
|---|---|---|
| Duplicate groups | 123 (128 extra copies) | 95 (98 extra copies) |
| Groups spanning two splits | 63 | 48 |
| Groups with **conflicting grades** | 30 | 0 |
| Test photos with a twin in train/val | 36 of 550 | 21 of 1,879 |

Thirty APTOS photos carry two different grades for the same pixels, which is
direct evidence of grader inconsistency. Removing the leaked test photos moves
kappa by at most +0.005, and the APTOS-only model scores *lower* on the 36
leaked photos (69 % accuracy) than on the rest, so nothing was memorised. All
headline numbers above use the de-duplicated test sets (`test_clean.csv`).

## Reproducing a run

Every experiment writes its own directory:

```
runs/<name>/
├── config.json      every hyperparameter used
├── history.json     per-epoch metrics
├── best.pt          weights from the best epoch
├── results.json     final test metrics
└── figures/         confusion matrix, training curves, Grad-CAM gallery
```

The train/val/test split is written to `data/splits/` once and reused by every
run, so results are directly comparable and the test set is never trained on.

## Not a medical device

A university coursework prototype. Not validated for clinical use and not a
substitute for examination by a clinician.
