# Data and preprocessing

## GDSC and TCGA

The processed expression, annotation, and response tables originate from the
[THERAPI data release](https://github.com/Sunginyoung/THERAPI). Download the
files from the [THERAPI Google Drive folder](https://drive.google.com/drive/folders/1rnYXYwwqwfqS-q50D68EnYjkUUfw2an2?usp=drive_link)
and run:

```bash
python scripts/prepare_therapi_data.py --source downloads/THERAPI/data
```

The installer copies the required tables into `data/THERAPI`. GDSC provides
supervised cell-line-drug response pairs.
TCGA unlabeled tumors are used for adversarial domain adaptation, and the 383
labeled patient-drug pairs are reserved for final evaluation.

Expression matrices are aligned to the shared 978-gene order. TCGA sample IDs
are reduced to patient-level identifiers before duplicate samples are averaged.
Drug structures are represented by canonical SMILES.

## CSG2A features

`data/THERAPI_CSG2A_zscore` contains row-aligned float16 caches for the
CSG2A-inferred post-treatment expression and chemical embeddings. The TCGA
evaluation caches and gene-order file are distributed in
[`PANDA_release_assets_v1.zip`](https://drive.google.com/file/d/1SJTRH_jq1zuD1pW3fm0XWe3FPYNPaEIh/view?usp=sharing).
The large GDSC caches needed for retraining are distributed separately in
[`PANDA_optional_training_assets_v1.zip`](https://drive.google.com/file/d/1Qmb-ziv--KKTcx5rxmmdLyfbgEplJE0K/view?usp=sharing).
The input
expression is sample-wise z-normalized in the CSG2A gene order. The frozen
CSG2A model uses dose 0.1 and time 1.0, corresponding to 10 micromolar and
72 hours under the original scaling.

These PANDA-specific caches were regenerated after sample-wise z-normalization
and are not byte-identical to the perturbation arrays distributed with THERAPI.
They must therefore be distributed as PANDA artifacts or regenerated with the
documented model and row order.

The upstream CSG2A Landmark pretrained weight is optional for the documented
checkpoint-based evaluation. It is available from the
[CSG2A release link](https://drive.google.com/file/d/17-fFIxXB1avNZ9gWsccZ7Ia5Myqfzi9M/view)
and should be placed at
`external/CSG2A_asg_minimal/ckpts/CSG2A_LINCSpretrained_Landmark.pt`.
The CSG2A subnetwork in PANDA was initialized from this Landmark checkpoint and
kept frozen during downstream GDSC/MOSA wrapper training. Exact feature
regeneration uses `260207_CSG2A_finetune_GDSC_MOSA_platform.pt`, supplied in
`PANDA_release_assets_v1.zip`, which stores the Landmark-derived core together
with the complete wrapper state and updated BatchNorm running statistics.

## PANDA asset bundles

[`PANDA_release_assets_v1.zip`](https://drive.google.com/file/d/1SJTRH_jq1zuD1pW3fm0XWe3FPYNPaEIh/view?usp=sharing)
is required for checkpoint-based reproduction.
It contains the 20 seed-0 fold checkpoints, TCGA CSG2A caches, the GDSC
gene-order file, and the full CSG2A inference wrapper checkpoint.

[`PANDA_optional_training_assets_v1.zip`](https://drive.google.com/file/d/1Qmb-ziv--KKTcx5rxmmdLyfbgEplJE0K/view?usp=sharing)
supplements the release bundle with
the GDSC post-treatment and compound caches, row metadata, and cache metadata
needed for optional seed-0 retraining.

## I-SPY2

The expression and clinical covariate tables in `data/External` originate from
the THERAPI data release. PANDA additionally supplies
`External_labeled_response.csv`, which contains the pCR endpoint used for the
179-patient evaluation.

The cohort is the
HER2-negative paclitaxel control cohort derived from GSE194040. The labeled
subset contains 31 pCR and 148 non-pCR patients. Additional unlabeled samples
from the same cohort are used for transductive target-domain adaptation.

## Redistribution

The THERAPI GitHub repository and data download do not currently display an
explicit redistribution license. Keep the upstream data out of the Git history
unless redistribution permission has been confirmed; the supplied installer
allows users to obtain it directly from the original source. CSG2A-generated
content is subject to the upstream CC BY-NC-SA 4.0 notice.
