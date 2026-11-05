# PANDA

Official implementation of the **Patient-level Adversarial Network for Dual
Alignment of pre-treatment and drug-induced post-treatment transcriptomic
states (PANDA)** for patient-level drug-response prediction.

PANDA combines pre-treatment transcriptomic alignment, CSG2A-inferred
post-treatment representations, and UGCN/MAT-derived chemical information to
transfer drug-response knowledge from GDSC cell lines to patient cohorts.

## Upstream foundations

PANDA was developed by adapting and extending components and resources from
the following projects:

- [PANCDR](https://github.com/DMCB-GIST/PANCDR): pre-treatment expression
  encoder, molecular graph encoder, adversarial domain-alignment strategy, and
  response-prediction components.
- [CSG2A](https://github.com/eugenebang/CSG2A): condition-specific
  perturbation model, MAT-based chemical representation, molecular
  featurization utilities, and LINCS L1000 Landmark-gene pretrained weights.
- [THERAPI](https://github.com/Sunginyoung/THERAPI): processed GDSC, TCGA, and
  I-SPY2 data resources used by PANDA.

PANDA adds dual pre-/post-treatment alignment, integration of the two chemical
encoders, patient-level ensemble inference, and the evaluation procedures in
this repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
provenance, modifications, and licensing details. Use of these projects does
not imply endorsement by their original authors.

## Reproducible scope

The repository provides the seed-0 analysis used for the main PANDA model:

- PANDA model architecture and branch-level data flow
- ten fold-specific checkpoints and ensemble prediction
- TCGA AUROC, AUPRC, F1, and patient-cluster bootstrap confidence intervals
- transductive I-SPY2 paclitaxel evaluation
- seed-0 ten-fold training
- prediction for new patient-drug pairs
- pre- and post-treatment latent representation extraction

This release is centered on checkpoint-based reproduction of the primary
TCGA and I-SPY2 analyses and also provides the seed-0 training and
patient-drug inference workflows.

## Repository layout

```text
PANDA/
|-- checkpoints/          # Seed-0 TCGA and I-SPY2 fold checkpoints
|-- data/                 # GDSC, TCGA, I-SPY2, and cached CSG2A inputs
|-- examples/             # Example patient expression and drug SMILES
|-- external/             # Minimal frozen CSG2A inference dependency
|-- reference/            # Published seed-0 predictions and fold metrics
|-- scripts/              # Upstream data preparation utility
|-- src/                  # Training, inference, and evaluation code
|-- tests/                # Reference-metric tests
|-- environment.yml
`-- requirements.txt
```

## Installation

```bash
git lfs install
git clone https://github.com/sugyun/PANDA.git
cd PANDA
conda env create -f environment.yml
conda activate panda
```

For a CPU-only environment, install a compatible PyTorch build first and then
run `pip install -r requirements.txt`.

### Install checkpoint-reproduction assets

Download
[`PANDA_release_assets_v1.zip`](https://drive.google.com/file/d/1SJTRH_jq1zuD1pW3fm0XWe3FPYNPaEIh/view?usp=sharing)
and extract it into the repository root. The archive provides:

```text
checkpoints/tcga_seed0/                         # 10 fold checkpoints
checkpoints/ispy2_seed0/                        # 10 fold checkpoints
data/THERAPI_CSG2A_zscore/                      # TCGA inference caches and gene order
external/CSG2A_asg_minimal/ckpts/260207_CSG2A_finetune_GDSC_MOSA_platform.pt
SHA256SUMS
```

```bash
python -m zipfile -e PANDA_release_assets_v1.zip .
```

This is the primary asset bundle for reproducing checkpoint-based TCGA and
I-SPY2 predictions, ensemble metrics, and bootstrap confidence intervals.

## Download upstream assets

### THERAPI data

Download the model dataset from the
[THERAPI Google Drive folder](https://drive.google.com/drive/folders/1rnYXYwwqwfqS-q50D68EnYjkUUfw2an2?usp=drive_link)
and extract it under `downloads/THERAPI/data`:

```text
downloads/THERAPI/data/
|-- GDSC/
|   |-- GDSC_split/
|   |-- GDSC_gex.csv
|   |-- GDSC_info.csv
|   |-- GDSC_Drug_SMILES_Response.csv
|   |-- GDSC_perturbation_float16.npy
|   |-- GDSC_perturbation_compound_float16.npy
|   `-- GDSC_rankrepresentation.csv
|-- TCGA/
|   |-- TCGA_unlabeled_gex.csv
|   |-- TCGA_unlabeled_info.csv
|   |-- TCGA_labeled_gex.csv
|   |-- TCGA_labeled_info.csv
|   |-- TCGA_Drug_SMILES_Response.csv
|   |-- TCGA_perturbation_float16.npy
|   |-- TCGA_perturbation_compound_float16.npy
|   |-- TCGA_rankrepresentation.csv
|   `-- TCGA_Survival_info.csv
`-- External/
    |-- External_unlabeled_gex.csv
    |-- External_unlabeled_info.csv
    |-- External_labeled_gex.csv
    `-- External_labeled_info.csv
```

Install the files used by PANDA into its expected paths:

```bash
python scripts/prepare_therapi_data.py --source downloads/THERAPI/data
```

The installer copies the expression, annotation, and response tables to
`data/THERAPI` and `data/External`. The PANDA-specific CSG2A z-score feature
caches under `data/THERAPI_CSG2A_zscore` are separate project artifacts; they
are not interchangeable with the perturbation arrays in the THERAPI download.

### Optional: CSG2A Landmark pretrained weights

Download the approximately 170MB
[CSG2A Landmark pretrained model](https://drive.google.com/file/d/17-fFIxXB1avNZ9gWsccZ7Ia5Myqfzi9M/view)
linked by the [CSG2A repository](https://github.com/eugenebang/CSG2A), retain
the filename `CSG2A_LINCSpretrained_Landmark.pt`, and place it at:

```text
external/CSG2A_asg_minimal/ckpts/CSG2A_LINCSpretrained_Landmark.pt
```

This download is not required for checkpoint-based PANDA evaluation. It is the
Landmark checkpoint used to initialize the CSG2A subnetwork. The
CSG2A core was frozen during the downstream GDSC/MOSA wrapper training. Exact
PANDA feature generation uses the saved full wrapper checkpoint:

```text
external/CSG2A_asg_minimal/ckpts/260207_CSG2A_finetune_GDSC_MOSA_platform.pt
```

The full checkpoint contains the same Landmark-derived CSG2A core together
with the wrapper's scaling and prediction/projection layers and updated
BatchNorm running statistics. Therefore, download the Landmark checkpoint to
rerun the upstream initialization step; retain the full wrapper checkpoint to
reproduce the cached PANDA features exactly.

## Quick evaluation from reference predictions

TCGA:

```bash
python src/evaluate_predictions.py \
  --predictions reference/tcga/predictions.csv \
  --output-dir outputs/tcga_reference
```

I-SPY2:

```bash
python src/evaluate_predictions.py \
  --predictions reference/ispy2/predictions.csv \
  --output-dir outputs/ispy2_reference
```

The expected TCGA ensemble estimates are AUROC 0.783, AUPRC 0.760, and F1
0.726. The expected I-SPY2 estimates are AUROC 0.674 and AUPRC 0.343.

## Evaluation directly from checkpoints

```bash
python src/evaluate_tcga_checkpoints.py --device cuda
python src/evaluate_predictions.py \
  --predictions outputs/tcga_checkpoint_predictions.csv \
  --output-dir outputs/tcga_checkpoint_evaluation
```

```bash
python src/evaluate_ispy2_checkpoints.py --device cuda
python src/evaluate_predictions.py \
  --predictions outputs/ispy2_checkpoint_predictions.csv \
  --output-dir outputs/ispy2_checkpoint_evaluation
```

The I-SPY2 experiment is a transductive external evaluation: unlabeled samples
from the cohort are used for target-domain adaptation, while pCR labels are
withheld from training.

## Train the seed-0 models

To retrain PANDA, also download
[`PANDA_optional_training_assets_v1.zip`](https://drive.google.com/file/d/1Qmb-ziv--KKTcx5rxmmdLyfbgEplJE0K/view?usp=sharing)
and extract it into the repository root:

```bash
python -m zipfile -e PANDA_optional_training_assets_v1.zip .
```

This supplementary archive provides the large GDSC CSG2A perturbation and
compound caches. The THERAPI base data described above are also required.

TCGA analysis:

```bash
python src/train_tcga.py
```

I-SPY2 transductive analysis:

```bash
python src/train_ispy2.py
```

Both commands use ten GDSC folds, `cv_seed=0`, and training seed 0. CUDA and
substantial training time are required. Small numerical variation may remain
across CUDA, driver, and hardware versions.

External outcome labels are evaluated only after each checkpoint has been
selected using GDSC validation performance.

The primary reproducibility pathway uses the supplied seed-0 checkpoints.
Training scripts are provided to permit retraining, but bitwise-identical
checkpoints are not guaranteed across software, CUDA, and hardware versions.

## Predict a new patient-drug pair

Patient expression must contain the required 978 genes on the same
log2(TPM+1)-like scale used by the provided inputs. Missing genes raise an
error unless `--allow-missing-zero` is explicitly supplied.

```bash
python src/predict.py \
  --expression-csv examples/patient_expression.csv \
  --sample-id TCGA-A2-A3XS \
  --id-col Sample \
  --smiles-csv examples/drugs.csv \
  --smiles-col smiles \
  --drug-name-col drug_name \
  --save-latents \
  --out-csv outputs/example_predictions.csv
```

`--save-latents` writes the pre-treatment, post-treatment, and chemical latent
representations to a JSON file next to the prediction CSV.

## Data and model details

See [DATA.md](DATA.md) for file provenance and preprocessing assumptions.
Reference outputs are provided for verification; TCGA labels are not used for
training, early stopping, or checkpoint selection. I-SPY2 pCR labels are also
withheld from training.

See [UPLOAD_MANIFEST.md](UPLOAD_MANIFEST.md) for the files included in the
public release and the upstream assets that users download separately.

## License

CSG2A-derived source code is provided under GPL-3.0. CSG2A-generated content
and weights are subject to the upstream CC BY-NC-SA 4.0 notice. The PANCDR and
THERAPI GitHub repositories do not currently display an explicit repository
license; attribution does not itself grant redistribution rights. Obtain any
required permission before publishing adapted code or redistributed data. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
