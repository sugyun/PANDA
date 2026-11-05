# Upload manifest

## Include in the GitHub repository

- `src/`, `scripts/`, `tests/`, and `examples/`
- `data/External/External_labeled_response.csv`
- minimal CSG2A source, gene order, and STRING data under
  `external/CSG2A_asg_minimal/`
- `reference/`
- documentation, environment files, licenses, and checksum manifests

## Published release assets

### `PANDA_release_assets_v1.zip`

- Download: https://drive.google.com/file/d/1SJTRH_jq1zuD1pW3fm0XWe3FPYNPaEIh/view?usp=sharing

- `checkpoints/tcga_seed0/`
- `checkpoints/ispy2_seed0/`
- all `THERAPI_TCGA_LABELED_*` files under `data/THERAPI_CSG2A_zscore/`
- `data/THERAPI_CSG2A_zscore/THERAPI_GDSC_samplewise_zscore_978genes.csv`
- `external/CSG2A_asg_minimal/ckpts/260207_CSG2A_finetune_GDSC_MOSA_platform.pt`
- an archive-specific `SHA256SUMS`

This is the required checkpoint-reproduction bundle.

### `PANDA_optional_training_assets_v1.zip`

- Download: https://drive.google.com/file/d/1Qmb-ziv--KKTcx5rxmmdLyfbgEplJE0K/view?usp=sharing

- `data/THERAPI_CSG2A_zscore/THERAPI_GDSC_csg2a_zscore_perturbed_gex_float16.npy`
- `data/THERAPI_CSG2A_zscore/THERAPI_GDSC_csg2a_zscore_compound_float16.npy`
- `data/THERAPI_CSG2A_zscore/THERAPI_GDSC_csg2a_zscore_row_metadata.csv`
- `data/THERAPI_CSG2A_zscore/THERAPI_GDSC_csg2a_zscore_meta.json`
- an archive-specific `SHA256SUMS`

This archive is optional and supplements the release bundle for retraining.

## Do not upload

- `downloads/` or `outputs/`
- THERAPI base data under `data/THERAPI/`
- the four THERAPI-derived expression/annotation files under `data/External/`
- `external/CSG2A_asg_minimal/ckpts/CSG2A_LINCSpretrained_Landmark.pt`

Users obtain THERAPI base data from its original release. The Landmark
checkpoint is needed only to reconstruct or further fine-tune the CSG2A
feature generator, not for checkpoint-based PANDA evaluation.

## Release prerequisite

Confirm permission to redistribute PANCDR-adapted source because the upstream
PANCDR repository does not currently provide an explicit software license.
Also confirm the redistribution terms for THERAPI-derived data. Attribution
alone does not grant redistribution permission.
