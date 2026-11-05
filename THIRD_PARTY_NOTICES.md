# Third-party notices

## PANCDR

- Source: https://github.com/DMCB-GIST/PANCDR
- Publication: https://doi.org/10.1093/bib/bbae088
- Use in PANDA: pre-treatment expression encoder, graph-based drug encoder,
  adversarial discriminator, domain-alignment training pattern, and response
  prediction components were adapted and extended for dual alignment.
- PANDA modifications: integration of CSG2A-derived post-treatment and chemical
  representations, separate post-treatment adversarial alignment, updated
  checkpoint/inference code, and patient-level ensemble evaluation.

The PANCDR repository does not currently display an explicit software license.
Attribution and citation do not by themselves grant permission to redistribute
adapted source code. Permission from the PANCDR copyright holders should be
confirmed before public release.

## CSG2A

- Source: https://github.com/eugenebang/CSG2A
- Landmark weight: https://drive.google.com/file/d/17-fFIxXB1avNZ9gWsccZ7Ia5Myqfzi9M/view
- Use in PANDA: CSG2A perturbation network, CCE/MAT-based chemical encoder,
  molecular featurization utilities, 978 Landmark-gene ordering, and pretrained
  initialization.
- PANDA modifications: a minimal inference wrapper, patient-expression input
  handling, fixed reference dose/time conditions, and extraction of predicted
  post-treatment expression and chemical representations.

The minimal CSG2A inference implementation under
`external/CSG2A_asg_minimal` is derived from the upstream CSG2A codebase and is
distributed under GNU GPL v3. The upstream repository states that data or
content produced with CSG2A follows CC BY-NC-SA 4.0. Both license texts are
included in that directory.

The upstream Landmark checkpoint was loaded into the CSG2A subnetwork and its
learnable parameters were frozen during downstream GDSC/MOSA wrapper training.
The PANDA-compatible checkpoint is the saved full wrapper state: it contains
the Landmark-derived CSG2A core plus wrapper-specific layers and updated
BatchNorm running statistics needed for exact PANDA feature regeneration.

## THERAPI

- Source: https://github.com/Sunginyoung/THERAPI
- Data: https://drive.google.com/drive/folders/1rnYXYwwqwfqS-q50D68EnYjkUUfw2an2?usp=drive_link
- Use in PANDA: processed GDSC expression and response data, TCGA labeled and
  unlabeled expression/annotation data, and I-SPY2 expression/clinical tables.

PANDA retains the source datasets' biological meaning and adds model-specific
normalization, row-aligned CSG2A caches, response manifests, and evaluation
artifacts. The THERAPI repository does not currently display an explicit
repository license. Confirm data and code redistribution rights before public
release; users should preferably download the original data from THERAPI.
