# Minimal CSG2A dependency

This directory contains a minimal inference subset derived from
[CSG2A](https://github.com/eugenebang/CSG2A) for PANDA feature generation.
Source modifications expose predicted post-treatment expression and compound
representations to PANDA.

Download the upstream Landmark checkpoint from the
[official CSG2A link](https://drive.google.com/file/d/17-fFIxXB1avNZ9gWsccZ7Ia5Myqfzi9M/view)
and place it at:

```text
ckpts/CSG2A_LINCSpretrained_Landmark.pt
```

The full `260207_CSG2A_finetune_GDSC_MOSA_platform.pt` checkpoint contains the
Landmark-derived frozen CSG2A core plus the complete downstream wrapper state
used to reproduce PANDA's cached features.

See `LICENSE` for GPL-3.0 and `CC-BY-NC-SA-4.0` for the upstream notice on
CSG2A-generated content.
