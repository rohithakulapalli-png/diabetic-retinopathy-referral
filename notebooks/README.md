# Kaggle notebooks

These are the notebooks behind the results in the main README. They ran on Kaggle's free T4 GPUs, with this repository's code attached as private Kaggle datasets (the `dr_*_code.bin` bundles the first cell unpacks).

| Notebook | What it ran |
|---|---|
| `external_test_idrid.ipynb` | Version 2 on all 516 IDRiD photos, fair site calibration, rim test and Grad-CAM sheet |
| `version3_aptos_ddr.ipynb` | Version 3: APTOS + DDR with strong augmentation, internal test and the IDRiD external test |
| `version4_rim_hidden.ipynb` | Version 4: as Version 3 with the outer 10% of the radius hidden, plus a DDR file-name diagnostic |

To reuse one, attach the datasets it names, copy this repository into `/kaggle/working`, and replace the first cell's unpacking step with `cd` into that folder.
