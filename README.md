# Diabetic retinopathy referral from retina photos

An honestly evaluated model for detecting **referable diabetic retinopathy** (moderate NPDR or worse, the point at which a patient should see an eye doctor) from colour fundus photos.

- **Data prep.** Crops the retina even when text is burnt into the photo border. Finds near-identical duplicate photos in APTOS and keeps one per group, so the same photo never appears in both training and test. (APTOS has no patient IDs, so different photos of one person can still land in different splits.)
- **Model.** EfficientNet-B4 at 512 px, ImageNet-pretrained via `timm`. It has two outputs: a 0–4 severity score and a referral probability.
- **Training.** Ben Graham normalisation (refined so photo edges don't create artefacts), rotation/exposure/colour augmentation, square-root class balancing and mixed precision.
- **Evaluation.** Thresholds are tuned once on the validation split and then locked. Test and external results report AUC, sensitivity, specificity and QWK with 95% confidence intervals from 1,000 bootstrap resamples.

> Research code only. It is not a medical device and must not be used for clinical decisions.

The full write-up, with every result and the lessons learned, is in [REPORT.md](REPORT.md).

## Results

Four versions were trained on Kaggle's free T4 GPUs. Thresholds were locked on validation data before any test result was seen. IDRiD (516 photos from an eye clinic in Nanded, India, taken with a different camera) was never used for training or tuning.

| | Version 2: APTOS only | Version 3: + DDR, strong augmentation | Version 4: + DDR, rim hidden |
|---|---|---|---|
| APTOS test AUC (349 photos) | 0.983 (95% CI 0.968–0.993) | 0.985 | 0.983 |
| IDRiD AUC (516 photos) | **0.949** (0.929–0.965) | 0.897 (0.869–0.921) | 0.928 (0.908–0.948) |
| IDRiD sensitivity / specificity | 0.746 / **0.984** | 0.885 / 0.648 | 0.839 / 0.798 |

- **Version 2 is the recommended model.** On APTOS it caught 86% of referable eyes and cleared 97% of the rest. Version 1, trained with the same recipe, scored AUC 0.981, so results reproduce.
- **A new camera shifts the scores.** At IDRiD, Version 2 still ranked photos well, but its locked threshold caught only 75% of referable eyes. A threshold re-tuned on IDRiD's training folder reached 91% sensitivity (69% specificity) on its separate testing folder (`site_threshold.py`).
- **More data created a shortcut.** Adding DDR (photos from 147 hospitals and 42 camera models) made Version 3 read the edge of the photo, a camera signature rather than disease. The rim test (`rim_test.py`) exposed it: blanking a 10% band along the edge raised Version 3's IDRiD AUC from 0.897 to 0.938. Hiding the rim during training (`--fov-frac 0.9`, Version 4) recovered part of the loss.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Crops/resizes images, removes APTOS duplicates and writes `manifest.csv` with patient-level train/val/test splits. Also prepares external test sets. |
| `train.py` | Trains the model, keeps the best epoch, and tunes and locks thresholds on validation. |
| `evaluate.py` | Evaluates the internal test split or an external dataset, with 95% CIs. |
| `dr/` | Shared code: preprocessing, dataset and augmentation, model, metrics. |
| `site_threshold.py` | Sets the referral threshold for a new clinic or camera from a few of its labelled photos (section 7). |
| `rim_test.py` | Shortcut check: blanks the rim of the retina and measures how predictions change (section 6). |
| `calibrate_quality.py`, `explain_examples.py`, `app.py` | Photo-quality check, Grad-CAM example sheets, and the demo web app (section 5). |
| `smoke_test.sh`, `tools/make_dummy_data.py` | Full pipeline on synthetic images, to check your setup before downloading real data. |

## 0. Check your setup (a few minutes, no data needed)

```bash
pip install -r requirements.txt   # on Kaggle/Colab usually only: pip install -q timm
bash smoke_test.sh
```

The smoke test trains a tiny model on cartoon fundus images. Ignore its scores; it only proves every step runs. It was run successfully on Python 3.12 with PyTorch 2.14, timm 1.0, pandas 3.0 and NumPy 2.4, on CPU.

## 1. Get the data

| Dataset | Role | Where |
|---|---|---|
| **APTOS 2019** (3,662 graded photos, India) | Training. **Start here.** | Kaggle competition *APTOS 2019 Blindness Detection* (accept the rules first) |
| **EyePACS 2015** (35,126 graded photos, US) | Extra training data (optional, much larger) | Kaggle competition *Diabetic Retinopathy Detection* (about 80 GB), or any resized copy that includes `trainLabels.csv` (columns `image`, `level`) |
| **Messidor-2** (1,748 photos, France) | **External test.** Never train or tune on it. | Images from the official Messidor-2 page (licence form). Adjudicated grades published by Google as a Kaggle dataset. |
| **DDR** (13,673 photos from 147 hospitals in China) | Extra training data (Versions 3 and 4) | The authors' page on GitHub (`nkicsl/DDR-dataset`) or a Kaggle copy; the authors ask users to cite their paper |
| **IDRiD** (516 photos, India) | **External test used in this project** | [idrid.grand-challenge.org](https://idrid.grand-challenge.org/Data/) (CC BY 4.0) |

Check each dataset's terms before reusing it. APTOS and EyePACS come under Kaggle competition rules (EyePACS's rules limit use of the data to that competition), Messidor-2 needs a licence agreement, DDR's authors ask for a citation, and IDRiD is CC BY 4.0. A commercial product would need its own licensed data.

**Kaggle is the easiest place to run this.** It offers a free GPU, and APTOS attaches directly to a notebook.

1. Create a notebook and turn on the GPU.
2. Turn on **Internet**, which is needed to download the pretrained weights.
3. Add the APTOS competition data.
4. Upload this folder as a Kaggle Dataset, attach it, and copy it into `/kaggle/working`.
5. For long runs, use **Save Version → Save & Run All** so training continues in the background.

## 2. Prepare the data

```bash
# APTOS only (start here)
python prepare_data.py train-sets \
  --aptos-dir /kaggle/input/aptos2019-blindness-detection \
  --out-dir data/processed

# APTOS + EyePACS
python prepare_data.py train-sets \
  --aptos-dir /kaggle/input/aptos2019-blindness-detection \
  --eyepacs-labels /path/to/trainLabels.csv --eyepacs-images /path/to/train \
  --out-dir data/processed

# External test set: Messidor-2 (check the column names in your copy; the script lists them if one is wrong)
python prepare_data.py external --name messidor2 \
  --csv /path/to/messidor_data.csv --image-dir /path/to/messidor2/images \
  --image-col image_id --label-col adjudicated_dr_grade \
  --dme-col adjudicated_dme --gradable-col adjudicated_gradable \
  --out-dir data/processed

# APTOS + DDR (DDR's own train/valid/test folders are kept; photos graded ungradable are dropped)
python prepare_data.py train-sets \
  --aptos-dir /kaggle/input/aptos2019-blindness-detection \
  --ddr-dir /path/to/DDR-dataset \
  --out-dir data/processed

# External test set: IDRiD (prepare the training and testing folders separately, because they reuse file names)
python prepare_data.py external --name idrid_train \
  --csv idrid_train_labels.csv --image-dir "/path/to/1. Original Images/a. Training Set" \
  --image-col "Image name" --label-col "Retinopathy grade" --dme-col "Risk of macular edema" \
  --out-dir data/processed
python prepare_data.py external --name idrid_test \
  --csv idrid_test_labels.csv --image-dir "/path/to/1. Original Images/b. Testing Set" \
  --image-col "Image name" --label-col "Retinopathy grade" --dme-col "Risk of macular edema" \
  --out-dir data/processed
```

The script prints images per split and grade, plus any files it couldn't find or process (listed in `missing_*.csv` and `failed_*.csv`).

It also prints a **duplicate report** for APTOS. Real duplicates have an image correlation near 1.0, well apart from everything else. Open a few groups listed in `duplicates_aptos.csv` to confirm they really are the same photo. If different eyes were merged, raise `--dup-threshold` (default 0.95). Duplicate groups with conflicting grades are dropped, because their labels can't be trusted.

## 3. Train

```bash
# Version 2 (recommended): APTOS only
python train.py --manifest data/processed/manifest.csv --out runs/b4_512

# Version 4: APTOS + DDR, strong camera augmentation, outer 10% of the radius hidden
python train.py --manifest data/processed/manifest.csv --out runs/v4 --aug strong --epochs 8 --fov-frac 0.9
```

`--fov-frac` is saved in the checkpoint and applied automatically by `evaluate.py`, `rim_test.py`, `explain_examples.py` and `app.py`.

- **Rough timing on one T4:** about half an hour for APTOS only. With EyePACS added, several hours for 15 epochs; `--epochs 10` is fine there.
- **Out of GPU memory:** use `--batch-size 4 --accum-steps 4`, or a smaller model with `--backbone efficientnet_b3.ra2_in1k --batch-size 12`.
- **APTOS only after preparing both datasets:** add `--sources aptos`.
- **Outputs in `runs/b4_512/`:**
  - `best.pt`: the epoch with the best validation referral AUC.
  - `log.csv`
  - `thresholds.json`: the locked thresholds.
  - `val_predictions.csv`

## 4. Evaluate

```bash
python evaluate.py --run-dir runs/b4_512 --csv data/processed/manifest.csv --split test
python evaluate.py --run-dir runs/b4_512 --csv data/processed/external_messidor2.csv
```

Each run prints, and saves to `runs/b4_512/eval_*/`:
- AUC.
- Sensitivity and specificity at the locked threshold.
- PPV and NPV.
- Calibration error.
- QWK and a confusion matrix.
- A per-source breakdown.

## How to read the results

- **The external result is the one that matters.** Internal test scores are always optimistic, because the test images come from the same cameras and clinics as the training images.
- **Never change thresholds after looking at test or external results.** Once you do, the numbers are no longer an honest estimate. Tune only on `val`.
- **Rough ballparks** (from published Kaggle solutions and papers; not guarantees):
  - APTOS-only internal test: referral AUC about 0.95–0.98, QWK about 0.85–0.92.
  - Adding EyePACS usually *lowers* internal QWK (about 0.75–0.85) because its labels are noisier. That is expected.
  - Messidor-2 referral AUC: typically from the high 0.80s to the mid 0.90s for models trained on public data, rising with more training data.
- **Warning signs:**
  - Internal AUC of 0.995 or more suggests leakage; check the duplicate report.
  - Validation AUC peaking early and then falling means overfitting; use fewer epochs or more data.
  - A large gap between internal and external results is domain shift: the main open problem, and the most interesting thing to study.
- **DME.** When `--dme-col` is given, external "referable" also includes macular oedema. The model never saw DME labels, so it may miss eyes that are referable *only* because of DME. To see the difference, run evaluation with and without that flag.

## 5. Photo-quality check and demo app

```bash
# Set quality limits on the training photos, and test whether flagged test photos are harder for the model
python calibrate_quality.py --csv data/processed/manifest.csv --out quality_thresholds.json \
  --predictions runs/b4_512/eval_manifest_test/predictions.csv --examples quality_examples.jpg

# Grad-CAM sheet: caught cases, misses, false alarms and cleared cases from the test split
python explain_examples.py --run-dir runs/b4_512 --csv data/processed/manifest.csv \
  --predictions runs/b4_512/eval_manifest_test/predictions.csv --out gradcam_examples.jpg

# Demo web app (add --share for a temporary public link, e.g. when running on Kaggle)
pip install gradio
python app.py --run-dir runs/b4_512 --quality quality_thresholds.json
```

- **Photo-quality check.** This is rule-based. It measures six things: how much of the retina is visible, brightness, glare, sharpness, contrast, and whether a retina can be found at all.
  - **How photos get flagged.** A photo is flagged when it is more extreme than about 99% of the training photos on any measure. Sharpness and contrast are measured relative to brightness, so a dark photo is reported only as dark.
  - **What it isn't.** This catches clearly unusual photos; it is not a validated gradability model. Training a classifier on graded quality labels such as EyeQ is the proper upgrade.
  - **How to check it works.** `calibrate_quality.py` reports whether the model is less accurate on the photos it flags. If it isn't, the check isn't helping.
- **Grad-CAM.** The heatmap shows which regions raised the referral score. With this architecture (average pooling plus a linear head) it equals the classic class activation map. It is a rough guide to where the model looked, not a lesion map.
- **App.** It shows a referral suggestion, the referral score against the locked threshold, an estimated grade, the quality verdict and the heatmap. Flagged photos are reported as unreliable. Photos where no retina is found are reported as ungradable.

## 6. Shortcut check: does the model rely on the rim of the photo?

```bash
python rim_test.py --run-dir runs/b4_512 --csv data/processed/manifest.csv --split test --out rim_test
```

The script blanks parts of every test photo and re-scores it:
- **No rim 5%, 10% and 20%:** the outer 5%, 10% or 20% of the retina's radius is blanked.
- **Random patches:** random interior spots of the same total area as the 10% rim are blanked. This is the fair comparison.
- **Rim only 10%:** only the outer ring is kept.

For each version it reports AUC, sensitivity, specificity, the average change in referral score, and how many referral decisions flip. It also saves `examples.jpg`.

**How to read the results:**
- **Rim matters:** removing the rim hurts much more than removing the same area of interior. The random holes are less natural than a smaller field of view, so this is strong evidence the model leans on the rim.
- **Inconclusive:** if the random patches hurt more, the test can't tell you much about the rim.
- **Rim only:** a high AUC from the thin outer ring alone also points to the rim. Real lesions can sit in the periphery, so compare with what the heatmaps show.

## 7. Using the model with a new camera or clinic

The external test on IDRiD (516 photos from a different hospital and camera, never seen in training) showed:
- **Ranking holds up:** AUC was 0.949, against 0.983 on APTOS.
- **Scores run lower on the new camera:** at the APTOS threshold, sensitivity fell to 0.75.
- **Re-tuning on that site's own photos helps:** a threshold chosen on IDRiD's training folder reached 0.91 sensitivity on its separate testing folder, but specificity fell to 0.69.

Two tools address this.

**Site calibration.** Label a small set of the new site's photos (at least 30 referable), prepare them as an external set, then:

```bash
python evaluate.py --run-dir runs/b4_512 --csv data/processed/external_site_tune.csv --n-boot 0
python site_threshold.py --predictions runs/b4_512/eval_external_site_tune/predictions.csv \
  --thresholds runs/b4_512/thresholds.json --out runs/b4_512/thresholds_site.json
# measure it on DIFFERENT photos from the same site, then use it in the app
python evaluate.py --run-dir runs/b4_512 --csv data/processed/external_site_test.csv --thresholds runs/b4_512/thresholds_site.json
python app.py --run-dir runs/b4_512 --thresholds runs/b4_512/thresholds_site.json --quality quality_thresholds.json
```

**Stronger camera augmentation.** `train.py --aug strong` trains with wider colour, exposure and blur changes, plus lower resolution, JPEG compression and narrower fields of view. It's meant to make the model less dependent on one camera's look. In this project it didn't help on its own: adding DDR with strong augmentation (Version 3) lowered IDRiD AUC to 0.897, because the model learned a camera shortcut at the photo's edge. `--fov-frac 0.9` (Version 4) recovered part of the loss. Always re-run the rim test after changing the training data.

Image ids must be unique within one `--image-dir`. If two folders reuse file names (IDRiD's training and testing folders both contain `IDRiD_001.jpg`), `prepare_data.py` now skips those ids with a warning instead of guessing. Prepare each folder separately.

## Next steps

1. **Second external test.** Request access to Messidor-2 and test Versions 2 and 4 with their thresholds locked.
2. **Edge-following mask.** Hide a band along the retina's actual edge, wherever the photo ends, in training and use (IDRiD's photos appear to be cut flat at the top and bottom, inside the circle Version 4 keeps).
3. **Grading conventions.** Compare how DDR and IDRiD grade borderline cases, and try calibrating each source separately when datasets are mixed.
4. **Learned quality model.** Replace the rule-based check with a classifier trained on the EyeQ quality labels.
5. **Clinic pilot.** With a clinical partner, a prospective study against eye doctors' grades, with site calibration built into the setup.

## Data credits

- APTOS 2019 Blindness Detection, Kaggle (Asia Pacific Tele-Ophthalmology Society; photos from Aravind Eye Hospital).
- Li T., Gao Y., Wang K., Guo S., Liu H., Kang H. Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening. *Information Sciences* 501:511–522, 2019 (DDR).
- Porwal P. et al. Indian Diabetic Retinopathy Image Dataset (IDRiD): a database for diabetic retinopathy screening research. *Data* 3(3):25, 2018.

Dataset names credit their creators and don't imply any partnership.

## Acknowledgements

Built by Rohith Akulapalli with help from Claude, an AI assistant made by Anthropic, for coding, experiment runs and writing.

## Licence

The code is released under the MIT licence (see `LICENSE`). The datasets are not included and keep their own terms.
