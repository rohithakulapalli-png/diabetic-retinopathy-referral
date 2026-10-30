# Diabetic Retinopathy Detection from Retina Photos: Project Report

Rohith Akulapalli · 21 October 2026

## Summary

We set out to build a model that looks at a photo of the back of the eye and tells a person with diabetes whether they should see an eye doctor. Our best model, Version 2, was trained on the APTOS 2019 collection from India. On 349 APTOS photos it had never seen, it reached an AUC of 0.983 (95% CI 0.968–0.993), catching 86% of the eyes that needed referral and correctly clearing 97% of those that didn't.

The harder test was a different hospital with a different camera. On 516 photos from IDRiD, Version 2 still told sick eyes from healthy ones well (AUC 0.949), but its scores came out lower on the new camera, so it caught only 75% of referable eyes at its original threshold. Re-tuning the threshold with that clinic's own photos brought sensitivity back up to 91%, at the cost of more false alarms (specificity 69%).

We then tried to make the model sturdier by adding 12,522 photos from the DDR dataset, taken with 42 kinds of camera across China. It backfired. Version 3 learned to read the edge of the photo, a sign of which camera took it rather than of disease, and our rim test caught it (IDRiD AUC 0.897). Hiding the rim in Version 4 won back part of the loss (AUC 0.928), but Version 2 is still the model we recommend.

This is a research prototype, not a medical device. It hasn't been tested on patients in a clinic or reviewed by any regulator.

## Background and goal

Diabetic retinopathy is damage to the tiny blood vessels of the retina caused by diabetes. It is one of the leading causes of preventable blindness in working-age adults, and it gives no warning in its early stages, so people often don't know they have it until their sight is already affected. Caught in time, most of that vision loss can be prevented.

That is why people with diabetes are advised to have their eyes checked regularly. But reading retina photos takes trained specialists, and there aren't nearly enough of them. In parts of India there is only one eye-care expert for every 608,000 people (Porwal et al., 2018). A model that flags the photos needing a specialist could let nurses or technicians screen far more people, and let doctors spend their time on the patients who need them.

Our goal was a model that answers one practical question: does this eye have referable diabetic retinopathy? We counted an eye as referable at grade 2 (moderate) or worse on the international five-level scale, which runs from 0 (none) through mild, moderate and severe to 4 (proliferative). Predicting the exact grade was a secondary aim.

## Data

We worked with three public collections. APTOS and DDR were used for training, while IDRiD was held back as an outside test: the model never saw it during training or tuning.

| Dataset | Where it comes from | Photos used | How we used it | Licence |
| --- | --- | --- | --- | --- |
| APTOS 2019 Blindness Detection | Aravind Eye Hospital, India (Kaggle competition) | 3,486 of 3,662 | Training, validation and internal testing in every version | Kaggle competition rules; check permitted uses before publishing |
| DDR | 147 hospitals in 23 provinces of China; 42 camera models | 12,522 of 13,673 | Extra training data in Versions 3 and 4 | No licence stated; the authors ask users to cite their paper (Kaggle copy labelled MIT) |
| IDRiD, disease-grading set | Eye clinic in Nanded, India; Kowa VX-10α camera, 50° field of view | 516 | Outside test only | CC BY 4.0 (original release) |

APTOS needed some cleaning first. Comparing every pair of photos turned up 131 groups of near-identical duplicates covering 270 photos, and in 37 of those groups the copies had been given different grades. We kept one photo from each consistent group and dropped the conflicting groups altogether, which removed 176 photos.

The remaining 3,486 photos were split into 2,788 for training, 349 for validation and 349 for testing, with referable disease making up 39%, 35% and 38% of each. APTOS doesn't record which photos belong to which patient, so two photos of the same person could still have ended up in different splits.

DDR comes with its own training, validation and test folders, which we kept as they were. We left out the 1,151 photos its graders marked as ungradable, leaving 6,260, 2,503 and 3,759 photos in the three folders. The photos come from 9,598 patients, but neither patients nor hospitals are identified.

IDRiD has 413 photos in a training folder and 103 in a testing folder. Each is graded 0–4 for retinopathy, with a separate risk score for macular oedema. Both folders reuse the same file names, so we prepared them separately. For the outside test, we counted an eye as referable if it had grade 2 or worse or any oedema risk (62.6% of photos); in IDRiD this turned out to be exactly the same set of eyes as grade 2 or worse alone.

## Methods

### How a photo is processed

Every photo goes through the same steps, whether during training or in use. First we crop it to the retina, finding the largest bright region while ignoring any text printed on the border, and resize it to 512 × 512 pixels. Then we even out the lighting by subtracting a blurred copy of the image (Ben Graham's method), which makes small lesions stand out. The blur averages only retina pixels, so it doesn't leave a bright ring at the edge.

The model is an EfficientNet-B4 network, starting from Noisy Student ImageNet weights. For each photo it gives two outputs: a grade score between 0 and 4, and the probability that the eye needs referral. We score the photo and its three mirror images and average the results. The eye is referred if that probability reaches a fixed threshold, and the grade score is turned into a grade using tuned cut-offs.

### Training

The training loss combined a smooth-L1 loss on the grade score with a cross-entropy loss on referral, weighted equally. We used the AdamW optimiser with a learning rate of 0.0002 and weight decay of 0.01, a one-epoch warm-up followed by cosine decay, an effective batch of 16 photos and mixed precision. Rarer grades were sampled more often, and photos were randomly flipped, rotated, zoomed, shifted and varied in brightness, contrast, colour, gamma and focus. After each epoch we checked the validation AUC and kept the best epoch.

We trained four versions:

| Version | Training data | What changed | Epochs |
| --- | --- | --- | --- |
| 1 | APTOS | The baseline | 15 |
| 2 | APTOS | The same recipe, retrained to check reproducibility, with updated quality-check and heatmap tools | 15 |
| 3 | APTOS + DDR | Stronger camera augmentation: wider swings in colour and exposure, lower resolution, JPEG compression and a narrower field of view | 8 |
| 4 | APTOS + DDR | As Version 3, but with the outer 10% of each photo's radius hidden during training and use | 8 |

### Keeping the test honest

Once training finished, we set the referral threshold on the validation set so that it caught 90% of referable eyes, and tuned the grade cut-offs on the same set. Both were saved before we looked at any test result and never changed afterwards.

On each test set we report the AUC for referral, the sensitivity and specificity at that locked threshold, and the quadratic weighted kappa (QWK), which measures agreement with the true grade. The 95% confidence intervals come from 1,000 bootstrap resamples of the test photos.

### Checks and tools

To see what the model was looking at, we used Grad-CAM heatmaps, scaled to the strongest 0.5% of values inside the retina so that faint maps stay visible. A rule-based quality check flags photos with too little retina, poor exposure, glare, blur or low contrast. Its limits come from the spread of the training photos, and it warns rather than blocks.

We also built a rim test, because the edge of a photo carries signs of the camera and hospital that a model can learn in place of disease. The test re-scores photos with parts blanked out: a band along the retina's edge 5, 10 or 20% of its radius wide, random patches inside the retina covering the same area as the 10% band, or everything except that band. If losing the rim hurts more than losing the same amount of interior, the model is leaning on the rim.

Finally, for a new camera we can re-tune the referral threshold using a labelled set of that site's photos, again aiming for 90% sensitivity, and then measure it on different photos from the same site. For IDRiD we tuned on its training folder (413 photos) and measured on its testing folder (103 photos).

## How the model did on APTOS

Both APTOS-only versions reached an AUC of about 0.98 on the 349 held-out APTOS photos, and retraining from scratch reproduced the first run within its confidence intervals.

| APTOS test set, 349 photos | Version 1 | Version 2 |
| --- | --- | --- |
| Referral AUC (95% CI) | 0.981 (0.966–0.992) | 0.983 (0.968–0.993) |
| Sensitivity at the locked threshold | 0.886 | 0.864 |
| Specificity at the locked threshold | 0.963 | 0.968 |
| Grading agreement, QWK (95% CI) | 0.916 (0.887–0.938) | 0.905 (0.877–0.930) |
| Best epoch | 10 of 15 | 5 of 15 |

The rim test gave Version 2 a clean bill of health. Blanking the rim changed its results no more than blanking the same area inside the retina. The rim on its own still scored 0.942, which makes sense, because real lesions do reach the edge of the retina.

| Version 2, APTOS test | Retina removed | AUC | Decisions changed |
| --- | --- | --- | --- |
| Nothing blanked | 0% | 0.983 | 0% |
| Edge band, 5% of radius | 11% | 0.982 | 1.1% |
| Edge band, 10% of radius | 21% | 0.981 | 3.4% |
| Edge band, 20% of radius | 38% | 0.982 | 4.6% |
| Random interior patches, same area as the 10% band | 21% | 0.981 | 3.2% |
| Everything except the 10% band | 80% | 0.942 | 12.9% |

The quality check flagged 4.3% of the training photos. On the test set, AUC was 0.982 for the photos that passed; only 16 test photos were flagged, too few to say anything about them.

## Testing at another hospital

IDRiD was the real test: a different hospital, a different camera, and photos the model had never seen. Version 2 still told sick eyes from healthy ones well (AUC 0.949). But its scores came out lower on the new camera, so the threshold we had set on APTOS missed a quarter of the eyes that needed referral.

| Version 2 | APTOS test, 349 photos | IDRiD, 516 photos |
| --- | --- | --- |
| Referral AUC (95% CI) | 0.983 (0.968–0.993) | 0.949 (0.929–0.965) |
| Sensitivity at the locked threshold | 0.864 | 0.746 (0.699–0.793) |
| Specificity at the locked threshold | 0.968 | 0.984 (0.965–1.000) |
| Grading agreement, QWK | 0.905 | 0.776 (0.739–0.811) |

Most of the misses were eyes graded too mildly. Of the 168 IDRiD eyes with moderate disease, the model called 45 (27%) mild or healthy.

Re-tuning the threshold for the new site largely fixed this. A threshold chosen on IDRiD's training folder came out at 0.070, far below the APTOS value of 0.555, which shows just how much lower the scores ran. On the separate testing folder it lifted sensitivity from 73% to 91%, while specificity fell from 95% to 69%. With only 103 test photos, these numbers are rough.

| IDRiD testing folder, 103 photos (AUC 0.903) | APTOS threshold, 0.555 | Site threshold, 0.070 |
| --- | --- | --- |
| Sensitivity (95% CI) | 0.734 (0.618–0.843) | 0.906 (0.828–0.969) |
| Specificity (95% CI) | 0.949 (0.865–1.000) | 0.692 (0.553–0.839) |

The rim test held up on the new camera too. Blanking a 10% band along the edge left AUC at 0.952, while blanking the same area of interior lowered it to 0.938; the rim alone scored 0.867.

## Adding photos from more cameras

We hoped that training on photos from many more cameras would help the model travel better. It didn't. Version 3, trained on APTOS plus DDR, picked up a camera shortcut, and Version 4 fixed only part of it. Neither beat Version 2 at the outside hospital.

|  | Version 2: APTOS only | Version 3: + DDR | Version 4: + DDR, rim hidden |
| --- | --- | --- | --- |
| APTOS test AUC, 349 photos | 0.983 | 0.985 | 0.983 |
| IDRiD AUC, 516 photos (95% CI) | 0.949 (0.929–0.965) | 0.897 (0.869–0.921) | 0.928 (0.908–0.948) |
| IDRiD sensitivity | 0.746 | 0.885 | 0.839 |
| IDRiD specificity | 0.984 | 0.648 | 0.798 |
| IDRiD grading agreement, QWK | 0.776 | 0.658 | 0.696 |

On DDR's own test folder of 3,759 photos, Versions 3 and 4 reached AUCs of 0.962 and 0.953, so they had learned DDR well. At IDRiD, though, they over-called disease: Version 3 graded 80 of the 168 healthy eyes as moderate, and Version 4 graded 64.

The rim test showed why. Blanking the edge band lifted Version 3's IDRiD AUC from 0.897 to 0.938 and its specificity from 65% to 94%, while blanking the same area inside the retina moved its scores only half as much. Version 3 had learned to read the edge of the photo, most likely as a clue to which camera or hospital it came from. If DDR's hospitals see different amounts of disease, that clue would help within DDR but mislead anywhere else; we can't confirm this, because DDR doesn't identify its hospitals.

| IDRiD AUC, with part of each photo blanked | Version 2 | Version 3 | Version 4 |
| --- | --- | --- | --- |
| Nothing blanked | 0.949 | 0.897 | 0.928 |
| 10% band along the retina's edge | 0.952 | 0.938 | 0.946 |
| Random interior patches, same area | 0.938 | 0.919 | 0.927 |

Hiding the rim isn't a new idea: the winner of the 2015 Kaggle diabetic retinopathy competition also cropped photos to 90% of the radius ([TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection)). Version 4 never sees the outer 10% of the radius, and that brought its IDRiD AUC up to 0.928. It wasn't a complete fix, though. IDRiD's photos appear to be cut flat at the top and bottom, which would put part of the photo's edge inside the circle Version 4 keeps, and blanking a band along the retina's real edge still raised Version 4's AUC to 0.946 and its specificity to 98%.

One puzzle remains. For both versions, the locked threshold caught 90% of referable eyes on validation, which was mostly DDR, but only 75% and 74% on DDR's own test folder. DDR's folders may not be split by patient, or may differ in other ways; because its files don't identify patients or hospitals, we couldn't check.

## Demo app

We also built a small web app around Version 2. You upload a retina photo, and it returns a referral suggestion, a referral score, a Grad-CAM heatmap and a verdict on photo quality. It runs on Kaggle's free GPU behind a temporary public link, and it can use a site-specific threshold from the calibration tool.

In a live test with photos from the APTOS test set, it gave the right answer for each grade we tried:

| True grade | App's suggestion | Referral score |
| --- | --- | --- |
| 0, no retinopathy | No referable DR found | 0.00 |
| 2, moderate | Refer to an eye doctor | 0.81 |
| 4, proliferative | Refer to an eye doctor | 0.99 |

## What we learned

The biggest lesson is that a model can look excellent on the data it came from and still stumble at a new hospital, and that adding more data can make things worse unless you check what the model is actually learning.

Ranking travelled better than thresholds. Version 2's AUC dipped only from 0.983 to 0.949 at IDRiD, but at the same threshold its sensitivity fell from 86% to 75%. The new camera shifted the scores more than it scrambled their order.

That makes the threshold a local decision. A few hundred labelled photos from a new site were enough to re-tune it and bring sensitivity back to 91%, but the price was more false alarms, so each site has to choose that trade-off for itself.

More data is not automatically better. Mixing collections with different cameras handed the model a shortcut, and a cheap check like the rim test was enough to catch it and point towards a fix. We also suspect the datasets draw the line between mild and moderate disease differently, since models trained with DDR called many healthy IDRiD eyes moderate, though we haven't tested that directly.

For now, Version 2 is the model we recommend. It ranks IDRiD photos best and raises the fewest false alarms, and with a site-tuned threshold it reaches 91% sensitivity at the outside hospital.

## Limitations

These results come from public photo collections rather than from patients in a clinic, so they should be read as early research evidence.

- We tested at only one outside hospital, with one camera model and 516 photos, and the site-calibration result rests on just 103 of them.
- APTOS and DDR don't identify patients, so photos of the same person may appear in both the training and test sets, which can flatter the internal results.
- Each photo has a single public grade, produced by a process we can't check. APTOS has no macular-oedema labels, so referral there is based on the retinopathy grade alone.
- Everything here is retrospective. There has been no clinic study and no comparison with eye doctors grading the same patients.
- Photo quality is judged by fixed rules rather than a trained model, and the referral model itself can't say that a photo is ungradable.
- Versions 3 and 4 were each trained only once, with little tuning. Versions 1 and 2 suggest that run-to-run variation is small, but only on APTOS.
- This is not a medical device. Using it on patients would need a prospective study and regulatory approval.

## Next steps

The most useful next step is a second outside test, because a single hospital can't tell us how well any version generalises.

1. Request access to Messidor-2, a widely used reference set from France, and test Versions 2 and 4 on it with their thresholds locked.
2. Try a Version 5 that hides a band along the retina's actual edge, wherever the photo ends, in both training and use. This would show whether the remaining edge effect explains Version 4's gap to Version 2.
3. Compare how DDR and IDRiD grade borderline cases, and try calibrating each source separately when datasets are mixed.
4. Replace the rule-based quality check with a model trained on photo-quality labels, such as those in EyeQ.
5. Find a clinical partner for a prospective study against eye doctors' grades, with site calibration built into the setup.

## Reproducibility and references

All the code is in this repository, together with a smoke test that runs the whole pipeline on generated photos in a few minutes. Every result in this report came from private notebooks running on Kaggle's free T4 GPUs:

| Kaggle notebook (private) | What it ran | Run time |
| --- | --- | --- |
| DR baseline - APTOS | Versions 1 and 2: training, APTOS test, rim test, quality check | About 35 minutes each |
| DR external test - IDRiD | Version 2 on IDRiD, site calibration, rim test | 8.5 minutes |
| DR baseline v3 - APTOS + DDR | Version 3 (notebook version 1) and Version 4 (notebook version 2) | 3.3 hours and 1.9 hours |
| DR demo app v2 | The demo app with a temporary public link | On demand |

As an example, these are the commands that prepared the data and trained and tested Version 4:

```bash
python prepare_data.py train-sets --aptos-dir <APTOS folder> --ddr-dir <DDR folder> --out-dir data/processed
python train.py --manifest data/processed/manifest.csv --out runs/v4 --aug strong --epochs 8 --fov-frac 0.9
python evaluate.py --run-dir runs/v4 --csv data/processed/manifest.csv --split test
```

**Data sources.** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) · [DDR, official page](https://www.github.com/nkicsl/DDR-dataset) and [Kaggle copy](https://www.kaggle.com/datasets/samriddhibagchi/ddr-dataset-credits-to-authors) · [IDRiD, official page](https://idrid.grand-challenge.org/Data/) and [Kaggle copy](https://www.kaggle.com/datasets/abdullahshafi315/indian-diabetic-retinopathy-image-datasetidrid)

**References.** Each was checked against publisher or index pages on 21 October 2026.

1. Porwal P., Pachade S., Kamble R., Kokare M., Deshmukh G., Sahasrabuddhe V., Meriaudeau F. Indian Diabetic Retinopathy Image Dataset (IDRiD): a database for diabetic retinopathy screening research. *Data* 3(3):25, 2018. [doi:10.3390/data3030025](https://www.doi.org/10.3390/DATA3030025)
2. Li T., Gao Y., Wang K., Guo S., Liu H., Kang H. Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening. *Information Sciences* 501:511–522, 2019 (the DDR dataset). [doi:10.1016/j.ins.2019.06.011](https://doi.org/10.1016/j.ins.2019.06.011)
3. Tan M., Le Q. EfficientNet: rethinking model scaling for convolutional neural networks. *Proceedings of the 36th ICML*, PMLR 97:6105–6114, 2019. [PMLR](https://proceedings.mlr.press/v97/tan19a.html)
4. Xie Q., Luong M.-T., Hovy E., Le Q. V. Self-training with Noisy Student improves ImageNet classification. *CVPR*, 2020. [doi:10.1109/CVPR42600.2020.01070](https://mlanthology.org/cvpr/2020/xie2020cvpr-selftraining/)
5. Graham B. Kaggle Diabetic Retinopathy Detection competition report. University of Warwick, 2015.
6. Selvaraju R. R., Cogswell M., Das A., Vedantam R., Parikh D., Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. *ICCV*, 2017, pp. 618–626. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391v4)
7. Wilkinson C. P., Ferris F. L. III, Klein R. E., et al. Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology* 110(9):1677–1682, 2003. [PubMed 13129861](https://pubmed.ncbi.nlm.nih.gov/13129861/)
