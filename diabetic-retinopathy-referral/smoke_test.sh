#!/usr/bin/env bash
# End-to-end check on synthetic images: a few minutes on a CPU, under a minute on a GPU.
# The scores it prints are meaningless; it only proves that every step runs.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf smoke

python tools/make_dummy_data.py --out smoke/raw

python prepare_data.py train-sets --aptos-dir smoke/raw/aptos \
  --eyepacs-labels smoke/raw/eyepacs/trainLabels.csv --eyepacs-images smoke/raw/eyepacs/train \
  --out-dir smoke/processed --size 256

python prepare_data.py external --name idrid_style --csv smoke/raw/external/labels.csv \
  --image-dir smoke/raw/external/images --image-col "Image name" --label-col "Retinopathy grade" \
  --dme-col "Risk of macular edema" --out-dir smoke/processed --size 256

python train.py --manifest smoke/processed/manifest.csv --out smoke/run --backbone resnet18 \
  --no-pretrained --img-size 256 --epochs 2 --batch-size 16 --accum-steps 1 --workers 2

python evaluate.py --run-dir smoke/run --csv smoke/processed/manifest.csv --split test --n-boot 200 --workers 2
python evaluate.py --run-dir smoke/run --csv smoke/processed/external_idrid_style.csv --n-boot 200 --workers 2

echo
echo "Smoke test passed. You can delete the smoke/ folder."
