#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="data/raw/CIFAKE"
DATASET_SLUG="${CIFAKE_KAGGLE_SLUG:-birdy654/cifake-real-and-ai-generated-synthetic-images}"

mkdir -p "$TARGET_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI was not found. Install requirements first: pip install -r requirements.txt"
  exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "Kaggle credentials not found at ~/.kaggle/kaggle.json"
  echo "Create an API token on Kaggle, place it there, then run: chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

echo "Downloading CIFAKE from Kaggle dataset: $DATASET_SLUG"
kaggle datasets download -d "$DATASET_SLUG" -p "$TARGET_DIR" --unzip

for folder in train/REAL train/FAKE test/REAL test/FAKE; do
  if [ ! -d "$TARGET_DIR/$folder" ]; then
    echo "Expected folder missing: $TARGET_DIR/$folder"
    echo "Please inspect the downloaded archive and move files into the expected CIFAKE structure."
    exit 1
  fi
done

echo "CIFAKE is ready at $TARGET_DIR"

