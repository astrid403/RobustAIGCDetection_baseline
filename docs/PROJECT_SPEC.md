# Project Spec

## Goal

This project builds a beginner-friendly PyTorch baseline for binary AI-generated image detection.

Label convention:

- `0`: real
- `1`: fake / AI-generated

The main question is whether detectors trained on known datasets and generators generalize to unseen generators, external datasets, and post-processed images.

## Data Strategy

All loading is CSV-based. Dataset folders are scanned into inventories under `outputs/inventories/`, then reproducible split CSV files are written under `outputs/splits/`.

Supported datasets:

- CIFAKE: first full sanity check.
- GenImage: main benchmark for seen and unseen generator experiments.
- WildFake: optional external test set.

The code is designed to keep working when GenImage or WildFake are unavailable.

## Model Strategy

Baseline v1 implements:

- ResNet18 / ResNet50 binary classifier.
- Frozen CLIP image encoder plus trainable MLP classifier.

The baseline does not implement ViT fine-tuning, full CLIP fine-tuning, AIDE reproduction, Effort reproduction, multi-GPU, or distributed training.

## Output Strategy

Outputs are organized under:

```text
outputs/
├── inventories/
├── splits/
├── checkpoints/
├── logs/
├── metrics/
├── predictions/
├── figures/
├── features/
└── error_cases/
```

Every run saves configs, logs, metrics, predictions, figures, and checkpoints when applicable.

