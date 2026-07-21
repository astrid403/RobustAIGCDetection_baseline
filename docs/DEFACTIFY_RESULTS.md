# Defactify External Evaluation Results

## Reproducibility

- Dataset: `Rajarshi-Roy-research/Defactify_Image_Dataset`
- Revision: `787334f7857fa54f29027a7f09c30e895ad486ef`
- Official test manifest SHA256: `d17046f8af94215309c613e6aeab4d2aba33cd0bc671c8e3f666e9e6d27b4158`
- Balanced manifest SHA256: `439396912b1ebb91f03f457813015f1a365d3305297103e3fd9fbc6a7642461d`
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB
- Software: Python 3.10.20, PyTorch 2.11.0+cu128
- ResNet50 checkpoint SHA256: `9bca8ef1e23351d58198dbafa261e1888bbbb726fca969292ad667d426d5836a`
- CLIP-MLP checkpoint SHA256: `6e0bb27816491f7806c09601be5d453b5ae5066263c729feb5e8b72ee2db49b9`

Defactify was used only for external evaluation. Both checkpoints were trained on the completed GenImage available-generator split. WildFake results remain Milestone 2 legacy results.

## Data validation

The fixed revision contains 96,000 records: 42,000 train, 9,000 validation, and 45,000 test. Each official split contains equal counts for Label_B 0 through 5. Label_A and Label_B consistency, unique sample IDs, row indices, all 36 smoke images, and deterministic manifests were validated before inference.

The official test has 7,500 real and 37,500 fake images. The deterministic balanced subset contains all 7,500 real images and 1,500 images from each of the five fake generators. Balanced metrics were computed by filtering the completed full predictions by sample ID; the models did not run a second clean inference on those images.

## Performance calibration

| Model | Safe batch | Peak GPU memory | Throughput | First batch load | Estimated full | Actual full |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 32 | 710 MiB | 156.6 img/s | 0.93 s | 4.8 min | 4.4 min |
| CLIP-MLP ViT-B/32 | 64 | 745 MiB | 151.2 img/s | 1.18 s | 5.0 min | 4.6 min |

Calibration used 480 deterministic official-test samples, with 80 samples per Label_B. It did not change the checkpoint, image size, preprocessing, or formal sample scope.

## Main results

| Model | Evaluation | N | Accuracy | Balanced Accuracy | Precision | Recall | F1 | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | Official full | 45,000 | 0.623311 | 0.579213 | 0.868883 | 0.645360 | 0.740624 | 0.647152 |
| ResNet50 | Balanced from full predictions | 15,000 | 0.577200 | 0.577200 | 0.568424 | 0.641333 | 0.602681 | 0.644212 |
| CLIP-MLP | Official full | 45,000 | 0.837222 | 0.663667 | 0.885623 | 0.924000 | 0.904405 | 0.807892 |
| CLIP-MLP | Balanced from full predictions | 15,000 | 0.662400 | 0.662400 | 0.606973 | 0.921467 | 0.731865 | 0.807510 |

CLIP-MLP is substantially stronger on the external dataset. Its official-full AUROC exceeds ResNet50 by 0.160740. The large difference between official accuracy and balanced accuracy, especially for CLIP-MLP, demonstrates why the 1:5 real/fake class ratio must be reported explicitly.

## Generator analysis

### Official full AUROC

| Generator | ResNet50 | CLIP-MLP |
|---|---:|---:|
| Stable Diffusion 2.1 | 0.587731 | 0.801037 |
| Stable Diffusion XL | 0.346802 | 0.779587 |
| Stable Diffusion 3 | 0.433850 | 0.769707 |
| DALL-E 3 | 0.974783 | 0.871161 |
| Midjourney 6 | 0.892594 | 0.817967 |

ResNet50 transfers well to DALL-E 3 and Midjourney 6 but fails badly on SDXL and SD3. CLIP-MLP is more consistent across generators; SD3 is its hardest generator and DALL-E 3 its easiest by AUROC.

Balanced per-generator tables additionally record mean fake probability, false-negative counts, and false-negative rates in:

- `outputs/metrics/defactify_external_available_resnet50_balanced_by_generator.csv`
- `outputs/metrics/defactify_external_available_clip_mlp_balanced_by_generator.csv`

## Balanced robustness

| Model | Transform | AUROC | AUROC change from clean | Balanced Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| ResNet50 | clean | 0.644212 | 0.000000 | 0.577200 | 0.602681 |
| ResNet50 | JPEG 70 | 0.636277 | -0.007936 | 0.583533 | 0.593109 |
| ResNet50 | resize 0.5 down-up | 0.504015 | -0.140197 | 0.424933 | 0.545234 |
| ResNet50 | Gaussian blur 1.0 | 0.525583 | -0.118629 | 0.439600 | 0.569100 |
| CLIP-MLP | clean | 0.807511 | 0.000000 | 0.662400 | 0.731865 |
| CLIP-MLP | JPEG 70 | 0.833997 | +0.026486 | 0.706600 | 0.755541 |
| CLIP-MLP | resize 0.5 down-up | 0.716374 | -0.091137 | 0.560467 | 0.681297 |
| CLIP-MLP | Gaussian blur 1.0 | 0.695549 | -0.111963 | 0.536600 | 0.673800 |

Resize and blur cause the largest degradation for both models. JPEG has little effect on ResNet50 and improves the measured CLIP-MLP result on this fixed balanced subset; this negative robustness drop is reported as observed and is not treated as a general augmentation claim.

## Integrity checks

- Full metrics were independently recomputed from 45,000-row predictions and matched the evaluation output within `1e-12`.
- Both full prediction files contain unique sample IDs and exactly 7,500 samples per Label_B.
- Both balanced prediction files contain exactly the 15,000 IDs in the deterministic balanced manifest.
- CLIP cache contains 45,000 by 512 features and matches the dataset revision, manifest hash, model, and preprocessing signature.
- All official evaluation and robustness processes exited with code 0.
- Data, manifests, checkpoints, feature caches, predictions, logs, and error-case images remain ignored and are not committed.
