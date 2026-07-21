# Robust AI-Generated Image Detection Baseline

This is a clean PyTorch baseline project for detecting real vs AI-generated images under distribution shift. It supports CIFAKE, GenImage, and Defactify external evaluation. WildFake is retained as Milestone 2 legacy functionality.

The binary label convention is:

- `0`: real
- `1`: fake / AI-generated

## Current Completed Baseline Results

Report-ready summaries are available at:

- `outputs/metrics/summary_all_experiments.csv`
- `docs/RESULTS_SUMMARY.md`

Completed baseline stages:

- CIFAKE full ResNet50 sanity check.
- GenImage fallback available-generator cross-generator experiments with ResNet50 and CLIP-MLP.
- Post-processing robustness evaluation on the GenImage fallback unseen-generator test split.
- WildFake subset external evaluation with ResNet50 and CLIP-MLP (Milestone 2 legacy).
- Defactify official-full and deterministic-balanced external evaluation with ResNet50 and CLIP-MLP.
- Defactify balanced post-processing robustness evaluation.

Important caveats:

- The official default GenImage Split B is pending because `Stable Diffusion V1.4` is missing locally.
- The completed GenImage experiments are fallback available-generator experiments, not the official default Split B.
- WildFake external evaluation was done on a CelebA-HQ/DDIM subset, not full WildFake.
- Defactify is evaluated only as an external test set; no Defactify image was used to train the reported checkpoints.

Defactify results and reproducibility details are reported in [`docs/DEFACTIFY_RESULTS.md`](docs/DEFACTIFY_RESULTS.md). The main external-test AUROC is 0.647152 for ResNet50 and 0.807892 for CLIP-MLP; deterministic balanced AUROC is 0.644212 and 0.807510 respectively.

## Environment Setup

Create and activate a conda environment manually:

```bash
conda env create -f environment.yml
conda activate aigc_det_baseline
```

Or install with pip:

```bash
pip install -r requirements.txt
```

The CLIP-MLP baseline uses `open_clip_torch`.

## Data Layout

Put datasets under:

```text
data/raw/
├── CIFAKE/
├── GenImage/
└── WildFake/
```

CIFAKE should look like:

```text
data/raw/CIFAKE/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

GenImage can be partial. Available generator folders are scanned automatically.

## Download CIFAKE

Configure Kaggle credentials first:

```bash
mkdir -p ~/.kaggle
# place kaggle.json in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Then run:

```bash
bash scripts/download_cifake.sh
```

## Prepare GenImage

Manually download and extract GenImage generator folders into `data/raw/GenImage/`.

Expected folder pattern:

```text
data/raw/GenImage/
├── ADM/
│   ├── train/
│   │   ├── ai/
│   │   └── nature/
│   └── val/
│       ├── ai/
│       └── nature/
├── BigGAN/
├── GLIDE/
├── Midjourney/
├── Stable Diffusion V1.4/
├── Stable Diffusion V1.5/
├── VQDM/
└── Wukong/
```

For the default Split B experiment, the minimum configured folders are:

- Training generators: `Stable Diffusion V1.4`, `ADM`, `BigGAN`
- Unseen test generators: `GLIDE`, `Midjourney`, `Wukong`, `VQDM`

Then run:

```bash
bash scripts/prepare_genimage.sh
```

If requested generators are missing, the split script prints available, requested, and missing generator names.

## Prepare Defactify

Defactify is read directly from the Hugging Face cache. Images are not exported into a second directory tree. Pin the dataset to a verified commit SHA before preparing manifests:

```bash
export DEFACTIFY_REVISION=787334f7857fa54f29027a7f09c30e895ad486ef
bash scripts/prepare_defactify.sh
```

This creates lightweight manifests under `outputs/inventories/` and `outputs/splits/`. Run the non-training smoke evaluation before the full external evaluation:

```bash
bash scripts/run_defactify_smoke_test.sh
bash scripts/run_defactify_external.sh
```

The smoke and external scripts reuse the completed GenImage checkpoints and use new Defactify experiment names, so they do not overwrite Milestone 2 outputs.

## Prepare WildFake (Milestone 2 legacy)

Folder-based scanning:

```bash
bash scripts/prepare_wildfake.sh
```

Metadata CSV scanning:

```bash
bash scripts/prepare_wildfake.sh --metadata path/to/wildfake.csv
```

The metadata CSV must contain at least `path,label`.

## Build Inventories

Inventories are saved under `outputs/inventories/`.

```bash
python data_pipeline/build_inventory.py --dataset CIFAKE
python data_pipeline/build_inventory.py --dataset GenImage
python data_pipeline/build_inventory.py --dataset WildFake
```

Each inventory row includes:

```csv
sample_id,storage_backend,path,dataset,dataset_id,dataset_revision,hf_split,row_index,label,label_b,generator,source,caption,width,height,format,is_valid,split
```

## Generate Splits

Splits are saved under `outputs/splits/`.

```bash
python data_pipeline/make_splits.py --config configs/debug_resnet18.yaml --dataset CIFAKE
python data_pipeline/make_splits.py --config configs/genimage_splitB_resnet50.yaml --dataset GenImage
python data_pipeline/make_splits.py --config configs/wildfake_external_resnet50.yaml --dataset WildFake
```

## Debug Smoke Test

Run this first after CIFAKE is available:

```bash
bash scripts/run_debug_smoke_test.sh
```

It builds inventory, generates splits, trains ResNet18 for one epoch on a small subset, evaluates, and saves metrics.

## CIFAKE Full Sanity Check

```bash
bash scripts/run_cifake_full.sh
```

If an 8GB GPU runs out of memory, edit `configs/cifake_full_resnet50.yaml` and set:

```yaml
batch_size: 16
```

## GenImage Split B ResNet50

```bash
bash scripts/run_genimage_splitB_resnet50.sh
```

This trains on configured seen generators and evaluates on held-out generators.

## GenImage Split B CLIP-MLP

```bash
bash scripts/run_genimage_splitB_clip_mlp.sh
```

CLIP image features are cached under `outputs/features/{experiment_name}/`.

## WildFake External Evaluation (Milestone 2 legacy)

Run after a GenImage checkpoint exists:

```bash
bash scripts/run_wildfake_external.sh
```

WildFake is no longer the current external benchmark. Its code and historical results remain available for Milestone 2 reproducibility.

## Robustness Evaluation

Edit `configs/robustness_eval.yaml` so `checkpoint` and `test_csv` point to the desired model and split:

```bash
bash scripts/run_robustness.sh
```

Supported transforms:

- clean
- JPEG compression
- resize down-up
- Gaussian blur

## Outputs

Important output locations:

- Inventories: `outputs/inventories/`
- Splits: `outputs/splits/`
- Checkpoints: `outputs/checkpoints/{experiment_name}/`
- Logs: `outputs/logs/{experiment_name}/train_log.csv`
- Metrics: `outputs/metrics/`
- Predictions: `outputs/predictions/`
- Figures: `outputs/figures/{experiment_name}/`
- CLIP features: `outputs/features/{experiment_name}/`
- Error cases: `outputs/error_cases/{experiment_name}/`

## Metrics

Evaluation reports:

- accuracy
- precision
- recall
- F1-score
- AUROC
- confusion matrix

AUROC is often the most useful metric for binary detection because it uses prediction scores rather than a single threshold.

## Common Errors

`CIFAKE folder not found at data/raw/CIFAKE.`

Run:

```bash
bash scripts/download_cifake.sh
```

`Inventory not found`

Build inventories before generating splits:

```bash
python data_pipeline/build_inventory.py --dataset CIFAKE
```

`GenImage requested generators are missing`

Check which folders exist under `data/raw/GenImage/` and edit the generator lists in the YAML config.

CUDA out of memory:

Lower `batch_size` in the config, usually from `32` to `16` or `8`.

`open_clip_torch is required`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Recommended Order

```bash
bash scripts/run_debug_smoke_test.sh
```

Then:

```bash
bash scripts/download_cifake.sh
bash scripts/run_cifake_full.sh
```

Later:

```bash
bash scripts/prepare_genimage.sh
bash scripts/run_genimage_splitB_resnet50.sh
bash scripts/run_genimage_splitB_clip_mlp.sh
```

Run WildFake and robustness only after the earlier steps work.
