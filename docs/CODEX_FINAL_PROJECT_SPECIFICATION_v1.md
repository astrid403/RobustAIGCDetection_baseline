# Codex Final Project Specification v1.0

## Project Title

Robust AI-Generated Image Detection under Cross-Generator and Cross-Dataset Distribution Shift

## Role for Codex

You are helping me build a complete PyTorch baseline project for an undergraduate deep learning / computer vision course project.

Please generate a clean, modular, runnable project according to this specification.

Important constraints:

1. Do not run long training jobs automatically.
2. Do not delete existing files without asking.
3. Do not hard-code machine-specific paths except in example config files.
4. All major scripts must be runnable through YAML config files.
5. The project must be beginner-friendly, because I am still learning deep learning project workflows.
6. Generate readable code with comments, clear error messages, and a detailed README.
7. The first goal is a complete baseline project, not a paper-level system.
8. The project should run locally on my WSL Ubuntu machine with an NVIDIA GPU of about 8GB VRAM.

---

# 1. Local Environment Assumptions

The project will be developed inside WSL Ubuntu using VS Code and conda.

Target project directory:

```text
/home/rong/RobustAIGCDetection_baseline
```

The user will manually create and activate the conda environment. Codex should not automatically create the environment or install packages unless explicitly asked later.

Suggested environment name:

```text
aigc_det_baseline
```

The project should still include:

```text
requirements.txt
environment.yml
README.md
```

The local GPU has approximately 8GB VRAM, so the implementation should be hardware-aware.

The code must support:

```text
1. configurable batch size
2. configurable image size
3. CUDA automatic detection
4. mixed precision training
5. small debug mode
6. subset-based training and evaluation
7. checkpoint saving and resume
8. frozen CLIP feature extraction
9. no multi-GPU requirement
10. no distributed training requirement
```

---

# 2. Project Goal

The project studies AI-generated image detection under distribution shift.

The basic task is binary classification:

```text
input: one image
output: real or fake
```

Label convention:

```text
0 = real
1 = fake / AI-generated
```

The main research question is:

```text
Can AI-generated image detectors trained on known datasets and generators generalize to unseen generators, newer datasets, and post-processed images?
```

The project should not only report in-distribution accuracy. It should evaluate:

```text
1. CIFAKE full sanity check
2. GenImage in-distribution performance
3. GenImage cross-generator generalization
4. GenImage to WildFake external evaluation
5. post-processing robustness
```

---

# 3. Dataset Strategy

The project should support three datasets:

```text
1. CIFAKE
2. GenImage
3. WildFake
```

Dataset roles:

```text
CIFAKE:
full sanity-check dataset

GenImage:
main controlled benchmark for cross-generator evaluation

WildFake:
newer optional external test dataset
```

The project must not require all datasets to be successfully downloaded before any code can run.

Minimum runnable path:

```text
CIFAKE only → full sanity check
```

Then extend to:

```text
GenImage subset → cross-generator experiments
```

Then optional:

```text
WildFake subset → external evaluation
```

---

# 4. Dataset Directory Structure

Default dataset root:

```text
data/raw/
```

Expected structure:

```text
data/raw/
├── CIFAKE/
├── GenImage/
└── WildFake/
```

CIFAKE should be freshly downloaded into the new project folder.

Expected CIFAKE structure:

```text
data/raw/CIFAKE/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

GenImage expected structure may look like:

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

However, do not assume every generator folder exists. The code must scan available folders and print clear messages about missing or available generators.

WildFake structure is currently uncertain. Therefore, WildFake loading must be flexible and should support either:

```text
1. folder-based scanning
2. user-provided metadata CSV
3. manually prepared path,label CSV
```

---

# 5. Dataset Download and Preparation Scripts

Generate dataset preparation scripts under:

```text
scripts/
```

Required scripts:

```text
scripts/download_cifake.sh
scripts/prepare_genimage.sh
scripts/prepare_wildfake.sh
```

## 5.1 CIFAKE

The CIFAKE script should support downloading via Kaggle API.

If Kaggle credentials are missing, print a clear message explaining how to configure `kaggle.json`.

The script should download CIFAKE into:

```text
data/raw/CIFAKE/
```

After download, it should verify that these folders exist:

```text
train/REAL
train/FAKE
test/REAL
test/FAKE
```

## 5.2 GenImage

GenImage downloading may require manual steps depending on source availability.

The GenImage preparation script should support:

```text
1. already extracted GenImage folder
2. manually downloaded zip files
3. user-provided download URLs
4. partial generator availability
```

Do not require full GenImage before running experiments.

## 5.3 WildFake

WildFake is optional in the baseline stage.

The WildFake preparation script should support:

```text
1. already extracted WildFake folder
2. manually provided CSV
3. partial dataset availability
```

If WildFake is unavailable, the rest of the project must still run.

---

# 6. Inventory CSV Requirement

All dataset loading must be CSV-based.

Generate inventories under:

```text
outputs/inventories/
```

Required inventory files:

```text
outputs/inventories/cifake_inventory.csv
outputs/inventories/genimage_inventory.csv
outputs/inventories/wildfake_inventory.csv
```

Each row should represent one image.

Required columns:

```csv
path,label,dataset,split,generator,source,width,height,format,is_valid
```

Column meanings:

```text
path:
absolute or project-relative image path

label:
0 for real, 1 for fake

dataset:
CIFAKE / GenImage / WildFake

split:
train / val / test / unknown / external

generator:
AI generator name if known, otherwise unknown

source:
higher-level source information if available, otherwise unknown

width:
image width

height:
image height

format:
JPEG / PNG / WEBP / etc.

is_valid:
whether the image can be opened successfully
```

The inventory builder should:

```text
1. recursively scan image files
2. support jpg, jpeg, png, webp, bmp if possible
3. open images with Pillow
4. convert corrupted images to is_valid = false
5. not crash on broken files
6. infer labels from folder names where possible
7. print a summary table after scanning
```

---

# 7. Split Generation

Generate split files under:

```text
outputs/splits/
```

Required split files:

```text
outputs/splits/cifake_full_train.csv
outputs/splits/cifake_full_test.csv

outputs/splits/genimage_splitA_train.csv
outputs/splits/genimage_splitA_val.csv
outputs/splits/genimage_splitA_test_unseen.csv

outputs/splits/genimage_splitB_train.csv
outputs/splits/genimage_splitB_val.csv
outputs/splits/genimage_splitB_test_seen.csv
outputs/splits/genimage_splitB_test_unseen.csv

outputs/splits/wildfake_debug_test.csv
outputs/splits/wildfake_baseline_test.csv
outputs/splits/wildfake_final_test.csv
```

The split generator should support a fixed random seed.

Default seed:

```yaml
seed: 42
```

## 7.1 CIFAKE Split

Use official split:

```text
train:
data/raw/CIFAKE/train

test:
data/raw/CIFAKE/test
```

Run full CIFAKE sanity check.

## 7.2 GenImage Split A

Single-generator training, multi-generator testing.

Example default:

```text
train fake generator:
Stable Diffusion V1.4

test fake generators:
ADM
GLIDE
BigGAN
Midjourney
Wukong
VQDM
Stable Diffusion V1.5
```

Purpose:

```text
Test whether a detector trained on one generator can generalize to unseen generators.
```

## 7.3 GenImage Split B

Multi-generator training, held-out-generator testing.

This is the primary baseline split.

Example default:

```text
train fake generators:
Stable Diffusion V1.4
ADM
BigGAN

test fake generators:
GLIDE
Midjourney
Wukong
VQDM
```

Purpose:

```text
Train a more stable detector on several seen generators, then evaluate on held-out unseen generators.
```

The code must allow the generator names to be configured in YAML.

If some generators are missing, print a clear error message listing:

```text
available generators
requested generators
missing generators
```

## 7.4 GenImage Subset Sizes

Support three data sizes:

```text
debug:
per generator real 200 + fake 200

baseline:
per generator real 2,000 + fake 2,000

stronger:
per generator real 5,000 + fake 5,000
```

The code should balance real/fake samples whenever possible.

## 7.5 WildFake Subset Sizes

WildFake is used only as an external test set in the baseline stage.

Support three subset sizes:

```text
debug:
100 real + 100 fake

baseline:
2,000 real + 2,000 fake

final:
5,000 real + 5,000 fake
```

WildFake subsets should be balanced and sampled with a fixed seed.

---

# 8. Model Requirements

Implement two detector types:

```text
1. ResNet detector
2. Frozen CLIP feature + MLP detector
```

Do not implement ViT fine-tuning, full CLIP fine-tuning, AIDE reproduction, or Effort reproduction in the first version.

AIDE and Effort are only references, not required experiments for baseline v1.0.

---

# 9. ResNet Detector

File:

```text
models/resnet_detector.py
```

Support:

```text
resnet18
resnet50
```

Usage:

```text
ResNet18:
debug / quick test

ResNet50:
formal baseline
```

Default formal setting:

```yaml
model_name: resnet50
pretrained: true
freeze_backbone: false
num_classes: 1
```

The ResNet detector should:

```text
1. load torchvision ResNet18 or ResNet50
2. optionally load ImageNet pretrained weights
3. replace final classification layer with a binary output layer
4. output one logit per image
5. support freeze_backbone
```

Loss:

```text
BCEWithLogitsLoss
```

Prediction:

```text
fake_prob = sigmoid(logit)
pred_label = 1 if fake_prob >= 0.5 else 0
```

---

# 10. CLIP-MLP Detector

File:

```text
models/clip_mlp_detector.py
```

Use:

```text
Frozen CLIP image encoder + MLP classifier
```

Default CLIP model:

```yaml
clip_model: ViT-B/32
```

The CLIP encoder must be frozen.

Required behavior:

```text
1. load CLIP image encoder
2. freeze all CLIP parameters
3. extract image feature vectors
4. train a small MLP classifier on top
5. output one logit per image
```

MLP default:

```yaml
mlp_hidden_dim: 512
dropout: 0.2
num_outputs: 1
```

Feature caching is required.

Default:

```yaml
cache_clip_features: true
```

Feature cache path:

```text
outputs/features/{experiment_name}/
```

Feature caching should save extracted CLIP features and labels so that MLP training can reuse them without repeatedly running CLIP.

---

# 11. Preprocessing

All images must be opened with Pillow and converted to RGB:

```python
image = image.convert("RGB")
```

Default image size:

```yaml
image_size: 224
```

ResNet preprocessing:

```text
resize/crop to 224×224
tensor conversion
ImageNet normalization
```

CLIP preprocessing:

```text
use CLIP-compatible preprocessing and normalization
```

Training augmentation should be light and configurable.

Default training augmentation for ResNet:

```text
RandomResizedCrop
RandomHorizontalFlip
optional mild ColorJitter
```

Validation/test preprocessing should be deterministic.

Do not apply training augmentation to test sets.

---

# 12. Training Pipeline

Training script:

```text
training/train.py
```

It must be runnable as:

```bash
python training/train.py --config configs/cifake_full_resnet50.yaml
```

The training pipeline must support:

```text
1. YAML config loading
2. CSV-based dataset loading
3. CUDA automatic detection
4. mixed precision
5. BCEWithLogitsLoss
6. AdamW optimizer
7. validation after each epoch
8. checkpoint saving
9. resume from checkpoint
10. train_log.csv saving
```

Default optimizer:

```yaml
optimizer: AdamW
```

Default learning rates:

```yaml
resnet_learning_rate: 0.0001
clip_mlp_learning_rate: 0.001
```

Mixed precision:

```yaml
mixed_precision: true
```

If CUDA is unavailable, disable mixed precision automatically.

---

# 13. Recommended Config Files

Generate these config files:

```text
configs/debug_resnet18.yaml
configs/cifake_full_resnet50.yaml
configs/genimage_splitA_resnet50.yaml
configs/genimage_splitB_resnet50.yaml
configs/genimage_splitB_clip_mlp.yaml
configs/wildfake_external_resnet50.yaml
configs/wildfake_external_clip_mlp.yaml
configs/robustness_eval.yaml
```

## 13.1 Debug Config

```yaml
experiment_name: debug_resnet18
model_type: resnet
model_name: resnet18
image_size: 224
batch_size: 16
epochs: 1
max_samples_per_class: 200
pretrained: false
freeze_backbone: false
mixed_precision: true
learning_rate: 0.0001
seed: 42
```

## 13.2 CIFAKE Full Config

```yaml
experiment_name: cifake_full_resnet50
model_type: resnet
model_name: resnet50
dataset: CIFAKE
train_csv: outputs/splits/cifake_full_train.csv
val_csv: outputs/splits/cifake_full_test.csv
test_csv: outputs/splits/cifake_full_test.csv
image_size: 224
batch_size: 32
epochs: 3
pretrained: true
freeze_backbone: false
mixed_precision: true
learning_rate: 0.0001
seed: 42
```

If 8GB GPU runs out of memory, user can reduce:

```yaml
batch_size: 16
```

## 13.3 GenImage Split B ResNet50 Config

```yaml
experiment_name: genimage_splitB_resnet50
model_type: resnet
model_name: resnet50
dataset: GenImage
train_csv: outputs/splits/genimage_splitB_train.csv
val_csv: outputs/splits/genimage_splitB_val.csv
test_csv: outputs/splits/genimage_splitB_test_unseen.csv
image_size: 224
batch_size: 16
epochs: 5
pretrained: true
freeze_backbone: false
mixed_precision: true
learning_rate: 0.0001
samples_per_generator_per_class: 2000
seed: 42
```

## 13.4 GenImage Split B CLIP-MLP Config

```yaml
experiment_name: genimage_splitB_clip_mlp
model_type: clip_mlp
clip_model: ViT-B/32
dataset: GenImage
train_csv: outputs/splits/genimage_splitB_train.csv
val_csv: outputs/splits/genimage_splitB_val.csv
test_csv: outputs/splits/genimage_splitB_test_unseen.csv
image_size: 224
batch_size: 32
epochs: 10
learning_rate: 0.001
mlp_hidden_dim: 512
dropout: 0.2
cache_clip_features: true
mixed_precision: true
seed: 42
```

---

# 14. Evaluation Pipeline

Evaluation script:

```text
evaluation/evaluate.py
```

It must be runnable as:

```bash
python evaluation/evaluate.py --config configs/cifake_full_resnet50.yaml --checkpoint outputs/checkpoints/cifake_full_resnet50/best_model.pt
```

Evaluation must compute:

```text
accuracy
precision
recall
F1-score
AUROC
confusion matrix
```

Evaluation must save:

```text
outputs/metrics/{experiment_name}_metrics.csv
outputs/predictions/{experiment_name}_predictions.csv
outputs/figures/{experiment_name}/confusion_matrix.png
outputs/figures/{experiment_name}/roc_curve.png
outputs/error_cases/{experiment_name}/
```

Prediction CSV columns:

```csv
path,label,pred_label,fake_prob,dataset,generator,source,is_correct
```

Metric CSV columns:

```csv
experiment_name,model,train_dataset,test_dataset,test_split,train_generators,test_generators,num_real,num_fake,accuracy,precision,recall,f1,auroc
```

---

# 15. Post-processing Robustness

Robustness script:

```text
robustness/evaluate_robustness.py
```

It must be runnable as:

```bash
python robustness/evaluate_robustness.py --config configs/robustness_eval.yaml
```

Support transforms:

```text
1. clean
2. JPEG compression
3. resize down-up
4. Gaussian blur
```

Default settings:

```yaml
jpeg_quality: 70
resize_scale: 0.5
blur_radius: 1.0
```

The transforms should be applied during evaluation only. Do not overwrite original images.

Robustness metrics:

```text
clean AUROC
processed AUROC
robustness drop = clean AUROC - processed AUROC
```

Save:

```text
outputs/metrics/{experiment_name}_robustness.csv
outputs/figures/{experiment_name}/robustness_bar_chart.png
```

Robustness CSV columns:

```csv
experiment_name,model,test_set,transform,clean_auroc,processed_auroc,robustness_drop
```

---

# 16. Output Requirements

The project must generate organized outputs.

Required output directories:

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

## 16.1 Checkpoints

Save:

```text
outputs/checkpoints/{experiment_name}/best_model.pt
outputs/checkpoints/{experiment_name}/last_model.pt
outputs/checkpoints/{experiment_name}/config_used.yaml
```

`best_model.pt` should be selected using validation AUROC or validation F1.

## 16.2 Logs

Save:

```text
outputs/logs/{experiment_name}/train_log.csv
```

Columns:

```csv
epoch,train_loss,val_loss,val_accuracy,val_precision,val_recall,val_f1,val_auroc,learning_rate
```

## 16.3 Figures

Save:

```text
outputs/figures/{experiment_name}/confusion_matrix.png
outputs/figures/{experiment_name}/roc_curve.png
outputs/figures/{experiment_name}/train_loss_curve.png
outputs/figures/{experiment_name}/val_auroc_curve.png
outputs/figures/{experiment_name}/robustness_bar_chart.png
```

Use matplotlib.

## 16.4 Error Cases

Save at most 50 images for each error type:

```text
outputs/error_cases/{experiment_name}/fake_predicted_real/
outputs/error_cases/{experiment_name}/real_predicted_fake/
```

## 16.5 Summary File

Generate:

```text
outputs/metrics/summary_all_experiments.csv
```

Columns:

```csv
experiment_name,model,train_setting,test_setting,accuracy,f1,auroc,notes
```

---

# 17. Main Experiments

The project should support these experiments.

## Experiment 1: CIFAKE Full Sanity Check

Purpose:

```text
Verify that the full pipeline works.
```

Train:

```text
CIFAKE full train set
```

Test:

```text
CIFAKE full test set
```

Model:

```text
ResNet50
```

Optional:

```text
CLIP-MLP
```

## Experiment 2: GenImage In-distribution Evaluation

Purpose:

```text
Evaluate performance on seen generators.
```

Train:

```text
GenImage seen generators
```

Test:

```text
GenImage seen generators validation/test split
```

## Experiment 3: GenImage Cross-generator Evaluation

Purpose:

```text
Evaluate generalization to unseen generators.
```

Primary split:

```text
Split B
```

Train fake generators:

```text
Stable Diffusion V1.4
ADM
BigGAN
```

Test fake generators:

```text
GLIDE
Midjourney
Wukong
VQDM
```

The generator names should be configurable.

Models:

```text
ResNet50
CLIP-MLP
```

## Experiment 4: GenImage to WildFake External Evaluation

Purpose:

```text
Evaluate whether a model trained on GenImage generalizes to a newer external dataset.
```

Train:

```text
GenImage selected subset
```

Test:

```text
WildFake balanced subset
```

WildFake is optional. If WildFake is unavailable, the project should keep the interface and print clear instructions.

## Experiment 5: Post-processing Robustness

Purpose:

```text
Evaluate whether model performance drops after common image post-processing.
```

Transforms:

```text
JPEG compression
resize down-up
Gaussian blur
```

Report:

```text
clean AUROC
processed AUROC
robustness drop
```

---

# 18. Scripts

Generate these scripts under:

```text
scripts/
```

Required scripts:

```text
scripts/download_cifake.sh
scripts/prepare_genimage.sh
scripts/prepare_wildfake.sh
scripts/run_debug_smoke_test.sh
scripts/run_cifake_full.sh
scripts/run_genimage_splitA_resnet50.sh
scripts/run_genimage_splitB_resnet50.sh
scripts/run_genimage_splitB_clip_mlp.sh
scripts/run_wildfake_external.sh
scripts/run_robustness.sh
```

Do not automatically run long experiments after code generation.

The user will manually run scripts.

The first script to run should be:

```bash
bash scripts/run_debug_smoke_test.sh
```

This script should test:

```text
inventory generation
split generation
1 epoch training
evaluation
metrics saving
```

on a very small sample.

---

# 19. Recommended Project Structure

Generate this structure:

```text
RobustAIGCDetection_baseline/
├── configs/
│   ├── debug_resnet18.yaml
│   ├── cifake_full_resnet50.yaml
│   ├── genimage_splitA_resnet50.yaml
│   ├── genimage_splitB_resnet50.yaml
│   ├── genimage_splitB_clip_mlp.yaml
│   ├── wildfake_external_resnet50.yaml
│   ├── wildfake_external_clip_mlp.yaml
│   └── robustness_eval.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── data_pipeline/
│   ├── __init__.py
│   ├── build_inventory.py
│   ├── make_splits.py
│   ├── csv_image_dataset.py
│   └── download_helpers.py
│
├── models/
│   ├── __init__.py
│   ├── resnet_detector.py
│   └── clip_mlp_detector.py
│
├── training/
│   ├── train.py
│   ├── losses.py
│   └── engine.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   ├── plots.py
│   └── error_analysis.py
│
├── robustness/
│   ├── transforms.py
│   └── evaluate_robustness.py
│
├── scripts/
│   ├── download_cifake.sh
│   ├── prepare_genimage.sh
│   ├── prepare_wildfake.sh
│   ├── run_debug_smoke_test.sh
│   ├── run_cifake_full.sh
│   ├── run_genimage_splitA_resnet50.sh
│   ├── run_genimage_splitB_resnet50.sh
│   ├── run_genimage_splitB_clip_mlp.sh
│   ├── run_wildfake_external.sh
│   └── run_robustness.sh
│
├── outputs/
│   ├── inventories/
│   ├── splits/
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   ├── features/
│   └── error_cases/
│
├── docs/
│   ├── PROJECT_SPEC.md
│   └── EXPERIMENT_PLAN.md
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

# 20. Requirements

Include at least:

```text
torch
torchvision
pandas
numpy
Pillow
tqdm
scikit-learn
matplotlib
PyYAML
kaggle
open_clip_torch or clip-compatible package
```

If using `open_clip_torch`, document that clearly in README.

---

# 21. README Requirements

Write a beginner-friendly README.

It must include:

```text
1. project overview
2. environment setup
3. dataset preparation
4. expected data directory structure
5. how to download CIFAKE
6. how to manually prepare GenImage
7. how to manually prepare WildFake
8. how to build inventory CSV files
9. how to generate splits
10. how to run debug smoke test
11. how to run CIFAKE full sanity check
12. how to run GenImage Split B ResNet50
13. how to run GenImage Split B CLIP-MLP
14. how to run WildFake external evaluation
15. how to run robustness evaluation
16. where outputs are saved
17. how to interpret metrics
18. common errors and troubleshooting
```

The README should include example commands.

---

# 22. Documentation Files

Generate:

```text
docs/PROJECT_SPEC.md
docs/EXPERIMENT_PLAN.md
```

`PROJECT_SPEC.md` should summarize the project goal, data strategy, model strategy, and output requirements.

`EXPERIMENT_PLAN.md` should list all experiments and expected output files.

---

# 23. Error Handling Requirements

The code must print clear error messages.

Examples:

```text
CIFAKE folder not found at data/raw/CIFAKE.
Please run bash scripts/download_cifake.sh or update your config file.
```

```text
GenImage requested generators are missing.
Available generators: ADM, BigGAN
Requested generators: Stable Diffusion V1.4, GLIDE
Missing generators: Stable Diffusion V1.4, GLIDE
```

```text
WildFake is not available.
Skipping WildFake external evaluation.
You can add WildFake later by placing files under data/raw/WildFake or providing a metadata CSV.
```

---

# 24. Reproducibility Requirements

The project must support reproducibility.

Required:

```text
1. fixed random seed
2. saved split CSV files
3. saved config_used.yaml
4. saved train_log.csv
5. saved metrics CSV
6. saved prediction CSV
7. saved checkpoint files
```

Default seed:

```yaml
seed: 42
```

---

# 25. What Not to Do in Baseline v1.0

Do not implement these in the first version:

```text
1. ViT fine-tuning
2. full CLIP fine-tuning
3. AIDE full reproduction
4. Effort-AIGI-Detection full reproduction
5. multi-GPU training
6. distributed training
7. large hyperparameter search
8. full WildFake training
9. mandatory full GenImage download
10. automatic long experiment execution
```

These can be added later after the baseline is stable.

---

# 26. Implementation Phases

Please implement incrementally.

## Phase 1: Project Skeleton

Create the folder structure, config files, requirements, README skeleton, and docs skeleton.

## Phase 2: Dataset Inventory and Split

Implement:

```text
data_pipeline/build_inventory.py
data_pipeline/make_splits.py
data_pipeline/csv_image_dataset.py
```

Support CIFAKE first.

## Phase 3: ResNet Training and Evaluation

Implement:

```text
models/resnet_detector.py
training/train.py
evaluation/evaluate.py
evaluation/metrics.py
evaluation/plots.py
```

Make CIFAKE debug and CIFAKE full runnable.

## Phase 4: GenImage Support

Implement GenImage inventory and generator-level split logic.

Support Split A and Split B.

## Phase 5: CLIP-MLP

Implement:

```text
models/clip_mlp_detector.py
CLIP feature extraction
feature caching
MLP training
```

## Phase 6: WildFake Optional External Evaluation

Implement flexible WildFake CSV/folder support.

Do not block the project if WildFake is missing.

## Phase 7: Robustness Evaluation

Implement:

```text
robustness/transforms.py
robustness/evaluate_robustness.py
```

Support JPEG, resize down-up, and Gaussian blur.

## Phase 8: Documentation Polish

Finish README, PROJECT_SPEC.md, and EXPERIMENT_PLAN.md.

---

# 27. First Command After Generation

After generating the project, do not start full experiments automatically.

Instead, tell me to run:

```bash
bash scripts/run_debug_smoke_test.sh
```

Then, after debug works:

```bash
bash scripts/download_cifake.sh
bash scripts/run_cifake_full.sh
```

Then later:

```bash
bash scripts/prepare_genimage.sh
bash scripts/run_genimage_splitB_resnet50.sh
bash scripts/run_genimage_splitB_clip_mlp.sh
```

WildFake and robustness should be run only after the earlier steps work.

---

# 28. Final Deliverables

After Codex finishes generating the project, the directory should contain:

```text
1. runnable PyTorch project code
2. config files
3. dataset preparation scripts
4. training scripts
5. evaluation scripts
6. robustness scripts
7. output directory structure
8. README.md
9. docs/PROJECT_SPEC.md
10. docs/EXPERIMENT_PLAN.md
11. requirements.txt
12. environment.yml
```

The project should be ready for manual step-by-step execution in VS Code + WSL.

---

# 29. Start Now

Please generate the project files according to this specification.

Do not run long experiments.

Do not delete files without permission.

After generating files, summarize:

```text
1. what files were created
2. how to install dependencies
3. how to run the debug smoke test
4. how to run the CIFAKE full sanity check
5. what to do next
```
