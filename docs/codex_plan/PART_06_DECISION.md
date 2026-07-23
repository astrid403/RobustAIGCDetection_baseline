# Part 06 Go/No-Go Decision Record

## Status

- Decision code: `GO_A1`.
- Recommendation: select penultimate-only A1 as the final proposed route.
- Approval status: awaiting explicit user approval.
- Model freeze: not active until approval.
- Evidence scope: GenImage validation and unseen only.
- Defactify used: no.

## Comparability audit

All four rows use seed 42, the frozen GenImage train/validation/unseen
manifests, batch size 32, 10 epochs, AdamW at learning rate 0.001,
BCEWithLogitsLoss, frozen OpenAI ViT-B-32 embeddings, validation-AUROC
checkpoint selection, and fixed threshold 0.5.

The split SHA256 values are:

- train: `2b2e1112f044f079a2ef4fa9255d42e850359861c934e8c3bf938527701b5bd8`;
- validation: `5c73cbc97dc41e8ef63b2021fb1758201b4b484f04d4ebb1559f3535c4e44698`;
- unseen: `0852a4bbe4ea67efecf7756b697e04a7a5cb5623999caf3fc3dfd625df063c07`.

B1, A1, and P were run once under the v2 cache contract. B2 is the frozen
historical seed-42 baseline: its complete unseen metrics were recomputed from
its frozen predictions, and its complete validation metrics were recomputed
from its frozen checkpoint using the verified v2 final validation features.
No B2 training or unseen inference was repeated.

## Seed-42 evidence

| Model | Scope | AUROC | AUPRC | Bal. Acc. | Macro-F1 | Real R. | Fake R. | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B1 linear | validation | 0.963592 | 0.964369 | 0.883333 | 0.883307 | 0.898333 | 0.868333 | 513 |
| B1 linear | unseen | 0.935733 | 0.934692 | 0.849500 | 0.849018 | 0.906000 | 0.793000 | 513 |
| B2 final MLP | validation | 0.996503 | 0.996419 | 0.971667 | 0.971665 | 0.965000 | 0.978333 | 263,169 |
| B2 final MLP | unseen | 0.983852 | 0.981709 | 0.933000 | 0.932920 | 0.967500 | 0.898500 | 263,169 |
| A1 penultimate MLP | validation | 0.996914 | 0.996798 | 0.973333 | 0.973333 | 0.970000 | 0.976667 | 263,169 |
| A1 penultimate MLP | unseen | **0.988048** | **0.987263** | **0.936250** | **0.936171** | 0.971500 | **0.901000** | 263,169 |
| P dual-level fusion | validation | **0.997183** | **0.997106** | **0.974167** | **0.974167** | **0.971667** | 0.976667 | 262,657 |
| P dual-level fusion | unseen | 0.986873 | 0.985283 | 0.935500 | 0.935404 | **0.974000** | 0.897000 | 262,657 |

There is no class-collapse signature: every model predicts both classes, all
confusion-matrix cells are finite, and both class recalls are nonzero. B1 is a
useful low-capacity baseline but is not competitive. P has the best validation
metrics and unseen real recall, but A1 has the best unseen AUROC, AUPRC,
balanced accuracy, macro-F1, and fake recall.

## Why A2 was not run

The existing B2 is already the required parameter-matched final-only control.
B2 and A1 each have 263,169 trainable parameters; P has 262,657, only 512
fewer (0.19%). Therefore head capacity is already controlled closely enough to
attribute the B2/A1 difference to feature level. A new A2 would duplicate B2
without answering a new question and would violate the minimal-scope rule.

## Decision

`GO_A1` is recommended because:

1. A1 leads the approved candidates on the unseen split in five central
   measures: AUROC, AUPRC, balanced accuracy, macro-F1, and fake recall.
2. A1 improves unseen AUROC by 0.004195 over B2 and 0.001174 over P.
3. A1 uses one feature tensor rather than two: 17,407,055 cached bytes versus
   34,814,110 for P, with lower measured training/evaluation wall time.
4. P's small validation advantage does not carry to the held-out generators.
5. A1 preserves the approved scientific hypothesis while remaining the
   simplest credible route.

The degradation-augmentation fallback is not selected because A1 already
provides a valid improvement signal. No augmentation pilot, fourth algorithm,
or new module is authorized. `SWITCH_FALLBACK` and `BLOCKED` therefore do not
apply.

## Proposed freeze contract

After user approval, freeze exactly:

- route: A1 penultimate-only;
- encoder: frozen OpenAI CLIP ViT-B-32;
- feature: Part 3 `penultimate`, 512-D, L2-normalized;
- head: `Linear(512,512)-ReLU-Dropout(0.2)-Linear(512,1)`;
- trainable parameters: 263,169;
- data: frozen GenImage train/validation manifests;
- optimizer/loss: AdamW, learning rate 0.001, BCEWithLogitsLoss;
- budget: 10 epochs, batch size 32, no added augmentation;
- checkpoint: maximum GenImage validation AUROC;
- threshold: fixed 0.5, never selected on unseen or Defactify;
- formal seeds: 42, 43, 44;
- cache: schema v2 with `clip_feature_mode: penultimate`;
- formal external evaluation: only after all formal configs/checkpoints are
  frozen.

The machine-readable candidate is
`configs/freeze_candidate_clip_penultimate_v2.yaml`. It must not be used for
Part 7 until the user explicitly approves `GO_A1`. After approval, STATUS may
set `Model frozen: yes` and `Next part: Part 07`.
