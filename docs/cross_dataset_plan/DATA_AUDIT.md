# Research v3 GenImage Development Data Audit

## Scope

This audit used only:

- `outputs/splits/genimage_splitB_available_train.csv` (3,000 rows);
- `outputs/splits/genimage_splitB_available_val.csv` (1,200 rows).

It did not read GenImage unseen or any Defactify manifest, image, feature,
prediction, or metric. The audit tool rejects manifest paths containing
`unseen` or `defactify`.

## Input integrity

| Role | Rows | SHA256 |
|---|---:|---|
| train | 3,000 | `2b2e1112f044f079a2ef4fa9255d42e850359861c934e8c3bf938527701b5bd8` |
| validation | 1,200 | `5c73cbc97dc41e8ef63b2021fb1758201b4b484f04d4ebb1559f3535c4e44698` |

Each input contains ADM, BigGAN, and Stable Diffusion V1.5. Each generator is
class-balanced: train has 500 real and 500 fake rows; validation has 200 real
and 200 fake rows. All 4,200 rows are recorded as JPEG. Image dimensions vary
by generator and label, so width and height must be treated as potential
shortcuts and retained in fold-level reporting.

## Identity and duplicate policy

- `sample_id` is a deterministic SHA256-derived ID from the manifest path.
- For real images, `source_group` is the lower-cased basename because that is
  the available ImageNet-style source identifier under generator directories.
- For fake images, `source_group` includes generator and basename.
- Exact file SHA256 detects byte-identical copies.
- 64-bit difference hash (dHash) with Hamming distance at most 4 defines the
  conservative near-duplicate exclusion used between a fold's train and
  validation partitions.
- Near-duplicate comparisons are label-aware; source-group and exact-content
  exclusions remain unconditional.

Across the original train and validation manifests:

- sample-ID overlap: 0;
- exact-content SHA256 overlap: 0;
- source-group overlap: 0.

One same-label perceptual near-duplicate was found between the proposed
holdout-ADM validation set and its candidate training set. The training row
`genimage-v3-8881bdfbcb63faa4cfb330aa`
(`BigGAN/train/ai/810_biggan_00128.jpg`) was excluded. No source or exact-file
duplicates required removal.

## Proposed source-aware LOGO folds

| Held-out generator | Train rows before | Train rows after | Validation rows | Near-duplicate removals |
|---|---:|---:|---:|---:|
| ADM | 2,000 | 1,999 | 400 | 1 |
| BigGAN | 2,000 | 2,000 | 400 | 0 |
| Stable Diffusion V1.5 | 2,000 | 2,000 | 400 | 0 |

Every validation fold has 200 real and 200 fake rows from exactly the held-out
generator. Every training fold contains the other two generators and both
labels. The holdout-ADM training fold has 1,000 real and 999 fake rows after
the required near-duplicate exclusion; this negligible imbalance must not be
silently repaired by dropping another sample. A class/generator-aware sampler
may address training batches later if Protocol v3 pre-registers it.

For every proposed fold:

- train/validation sample IDs are disjoint;
- source groups are disjoint;
- exact content hashes are disjoint;
- both labels remain present;
- the validation generator is absent from training.

## Lightweight artifacts

The authoritative machine-readable audit is
`outputs/research_v3/audits/audit_report.json`. It records every proposed
manifest's row count and SHA256. The six proposed manifests and
`logo_distribution.csv` are deterministic: a second run in an isolated
temporary directory produced byte-identical CSVs.

## Decision and limitations

Decision: **GO for Protocol v3 design**. The proposed three-fold LOGO protocol
is implementable without the leakage conditions checked here, and every fold
retains adequate samples and both classes.

Known boundaries:

- basename source grouping is the strongest source identity available in the
  current manifests; it cannot prove that differently named images lack a
  common upstream original;
- dHash is a compact perceptual screen, not a complete semantic
  near-duplicate detector;
- all inputs are JPEG and dimensions differ by generator, so format/size
  shortcuts remain an explicit analysis requirement;
- the proposed manifests are not authorized for training until Protocol v3
  and the S1 contract pass Task 03 approval.
