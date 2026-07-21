# Defactify External Evaluation

Defactify is the current external cross-dataset benchmark. WildFake code and results are retained unchanged as Milestone 2 legacy material.

## Data contract

- Dataset: `Rajarshi-Roy-research/Defactify_Image_Dataset`
- Verified revision: `787334f7857fa54f29027a7f09c30e895ad486ef`; `main` is rejected.
- Official splits: `train`, `validation`, `test`.
- `Label_A`: 0 real, 1 AI-generated.
- `Label_B`: 0 real, 1 SD 2.1, 2 SDXL, 3 SD3, 4 DALL-E 3, 5 Midjourney 6.

The manifest stores `sample_id`, backend, dataset ID/revision, official split and row index. It never stores image bytes or cache-internal absolute paths. Images are decoded directly from the Hugging Face cache.

## Preparation

Use the project interpreter and keep the Hugging Face cache outside the repository:

```bash
export HF_HOME=/home/rong/.cache/huggingface
export DEFACTIFY_REVISION=787334f7857fa54f29027a7f09c30e895ad486ef
bash scripts/prepare_defactify.sh
```

Generated inventory and split CSVs are ignored by Git. The preparation creates official split manifests, a balanced test manifest, and a 36-row smoke manifest containing two records per Label_B per official split.

## Evaluation

```bash
bash scripts/run_defactify_smoke_test.sh
bash scripts/run_defactify_external.sh
```

Both commands reuse the completed GenImage available-split checkpoints. They do not train on Defactify. The external script runs each model once on the official full test, then derives balanced and per-generator metrics by filtering the saved full predictions by deterministic sample ID; it does not perform a second clean balanced inference.

CLIP feature cache names include a hash of the complete manifest, dataset revision, CLIP model and preprocessing signature. Cache metadata is validated before reuse.

## Safety

- Do not commit Hugging Face blobs, Arrow/Parquet files, inventories, splits, checkpoints, features or predictions.
- Do not use an unpinned `main` revision.
- Do not overwrite or rename WildFake Milestone 2 outputs.
- Run the smoke evaluation before the 45,000-row official test.
