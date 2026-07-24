# Cross-Dataset Research Execution Rules

## Standing authorization

On 2026-07-24 the user granted standing execution authority on
`research/cross-dataset-robustness-v3`. Within the active S2 route, engineering
correctness fixes, config/cache/registry/provenance/AMP/training/analysis
repairs, correctness tests, tiny smokes, new-ID reruns, commits, and automatic
progression after a Go Gate are authorized. Failed and invalid runs must remain
preserved.

Work must stop before the first Defactify access; before changing model
structure, blocks, loss, data protocol, split, budget, threshold, or Gate; on
an S2 Development No-Go before selecting S3; before push/merge/main/force/
reset/delete/overwrite actions; or on leakage, frozen-asset drift, or an
unreliably repairable scientific-correctness issue. GenImage unseen is allowed
only after model freeze and cannot drive tuning.

## S2 CUDA AMP contract supplement

The original frozen contract required CUDA mixed precision but did not define
its execution details. This supplement resolves only that engineering
ambiguity and does not change the model, objective, data, budget, checkpoint
metric, threshold, or Gate.

- CUDA uses `torch.autocast(device_type="cuda", dtype=torch.float16)` around
  the RINE-lite forward and complete BCE-plus-SupCon objective.
- Training uses an enabled CUDA `GradScaler`; validation and inference use the
  same float16 autocast without scaling.
- Checkpoints contain scaler state so a training resume preserves AMP state.
- CPU correctness runs disable AMP. CUDA correctness evidence must verify that
  autocast and the scaler are actually enabled.
- Config, epoch log, checkpoint, and registry record requested and effective
  AMP state.

The selection follows standard CUDA float16 AMP practice and was made without
consulting the metrics of the invalid FP32 runs.
