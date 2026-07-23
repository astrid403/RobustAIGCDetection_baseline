# Part 09 — Compact Robustness, Efficiency, and Error Cases

## Goal and type

Add one compact, protocol-relevant analysis package for the frozen B2 and final
models: robustness, efficiency, and qualitative errors. Type: formal analysis.

## Preconditions

- Part 8 formal clean predictions and metrics are complete.
- Same model/seed set and frozen threshold remain authoritative.
- Existing JPEG/resize/blur implementation and available datasets are audited.

## Allowed and forbidden scope

Run the pre-approved compact JPEG/resize/Gaussian-blur settings on the selected
formal evaluation scope, using all required seeds where feasible and identical
settings for both models. Measure total/trainable parameters, training time,
inference time, peak memory if readily available, and cache size/cost. Generate
a small fixed error-case set from frozen predictions.

Do not add corruptions, tune severity, retrain, change threshold, select only a
favorable seed/generator, or copy error images into Git.

## Tasks and validation

1. Freeze a minimal severity list already supported by the repository.
2. Run identical robustness evaluation for B2/final; record clean and degraded
   metrics and delta.
3. Aggregate per-generator/worst-generator evidence from Part 8.
4. Measure params/trainable params, training time from logs, comparable
   inference latency protocol, and feature-cache storage/extraction cost.
5. Select deterministic false-positive, false-negative, and where available
   B2-wrong/final-correct examples; create a lightweight manifest and figure
   referencing source paths without committing image copies.
6. Audit units, hardware, warm-up/repeats, seeds, and provenance.

## Outputs and acceptance

- Robustness table, efficiency table, error-case manifest/figure, and metadata.
- Identical conditions for both models; no output overwrite; all rows trace to
  formal commit/config/checkpoint; no cherry-picking.
- Large logs/caches/images remain uncommitted.

## Stop, repair, and rollback

Stop for unequal transforms, threshold drift, timing protocol mismatch,
missing provenance, or accidental data copying. Allow one analysis-code repair.
If GPU time is insufficient, reduce only optional severity points uniformly
and document it; never drop an unfavorable model/seed selectively.

## Git and handoff

Suggested commit:
`analysis: add robustness efficiency and error cases`.
No push/merge. Handoff includes robustness deltas, worst-generator results,
efficiency units, error-case manifest, limitations, and every file Part 10
must aggregate.
