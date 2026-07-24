# Frozen S2 GenImage Unseen Diagnostic

The completely frozen three-fold equal-probability ensembles were evaluated
once on GenImage unseen. This diagnostic cannot trigger tuning. B2-v3 and S2
used seeds 42/43/44 and clean, JPEG 70, resize 0.5, and blur 1.0.

Mean AUROC across seeds:

| model | clean | JPEG | resize | blur |
|---|---:|---:|---:|---:|
| B2-v3 | 0.981996 | 0.962028 | 0.961249 | 0.958466 |
| S2 | 0.991108 | 0.986048 | 0.980622 | 0.977485 |

Clean S2-minus-B2 mean AUROC is `+0.009112`. All three S2 clean seed AUROCs
exceed their B2 counterparts. These results are read-only evidence and did
not alter the model, ensemble weights, threshold, or evaluation plan.

The evaluation and tee processes exited zero. All 24 prediction files contain
4,000 unique sample IDs; checkpoint, manifest, prediction, and metric hashes
passed, and selected metrics were independently recomputed at tolerance
`1e-12`. Defactify was not accessed.
