# S1 Seed-42 LOGO Development Decision

## Decision

**No-Go for S1.** Task 08 completed on the frozen Protocol-v3 development
data only. Tasks 09–13 are skipped by this gate. No GenImage unseen or
Defactify data was accessed, and no automatic switch to S2 is authorized.

## Frozen comparison

All values use three seed-42 held-out-generator folds and fixed threshold
0.5. The primary S1 result is fixed 0.5 B2 / 0.5 NPR probability fusion.

| Condition | Expert | Fold-mean AUROC | Worst-fold AUROC | Overall AUROC | AUPRC | Balanced accuracy | Real recall | Fake recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | B2-v3 | 0.966050 | 0.933925 | 0.904872 | 0.918140 | 0.809167 | 0.963333 | 0.655000 |
| clean | NPR | 0.950383 | 0.897850 | 0.952786 | 0.952556 | 0.829167 | 0.965000 | 0.693333 |
| clean | fixed fusion | 0.962717 | 0.918525 | 0.963125 | 0.963495 | 0.810000 | 0.988333 | 0.631667 |
| JPEG | B2-v3 | 0.937492 | 0.903825 | 0.900803 | 0.906001 | 0.830833 | 0.881667 | 0.780000 |
| JPEG | NPR | 0.931529 | 0.822313 | 0.950082 | 0.944108 | 0.870000 | 0.883333 | 0.856667 |
| JPEG | fixed fusion | 0.955383 | 0.896275 | 0.960258 | 0.962863 | 0.905833 | 0.951667 | 0.860000 |
| resize | B2-v3 | 0.919575 | 0.876225 | 0.853547 | 0.866255 | 0.780833 | 0.898333 | 0.663333 |
| resize | NPR | 0.949217 | 0.867950 | 0.960675 | 0.948196 | 0.900000 | 0.918333 | 0.881667 |
| resize | fixed fusion | 0.958508 | 0.902700 | 0.959344 | 0.952531 | 0.877500 | 0.955000 | 0.800000 |
| blur | B2-v3 | 0.921100 | 0.874000 | 0.869564 | 0.879719 | 0.793333 | 0.880000 | 0.706667 |
| blur | NPR | 0.941733 | 0.857375 | 0.945044 | 0.922664 | 0.886667 | 0.871667 | 0.901667 |
| blur | fixed fusion | 0.956167 | 0.904175 | 0.961061 | 0.951302 | 0.889167 | 0.943333 | 0.835000 |

Clean fold AUROC deltas for fusion minus B2-v3 were `+0.002475` on ADM,
`+0.002925` on BigGAN, and `-0.015400` on Stable Diffusion V1.5.

## Development Gate

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Clean fold-mean AUROC delta vs B2-v3 | >= +0.005 | -0.003333 | Fail |
| Clean worst-fold AUROC delta vs B2-v3 | >= -0.002 | -0.015400 | Fail |
| Improved clean folds | >= 2/3 | 2/3 | Pass |
| Mean degraded fold-mean AUROC delta | >= 0 | +0.030631 | Pass |
| Fusion over stronger clean single expert | >= +0.002 | -0.003333 | Fail |
| Clean overall real recall | >= 0.70 | 0.988333 | Pass |
| Clean overall fake recall | >= 0.70 | 0.631667 | Fail |

The clean error analysis contains 229 B2 errors, 205 NPR errors, 102 shared
errors, 127 B2-wrong/NPR-right cases, 103 NPR-wrong/B2-right cases, and 230
prediction disagreements among 1,200 OOF samples. This confirms measurable
complementarity, but it does not satisfy the preregistered fusion-improvement
or recall requirements.

## Integrity and failure record

All six successful training registries, validation-AUROC-selected best
checkpoints, four prediction conditions, ordered sample IDs, folds, labels,
and prediction hashes were verified before aggregation. Each OOF condition
contains 1,200 unique samples, exactly 400 from each held-out fold.

The first OOF summary attempt was retained at
`outputs/research_v3/task08_seed42_v3`. It produced clean fusion artifacts
before failing because fusion-only columns were incorrectly required for a
single-expert summary. The corrected analysis used the new non-overwriting
directory `outputs/research_v3/task08_seed42_v3_retry1`; none of the six
pilots was rerun.
