# Part 10 — Statistics, Final Figures, and Number Freeze

## Goal and type

Aggregate seed-level evidence, create final tables/figures, audit provenance,
and freeze all reportable numbers. Type: result analysis and decision gate.

## Preconditions

- Parts 7–9 have complete seed-level metrics, robustness, efficiency, and
  qualitative artifacts.
- Model remains frozen and no run is pending or invalid.

## Allowed and forbidden scope

Modify/add only statistics, table/plot generation, lightweight final summaries,
figures, manifests, and tests. Compute mean, sample standard deviation, paired
model deltas, and paired bootstrap 95% confidence intervals only if time
permits and sample pairing is valid.

Do not retrain, re-evaluate to seek better results, change thresholds/checkpoints,
exclude bad seeds, invent missing data, or redesign figures to obscure results.

## Tasks and validation

1. Validate the complete model×seed×dataset matrix and artifact hashes.
2. Aggregate each required metric as per-seed values, mean, and sample SD.
3. Compute paired final-minus-B2 deltas. If time permits, bootstrap paired
   predictions with a fixed seed and document method; otherwise mark CI omitted.
4. Select the honest contribution route defined in MASTER_PLAN.
5. Generate exactly the compact report set: method diagram, main results table,
   core ablation table, one generalization/robustness figure or table, and one
   qualitative error figure. Extra ROC/PR/confusion plots are supplemental.
6. Cross-check every displayed number against machine-readable summaries.
7. Write a number-freeze manifest with source files/hashes, commit, timestamp,
   selected route, and limitations; update STATUS to `Numbers frozen: yes`.
8. Stop for user acknowledgment before any correction that would reopen data.

## Outputs and acceptance

- Machine-readable aggregate results and five controlled report artifacts.
- Means/SDs recompute from all three seeds; deltas have correct direction;
  tables and captions state full versus balanced scope; no unsupported claim.
- Number-freeze manifest is complete and `git diff --check` passes.

## Stop, repair, and rollback

Stop for missing seed, provenance mismatch, impossible metric, or disagreement
between source and table. Allow two analysis-only repair rounds. Never repair a
number by retraining or threshold adjustment. Reopening frozen numbers requires
explicit user approval and an audit note.

## Git and handoff

Suggested commit:
`results: freeze final statistics and figures`.
No push/merge. Handoff includes frozen tables/figures, aggregate metrics,
optional-CI status, contribution wording, freeze manifest/commit, limitations,
and confirmation that Part 11 may write from these numbers only.
