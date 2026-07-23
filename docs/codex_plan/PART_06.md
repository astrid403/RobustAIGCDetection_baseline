# Part 06 — Fair Ablation, Go/No-Go, and Model Freeze

## Goal and type

Produce a controlled seed-42 comparison of B1, B2, penultimate-only, fusion,
and one necessary parameter-fair control; then recommend a route and stop for
user approval. Type: focused experiment/analysis and decision gate.

## Preconditions

- Valid Part 4 and Part 5 pilots with provenance.
- Same GenImage splits, metric implementation, threshold policy, feature
  definitions, and checkpoint rule are available for every row.
- No candidate has been changed based on Defactify.

## Allowed and forbidden scope

Add only a necessary parameter-matched head/config or comparison/statistics
script and tests. Reuse valid seed-42 pilots; run only missing/invalid fair
controls on GenImage. Analyze validation and the single approved unseen result.

Do not access Defactify, create new algorithm families, tune many widths,
increase data/epochs selectively, run seeds 43/44, or freeze a route without
explicit user approval.

## Tasks and validation

1. Audit comparability: train/validation manifests, seed, epochs/early-stop
   rule, optimizer policy, threshold, feature normalization, and metric code.
2. Add/run at most one parameter-fair control that distinguishes feature
   benefit from head capacity; record exact trainable parameters.
3. Build a table for B1, B2, penultimate-only, fusion, and fair control with
   validation and GenImage-unseen AUROC, AUPRC, balanced accuracy, macro-F1,
   real/fake recall, params, and runtime.
4. Check for pathological class-collapse and metric trade-offs.
5. Apply the MASTER_PLAN gate: recommend fusion, fallback penultimate-only, or
   negative-result B2. Record evidence and rejected alternatives.
6. Draft the exact final config contract: route, feature modes, head, loss,
   optimizer, epochs, checkpoint rule, threshold, seeds 42/43/44.
7. Update STATUS to `Awaiting Go/No-Go approval` and stop.

## Outputs and acceptance

- Fair comparison table and decision memo with traceable inputs.
- Any new code has tests; all commands exit 0; no Defactify access.
- Every row is comparable or explicitly marked non-comparable and excluded.
- Recommendation and final config are unambiguous.
- User approval is required before `Model frozen` becomes `yes`.

## Stop, repair, and rollback

Stop for unfair budgets, missing provenance, invalid pilots, leakage, or
ambiguous candidate correctness. Allow one repair of analysis/control code;
do not manufacture extra variants. No automatic fallback or continuation.

## Git and handoff

Suggested commit:
`research: compare CLIP feature variants for model freeze`.
No push/merge. Handoff includes table, provenance, recommended route, exact
freeze contract, known issues, and the explicit question requesting Go/No-Go
approval. After approval, record selected route and model freeze in STATUS;
only then is Part 7 ready.
