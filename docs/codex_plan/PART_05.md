# Part 05 — Dual-level Fusion Implementation and Seed-42 Pilot

## Goal and type

Implement the approved lightweight final+penultimate CLIP fusion and obtain one
seed-42 GenImage pilot. Type: focused code change plus pilot experiment.

## Preconditions

- Parts 3–4 accepted with frozen feature definitions.
- Valid seed-42 B1/B2 and penultimate-only evidence is available.
- No Defactify result has influenced candidate design.

## Allowed and forbidden scope

Modify only the CLIP feature model/factory, necessary extraction/cache routing,
training/evaluation entry points, fusion config, and tests. Run unit tests,
tiny smoke, one seed-42 fusion pilot, and one GenImage unseen evaluation.

The fusion must remain lightweight: combine the already approved final and
penultimate features with a small classifier and no new backbone/data. Do not
add frequency/residual branches, attention stacks, extra pretrained models,
multiple fusion variants, Defactify tuning, or formal multi-seed runs.

## Tasks and validation

1. Implement the already selected dual-level fusion with explicit input
   dimensions and deterministic feature pairing.
2. Ensure cache keys include both modes and extraction versions and that sample
   IDs/order match exactly before fusion.
3. Add isolated seed-42 pilot config and unit tests for shape, pairing,
   mismatch rejection, checkpoint round trip, frozen encoder, and B1/B2/
   penultimate compatibility.
4. Run smoke; then train once using validation-only checkpoint selection.
5. Evaluate once on GenImage unseen and record complete metrics, params,
   timings, config/commit/command, and artifact hashes.

## Outputs and acceptance

- Focused implementation/tests/config and lightweight pilot registry/metrics.
- Tests and commands exit 0; feature rows pair one-to-one; no cache collision;
  smoke/pilot/final isolation holds; no Defactify use or output overwrite.
- A valid poor pilot is still a completed Part and must not be hidden.

## Stop, repair, and rollback

Stop for sample-order mismatch, silent truncation, NaN, OOM exceeding the
approved lightweight budget, cache contamination, or overwrite. Permit one
correctness repair/rerun. Poor metrics are not grounds for redesign or repeated
tuning. Route choice is deferred to Part 6.

## Git and handoff

Suggested commit:
`feat: add dual-level CLIP fusion pilot`.
No push/merge. Handoff includes architecture summary, parameter count, config,
seed, validation/unseen metrics, artifact hashes, runtime, failures, and a
comparison-ready summary for Part 6.
