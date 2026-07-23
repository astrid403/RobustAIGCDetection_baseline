# Part 12 — Presentation and Final Submission Audit

## Goal and type

Create the 8–10 page presentation and perform the final reproducibility,
repository, report, and submission check. Type: documentation and audit.

## Preconditions

- Part 11 report conclusions and frozen figures/tables are complete.
- Submission format/deadline and presentation format are known.
- STATUS points to Part 12; model and numbers remain frozen.

## Allowed and forbidden scope

Create/update presentation source/export, final checklist, lightweight
submission manifest, and documentation-only fixes. Run tests, link/config
checks, report/presentation builds, and non-training reproduction smoke.

Do not add experiments, retrain, change threshold/model/numbers, create new
claims, include large/private artifacts, merge, or push without instruction.

## Tasks and validation

1. Build 8–10 pages: problem, motivation, baseline limitation, proposed method,
   protocol, main results, ablation/robustness, errors/limitations, conclusion.
2. Reuse Part 10 figures and Part 11 wording; keep units, seeds, scopes, and
   caveats visible.
3. Audit repository status, tracked files, secrets, large-file exclusions,
   licenses/references, paths, configs, and artifact manifest.
4. Run full unit tests and documented lightweight reproduction checks.
5. Cross-check README, report, slides, tables, and STATUS against the number
   freeze and selected route.
6. Produce a submission checklist and list intentionally external artifacts
   with retrieval/reproduction instructions.
7. Update STATUS to Part 12 complete only after every mandatory check passes;
   stop before merge/push/submission and request user direction.

## Outputs and acceptance

- Presentation source and export, final checklist/manifest, and final test log
  summary.
- Slide count 8–10; no inconsistent numbers or unsupported claim; all paths
  resolve; tests pass; Git diff contains only approved lightweight files; no
  data/cache/checkpoint/feature/full prediction/large log is staged.

## Stop, repair, and rollback

Stop for inconsistent frozen numbers, missing required submission format,
failed tests, secret/large artifact risk, or non-reproducible command. Allow
two documentation/build repair rounds. Do not submit, merge, or push without
explicit authorization.

## Git and handoff

Suggested commit:
`docs: finalize presentation and submission checklist`.
No automatic push/merge/submission. Final handoff reports branch/commit, tests,
report/slides, frozen route/numbers, artifact exclusions, known limitations,
and exact user actions remaining.
