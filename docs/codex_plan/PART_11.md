# Part 11 — Reproducibility Package and Final Report

## Goal and type

Clean the reproducibility-facing repository materials and write the formal
course report using only frozen methods and numbers. Type: documentation.

## Preconditions

- `Numbers frozen: yes` in STATUS and Part 10 manifest is valid.
- Course report template and submission constraints are available.
- All commands/configs/artifact hashes needed for reproduction are known.

## Allowed and forbidden scope

Update README, environment/config/run instructions, lightweight artifact
manifest, result references, report source, references, and documentation.
Run documentation link/path checks, config parsing, unit tests, and a report
build if local dependencies already exist.

Do not change algorithm/data/evaluation logic; retrain; adjust threshold or
numbers; download dependencies; commit large artifacts; or overstate claims.

## Tasks and validation

1. Reconcile README and commands with the frozen branch/config/run names.
2. Document environment, data preparation, split roles, training, evaluation,
   robustness, expected outputs, and unavailable large-artifact reproduction.
3. Update artifact manifest with config/commit/checkpoint/result hashes and
   clear exclusion policy.
4. Write report sections: abstract, introduction, related work, problem,
   method, protocol, main results, ablation, robustness/generalization,
   qualitative analysis, limitations, conclusion, references.
5. Insert only Part 10 figures/tables and use the approved claim route.
6. Check citations, table numbers, sample counts, seeds, metric directions,
   figure paths, commands, and report compilation.
7. Run the full lightweight test suite and final README reproduction smoke
   that does not train or download.

## Outputs and acceptance

- Clean README/config instructions/artifact manifest and complete report source
  (plus built PDF if supported).
- All internal paths/links resolve; configs parse; tests pass; report numbers
  exactly match freeze; limitations include dataset scale, near-duplicate audit
  boundary, external generalization, and any negative result.

## Stop, repair, and rollback

Stop if the course template is missing, a frozen number conflicts, a command
cannot be reproduced, or writing requires a changed experiment. Allow two
documentation/build repair rounds. Never silently alter a frozen result; ask
the user whether to reopen Part 10.

## Git and handoff

Suggested commit:
`docs: finalize reproducibility package and report`.
No push/merge. Handoff includes report path/build status, README commands,
manifest, tests, unresolved submission-format issues, main conclusions, and
the exact content Part 12 may present.
