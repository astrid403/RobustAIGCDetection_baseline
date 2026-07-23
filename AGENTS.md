# Codex Repository Instructions

These rules apply to the entire repository.

## Required startup sequence

Before changing files or running an experiment:

1. Read `docs/codex_plan/MASTER_PLAN.md`.
2. Read `docs/codex_plan/STATUS.md`.
3. Read the `PART_XX.md` file identified by `Next part` in `STATUS.md`.
4. Verify the current Git branch, HEAD, tracked changes, and untracked files.
5. Stop and report if repository state conflicts with the plan. Do not guess.

## Execution discipline

- Execute exactly one Part per user request and never begin the next Part automatically.
- Stay within that Part's allowed files, commands, tests, artifacts, and stopping conditions.
- On completion, update `docs/codex_plan/STATUS.md` and output the Part's standard completion summary.
- At every Go/No-Go gate, stop after recording the evidence and wait for explicit user approval.
- After model freeze, do not add or alter model structure.
- After number freeze, do not retrain, select another checkpoint, or adjust thresholds.
- Do not delete, move, stage, or modify pre-existing untracked user files.
- Do not commit raw data, caches, checkpoints, feature tensors, full predictions, or large logs.
- Do not merge or push unless the user explicitly requests it.
- Keep smoke, pilot, and final run names and output directories strictly separate.

Detailed goals, dependencies, experiment rules, and handoff formats live in
`docs/codex_plan/`; do not duplicate or reinterpret them here.
