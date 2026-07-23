# Codex Execution Status

- Current branch: `feature/defactify-external-eval`
- Last completed part: `Part 02`
- Last verified commit: `fdc149acb1e203a66bb30d3ffc21d39be36dd12a`
- Next part: `Part 03` (`docs/codex_plan/PART_03.md`)
- Selected route: `Main route planned: final + penultimate CLIP dual-level fusion; fallback: penultimate-only; Go/No-Go not yet reached`
- Model frozen: `no`
- Numbers frozen: `no`
- Blocking issues: `none`

## Completed parts

### Part 01 — Freeze Baselines and Protocol v2

- Status: completed and accepted.
- Commit: `7f4aac29fdd69d713eea282aa1d521144cdcba60`.
- Outputs: `docs/FINAL_PROTOCOL.md` and
  `artifacts/baseline_v2_manifest.json`.
- Verification: branch/HEAD/worktree and baseline assets audited; required
  split/checkpoint/metric/config SHA256 values recorded; Protocol v2 dataset
  roles and smoke/pilot/final naming frozen.
- Handoff: B0 ResNet50 and B2 CLIP-MLP are historical frozen baselines;
  Defactify is external evaluation only; pre-existing untracked files remain
  untouched.

### Part 02 — Complete Metrics, Tests, and CLIP Linear Probe

- Status: completed and accepted.
- Commit: `fdc149acb1e203a66bb30d3ffc21d39be36dd12a`.
- Tests: 18/18 unit tests passed; B1/B2 tiny forward and configurable-threshold
  smoke passed.
- Outputs: complete detection metrics, explicit single-class behavior,
  backward-compatible `recall`/`f1` aliases, B1 linear-probe model/config, and
  B1/B2 compatibility tests.
- Handoff: threshold defaults to 0.5; `0=real`, `1=fake`; single-class balanced
  accuracy/AUROC/AUPRC return NaN; no formal training or Defactify evaluation
  was run.

## Current boundary

The project has not entered penultimate feature implementation or algorithm
pilot work. Part 03 is a feature-contract and cache-safety Part only. It must
not start unless explicitly requested by the user.

## Status update template

After one Part completes, update the fields at the top, append a completion
entry with commit/tests/artifacts/decision/known issues, set the next Part, and
stop. At a Go/No-Go gate, set `Next part` to `Awaiting user approval` until the
decision is explicitly approved.
