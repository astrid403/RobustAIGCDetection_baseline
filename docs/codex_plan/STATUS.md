# Codex Execution Status

- Current branch: `feature/defactify-external-eval`
- Last completed part: `Part 04`
- Last verified commit: `189eee2791769c6b32a99e5971ac8823447d8ac5`
- Next part: `Part 05` (`docs/codex_plan/PART_05.md`)
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

### Part 03 — Verify CLIP Feature Contract and Cache Safety

- Status: completed and accepted.
- Commit: `7f1fbe4d5a4e9c2ba9aa49d85dc0a09839d480a7`.
- Tests: 24/24 unit tests passed; offline real ViT-B-32 one-image smoke
  verified both feature modes.
- Outputs: `docs/codex_plan/CLIP_FEATURE_CONTRACT.md`, explicit hook-free
  final/penultimate extraction, cache schema v2 fingerprints, legacy-cache
  rejection, and compatibility/safety tests.
- Handoff: `final` is the normalized normal 12-block CLIP output;
  `penultimate` skips only block 12 and then uses the same pooling, post-norm,
  projection, and L2 normalization. Both have shape `[batch, 512]` for
  ViT-B-32. Repeated extraction had maximum absolute difference `0.0`.
  Existing legacy caches must be regenerated because they lack feature-mode
  and extraction-version metadata. No classifier, pilot training, formal
  evaluation, or Defactify access was performed.

### Part 04 — Penultimate-only Implementation and Seed-42 Pilot

- Status: completed and accepted.
- Implementation commit:
  `a2759ea148ba74057dc80c5bca55c743176321d3`.
- Result commit: `189eee2791769c6b32a99e5971ac8823447d8ac5`.
- Tests: 27/27 unit tests passed; real ViT-B-32 penultimate feature/head smoke
  and post-run artifact verification passed.
- Outputs: isolated seed-42 penultimate-only config, complete validation log,
  complete GenImage-unseen metric CSV, and lightweight provenance registry.
- Validation: best checkpoint selected at epoch 10 by validation AUROC
  `0.9969138889`; validation AUPRC `0.9967979636`, balanced accuracy
  `0.9733333333`, and macro-F1 `0.9733330370`.
- GenImage unseen: AUROC `0.9880475000`, AUPRC `0.9872634407`, balanced
  accuracy `0.93625`, macro-F1 `0.9361706881`, real recall `0.9715`, and fake
  recall `0.901`.
- Handoff: train/validation/unseen cache shapes are respectively
  `[3000,512]`, `[1200,512]`, and `[4000,512]`; all use schema v2
  penultimate metadata and newly generated cache signatures. The MLP has
  263,169 trainable parameters. Training and unseen evaluation wall times were
  54.18 s and 57.72 s. Exactly one pilot training run and one unseen
  evaluation were performed. No Defactify access or fusion implementation
  occurred. Large caches, checkpoints, predictions, logs, and figures remain
  outside Git.

## Current boundary

The penultimate-only seed-42 pilot is complete, but no dual-level fusion has
been implemented or run. Part 05 must not start unless explicitly requested by
the user.

## Status update template

After one Part completes, update the fields at the top, append a completion
entry with commit/tests/artifacts/decision/known issues, set the next Part, and
stop. At a Go/No-Go gate, set `Next part` to `Awaiting user approval` until the
decision is explicitly approved.
