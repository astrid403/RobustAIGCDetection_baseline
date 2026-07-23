# Codex Execution Status

- Current branch: `feature/defactify-external-eval`
- Last completed part: `Part 06`
- Last verified commit: `435d9f08d3b519a014bbf69c37ab19deeda9aa96`
- Next part: `Part 07` (`docs/codex_plan/PART_07.md`)
- Selected route: `A1`
- Model frozen: `yes`
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

### Part 05 — Dual-level Fusion Implementation and Seed-42 Pilot

- Status: completed and accepted.
- Implementation commit:
  `3fdd1ab39a9699c25634a0b555fec6c8a8b3e280`.
- Result commit: `1098030f3301586ef17a07777a6069fe9e204b9b`.
- Tests: 32/32 unit tests passed; real final+penultimate extraction/pairing/
  fusion smoke and post-run provenance verification passed.
- Architecture: independently L2-normalize the frozen 512-D final and 512-D
  penultimate embeddings, use identity projections because dimensions match,
  concatenate in `[final, penultimate]` order, then apply
  `Linear(1024,256)-ReLU-Dropout(0.2)-Linear(256,1)`. No gating, attention,
  extra loss, patch token, frequency branch, or new backbone was added.
- Validation: best checkpoint selected at epoch 8 by validation AUROC
  `0.9971833333`; validation AUPRC `0.9971058604`, balanced accuracy
  `0.9741666667`, and macro-F1 `0.9741665052`.
- GenImage unseen: AUROC `0.9868732500`, AUPRC `0.9852834342`, balanced
  accuracy `0.9355`, macro-F1 `0.9354042530`, real recall `0.974`, and fake
  recall `0.897`.
- Fair seed-42 comparison: fusion AUROC is `+0.003021` versus frozen B2 and
  `-0.00117425` versus penultimate-only. Fusion uses 262,657 trainable
  parameters versus 263,169 for each comparator. Route selection remains
  deferred to Part 06.
- Handoff: six newly generated schema-v2 caches cover
  train/validation/unseen x final/penultimate and total 34,814,110 bytes.
  Training and unseen evaluation wall times were 87.86 s and 96.65 s.
  Exactly one training run and one unseen evaluation were performed. No
  Defactify access occurred. Large caches, checkpoints, predictions, logs, and
  figures remain outside Git.

### Part 06 — Fair Ablation and Go/No-Go Recommendation

- Status: completed; decision gate awaiting explicit user approval.
- B1 control config commit:
  `636a4a1adc673a7dee7e05e12846db0c64a99956`.
- Decision/evidence commit:
  `435d9f08d3b519a014bbf69c37ab19deeda9aa96`.
- Tests: 32/32 unit tests passed before the control run; B1 tensor/config
  smoke, v2 cache/checkpoint/metric provenance verification, and eight-row
  comparison-table recomputation passed.
- B1 control: validation AUROC `0.9635916667`; unseen AUROC `0.9357331250`,
  AUPRC `0.9346916379`, balanced accuracy `0.8495`, and macro-F1
  `0.8490180278`. It has 513 trainable parameters.
- Decision: `GO_A1` recommended. A1 has the best GenImage-unseen AUROC
  (`0.9880475`), AUPRC (`0.9872634`), balanced accuracy (`0.93625`),
  macro-F1 (`0.9361707`), and fake recall (`0.901`) among B1/B2/A1/P.
- A2: not run because frozen final-only B2 is already parameter matched:
  B2/A1 have 263,169 parameters and P has 262,657 (0.19% difference).
- Alternatives: P is rejected for freeze because its slight validation
  advantage does not persist on unseen and it doubles feature-cache cost.
  Degradation augmentation is not selected because A1 already supplies the
  approved improvement signal. No fourth algorithm or new module was added.
- Proposed freeze contract: frozen OpenAI ViT-B-32 penultimate 512-D feature,
  512-hidden MLP, dropout 0.2, AdamW 0.001, BCEWithLogitsLoss, batch 32,
  10 epochs, validation-AUROC checkpoint selection, fixed threshold 0.5,
  cache schema v2, and formal seeds 42/43/44.
- Handoff: decision record is `docs/codex_plan/PART_06_DECISION.md` and the
  machine-readable pending config is
  `configs/freeze_candidate_clip_penultimate_v2.yaml`. No Defactify access,
  seed-43/44 run, A2 run, degradation pilot, merge, or push occurred.
  The user explicitly approved `GO_A1`; the contract is now frozen.

## Current boundary

The user approved `GO_A1`; A1 and its complete training contract are frozen.
Part 07 may run the six formal B2/A1 seed jobs. No structure, feature mode,
head, budget, checkpoint rule, threshold, or seed change is permitted.

## Status update template

After one Part completes, update the fields at the top, append a completion
entry with commit/tests/artifacts/decision/known issues, set the next Part, and
stop. At a Go/No-Go gate, set `Next part` to `Awaiting user approval` until the
decision is explicitly approved.
