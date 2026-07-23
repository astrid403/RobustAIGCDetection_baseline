# Codex Execution Status

- Current branch: `feature/defactify-external-eval`
- Last completed part: `Part 10`
- Last verified commit: `061f9c813f65b0c73aeb13ff47bbc50a48f06a15`
- Next part: `Part 11` (`docs/codex_plan/PART_11.md`)
- Selected route: `A1`
- Model frozen: `yes`
- Numbers frozen: `yes`
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

### Part 07 — Frozen B2/A1 Three-seed Formal Training

- Status: completed and accepted.
- Freeze/config commit:
  `0ae35a061b7ffc7a901500dde3dfbe2bfeacee57`.
- Formal-run registry commit:
  `87450b386663157d459ce4c56f614b48ab72be4a`.
- Tests: 32/32 unit tests passed after all runs; the six config snapshots,
  best checkpoints, ten-epoch logs, cache schema-v2 metadata, metrics, timing,
  and SHA256 records passed post-run provenance verification.
- Runs: B2 final-feature MLP and frozen A1 penultimate-feature MLP each
  completed seeds 42/43/44. All six official jobs exited successfully without
  retries. Pilot seed 42 was not reused because its experiment stage/name was
  not the frozen formal contract.
- B2 validation mean +/- sample standard deviation: AUROC
  `0.996549 +/- 0.000047`, AUPRC `0.996457 +/- 0.000046`, balanced accuracy
  `0.972500 +/- 0.001443`, and macro-F1 `0.972499 +/- 0.001444`.
- A1 validation mean +/- sample standard deviation: AUROC
  `0.996939 +/- 0.000082`, AUPRC `0.996824 +/- 0.000091`, balanced accuracy
  `0.974167 +/- 0.000833`, and macro-F1
  `0.974166 +/- 0.000833`.
- Best checkpoints for Part 08:
  `outputs/checkpoints/final_clip_mlp_genimage_validation_seed{42,43,44}_v2/best_model.pt`
  and
  `outputs/checkpoints/final_clip_penultimate_genimage_validation_seed{42,43,44}_v2/best_model.pt`.
- Outputs: `artifacts/part07_formal_training_registry.json` contains config,
  split, checkpoint, log, cache, timing, and metric provenance;
  `outputs/metrics/part07_formal_training_validation.csv` contains the six
  per-seed validation rows.
- Handoff: training commit is
  `0ae35a061b7ffc7a901500dde3dfbe2bfeacee57`; the contract SHA256 is
  `5e7d544cb77ca33847e07cdcf55a0bf183eae74422eff70cc0d1a37bb623a1b9`.
  No Defactify access or GenImage-unseen evaluation occurred. Checkpoints,
  caches, full logs, and figures remain outside Git. No failed run was deleted
  or hidden.

### Part 08 — Formal GenImage and Defactify Evaluation

- Status: completed and accepted.
- Evaluation-config commit:
  `22f0f24bd1d596cef678c285fa909bca15c3e09a`.
- Formal-result registry commit:
  `b6d7e79f31162f0ed6f579142c159c7c35bea364`.
- Tests: 32/32 unit tests passed. All twelve full prediction files were
  independently recomputed at threshold 0.5; all metrics and confusion
  matrices matched at tolerance `1e-12`.
- Matrix: B2 and A1 seeds 42/43/44 were each evaluated exactly once on
  GenImage unseen (4,000) and Defactify official full (45,000). Each
  Defactify balanced result (15,000) was selected by sample_id from its
  corresponding full prediction without a second inference.
- GenImage unseen AUROC mean +/- sample standard deviation: B2
  `0.984071 +/- 0.000249`; A1 `0.988194 +/- 0.000129`. Balanced accuracy:
  B2 `0.933250 +/- 0.001639`; A1 `0.937000 +/- 0.003192`.
- Defactify full AUROC: B2 `0.809061 +/- 0.001318`; A1
  `0.796040 +/- 0.001584`. Balanced accuracy: B2
  `0.663524 +/- 0.009654`; A1 `0.662187 +/- 0.011423`.
- Defactify balanced AUROC: B2 `0.808732 +/- 0.001485`; A1
  `0.795021 +/- 0.001616`. Balanced accuracy: B2
  `0.662000 +/- 0.009739`; A1 `0.661622 +/- 0.011898`.
- Per-generator: five fake generators are present for every full and balanced
  model/seed analysis (60 rows total). Mean full worst-generator AUROC is
  `0.772858` for B2 on Stable Diffusion 3 and `0.726247` for A1 on Stable
  Diffusion XL.
- Integrity: 24,000 GenImage unseen rows, 270,000 Defactify full rows, and
  90,000 derived balanced rows were audited. Every prediction file has unique
  sample IDs, every balanced ID set exactly matches the frozen manifest, and
  all predicted labels equal `fake_prob >= 0.5`.
- Failure record: the initial B2 seed-42 Defactify invocation stopped before
  prediction creation because the sandbox could not create a Hugging Face
  cache lock file. The identical frozen command succeeded on one permission
  retry; the failed console log was retained and scientific results were
  unaffected.
- Outputs: `artifacts/part08_formal_evaluation_registry.json`,
  `outputs/metrics/part08_formal_evaluation_summary.csv`, and
  `outputs/metrics/part08_per_generator_summary.csv`. Full/balanced
  predictions, feature caches, and large logs remain outside Git.
- Handoff: Part 08 numbers are complete inputs for analysis but remain
  unfrozen until Part 10. The preliminary result is a domain trade-off: A1
  improves GenImage unseen AUROC by about `+0.004123`, while B2 exceeds A1 on
  Defactify full AUROC by about `+0.013021`. No model selection, retraining,
  threshold adjustment, or data-protocol change was made after observing
  Defactify.

### Part 09 — Compact Robustness, Efficiency, and Error Cases

- Status: completed and accepted.
- Analysis implementation commit:
  `27994e592fce50a5a2d0d8a1215d4a5ef32262a5`.
- Robustness aggregation repair commit:
  `39a579436da0901de296b580bfe42c7b4cc1972d`.
- User-approved registry repair commit:
  `8f05566239f70a9ad079ce45bc5bcdad7aa66e9b`.
- Result commit:
  `59806a4fe4183d55fcbc7b8a73de90c325ba6aeb`.
- Tests: 35/35 unit tests passed. Registry SHA256, robustness matrix,
  bootstrap table, efficiency table, and error manifests passed final
  provenance and row-count checks.
- Paired hierarchical bootstrap: 1,000 repetitions with fixed seed
  `20260723`, resampling both the three training seeds and matched sample IDs.
  On GenImage unseen, A1-minus-B2 is supported positive for AUROC
  (`95% CI [0.002493, 0.006033]`) and AUPRC
  (`[0.002962, 0.008906]`), but not balanced accuracy
  (`[-0.002362, 0.009813]`).
- External statistical result: A1-minus-B2 is supported negative on Defactify
  full for AUROC (`[-0.016487, -0.009541]`) and AUPRC
  (`[-0.004176, -0.001467]`), and on Defactify balanced for AUROC
  (`[-0.017987, -0.009504]`) and AUPRC
  (`[-0.014367, -0.002824]`). Balanced-accuracy intervals cross zero in both
  scopes.
- Robustness: on GenImage unseen, A1 mean AUROC exceeds B2 under clean
  (`+0.004123`), JPEG quality 70 (`+0.009764`), resize scale 0.5
  (`+0.007514`), and Gaussian blur radius 1.0 (`+0.008178`). A1 also has a
  smaller clean-to-corruption AUROC drop for all three corruptions. This is a
  compact in-domain robustness result, not evidence of universal
  cross-dataset superiority.
- Per-generator: A1 improves Defactify full DALL-E 3 AUROC by
  `+0.042666`, but declines on the other four generators. The largest decline
  is Stable Diffusion XL (`-0.054460`); the A1 worst-generator full AUROC is
  `0.726247`, versus B2's worst-generator AUROC `0.772858` on Stable
  Diffusion 3.
- Efficiency: both models have 151,540,482 total parameters including frozen
  CLIP, 263,169 trainable parameters, 3,165,301-byte checkpoints, and
  8,912,778-byte training feature caches per run. Mean training time is about
  62.62 s for B2 and 55.80 s for A1. The compact degraded-feature timing
  averaged about 24.89 images/s for B2 final and 26.26 images/s for A1
  penultimate under the recorded batch-32 protocol.
- Error alignment: every B2/A1 pair was joined by sample_id for all three
  seeds and scopes. GenImage unseen totals include 529 both-wrong,
  272 B2-wrong/A1-correct, and 227 B2-correct/A1-wrong cases. Defactify full
  shows the reverse trade-off: 4,792 B2-wrong/A1-correct versus 5,665
  B2-correct/A1-wrong cases.
- Outputs: `outputs/metrics/part09_analysis_registry.json`, mean/SD and delta
  tables, paired-bootstrap CIs, raw and aggregated robustness tables,
  efficiency table, per-generator delta table, error-count table, deterministic
  180-row error-case manifest, and
  `outputs/figures/part09_error_comparison.png`. No image copies, predictions,
  caches, checkpoints, or large logs were committed.
- Repairs: the first robustness run completed inference but failed at pandas
  column selection; one tested aggregation repair was applied. The first
  statistical run completed tables but failed registry serialization on a
  Python boolean typo; the user explicitly approved the tested registry-only
  repair. Neither repair changed scientific inputs or numbers; failed logs
  remain preserved.
- Handoff: the honest contribution is a generalization trade-off/controlled
  negative result rather than comprehensive superiority. Penultimate features
  provide statistically supported internal unseen-generator ranking gains and
  stronger compact degradation robustness, but statistically supported
  ranking degradation on Defactify. Part 10 must preserve this wording when
  freezing numbers.

### Part 10 — Final Statistics, Figures, and Number Freeze

- Status: completed and accepted; final reportable numbers are frozen.
- Statistics/artifact commit:
  `07d3fa662e74e3c45ed79551d35846bc4ecb31a5`.
- Number-freeze manifest commit:
  `061f9c813f65b0c73aeb13ff47bbc50a48f06a15`.
- Tests: 37/37 unit tests passed. The complete 2-model x 3-seed x 3-scope
  matrix, nine required metrics, paired deltas, bootstrap CIs, report tables,
  SVG validity, and all freeze-manifest SHA256 values passed final checks.
- Aggregate: `outputs/metrics/final_aggregate_results.csv` contains 27 exact
  scope/metric rows with B2 and A1 mean, sample SD, A1-minus-B2 delta, seeds,
  and threshold. No seed was excluded.
- Main result: A1 has supported internal GenImage unseen ranking gains
  (AUROC `+0.004123`, AUPRC `+0.005566`) and supported Defactify ranking
  degradation (full AUROC `-0.013021`; balanced AUROC `-0.013711`).
  Balanced-accuracy confidence intervals cross zero in all three scopes.
- Contribution wording frozen as: `generalization trade-off / controlled
  negative result`. A1 may be described as improving internal unseen-generator
  ranking and compact in-domain corruption robustness, but not as universally
  or comprehensively superior to B2.
- Controlled report set: exactly five primary artifacts are frozen:
  `reports/final_assets/method_diagram.svg`,
  `reports/final_assets/main_results_table.csv`,
  `reports/final_assets/core_ablation_table.csv`,
  `reports/final_assets/generalization_robustness.png`, and
  `outputs/figures/part09_error_comparison.png`.
- Outputs: `docs/FINAL_RESULTS_FREEZE.md` is the human-readable source for
  Parts 11–12; `artifacts/number_freeze_v2_manifest.json` is authoritative for
  source/artifact hashes, allowed claims, prohibited claims, limitations, and
  freeze policy.
- Freeze policy: no retraining, checkpoint reselection, threshold adjustment,
  seed exclusion, or silent numeric correction is permitted. Any correction
  requires explicit user approval, an audit note, and reopening the freeze.
- Handoff: Part 11 may clean reproducibility documentation and write the final
  report using only the frozen sources and claims. Supplemental plots may not
  replace or contradict the five primary artifacts.

## Current boundary

Part 10 is complete and final reportable numbers are frozen. Part 11 may
perform reproducibility packaging and report writing from
`docs/FINAL_RESULTS_FREEZE.md` and
`artifacts/number_freeze_v2_manifest.json` only. No retraining, reevaluation,
checkpoint reselection, threshold adjustment, seed exclusion, or numeric
correction is permitted without explicitly reopening the freeze. Do not start
Part 11 automatically.

## Status update template

After one Part completes, update the fields at the top, append a completion
entry with commit/tests/artifacts/decision/known issues, set the next Part, and
stop. At a Go/No-Go gate, set `Next part` to `Awaiting user approval` until the
decision is explicitly approved.
