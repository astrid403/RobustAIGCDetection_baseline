# RobustAIGCDetection Staged Execution Master Plan

## Scope and objective

This is a five-day Computer Vision course project about robust AI-generated
image detection. The deliverable is a reproducible research narrative:
reliable baselines, one literature-motivated improvement, fair pilots and
ablations, held-out and external evaluation, robustness/error analysis, and a
compact report and presentation. This plan does not authorize a larger
research program.

Protocol v2 in `docs/FINAL_PROTOCOL.md` is authoritative. Label `0` is real
and label `1` is fake. Defactify is final external evaluation only and must
never influence architecture, feature layer, hyperparameters, augmentation,
checkpoint, calibration, or threshold.

## Frozen baseline and selected routes

- B0: ImageNet-pretrained ResNet50, end-to-end.
- B1: frozen CLIP final embedding with one linear head.
- B2: frozen CLIP final embedding with the existing two-layer MLP.
- Main proposed route: frozen CLIP final and penultimate image features with a
  lightweight dual-level fusion classifier.
- Pre-approved fallback: penultimate-only frozen CLIP feature classifier.
- Negative-result route: if neither candidate beats B2 fairly and consistently,
  freeze B2 as the deployed model and report the attempted method, controlled
  ablation, failure analysis, and robustness/generalization findings without
  cherry-picking.

No frequency branch, residual branch, new backbone, additional training data,
or Defactify-driven redesign may be introduced.

## Part dependency graph

```text
Part 01 baseline/protocol freeze [complete]
  -> Part 02 metrics/tests/B1 [complete]
    -> Part 03 CLIP feature contract and cache safety
      -> Part 04 penultimate-only implementation and pilot
        -> Part 05 dual-level fusion implementation and pilot
          -> Part 06 fair ablation + Go/No-Go + model freeze
            -> Part 07 final three-seed training
              -> Part 08 formal held-out/external evaluation
                -> Part 09 robustness, efficiency, error cases
                  -> Part 10 statistics, tables/figures, number freeze
                    -> Part 11 reproducibility package and report
                      -> Part 12 presentation and final submission audit
```

Parts may not be skipped unless a Part file explicitly defines a fallback that
preserves the dependency. Parts 7 onward require a recorded Part 6 user
approval. Part 11 requires frozen numbers from Part 10. Part 12 requires the
main report conclusions from Part 11.

## Gates and freezes

### Gate A — baseline reliability

Completed by Parts 1–2. Protocol, artifact hashes, metric definitions, and B1/B2
compatibility must remain valid.

### Gate B — feature extraction correctness

Part 3 must prove exact final/penultimate extraction positions, shapes,
determinism, and cache separation before candidate training.

### Gate C — candidate viability and Go/No-Go

Part 6 compares seed-42 B1/B2, penultimate-only, fusion, and a parameter-fair
control using the same data, budget, threshold policy, and metric code.

- Go fusion: implementation is valid and fusion gives a meaningful,
  non-pathological validation/GenImage-unseen advantage or a clearly justified
  robustness/generalization signal without degrading core metrics severely.
- Fallback: fusion fails but penultimate-only gives the strongest credible
  signal; select penultimate-only.
- Negative route: neither improves credibly; select B2 and preserve candidate
  results as negative evidence.
- Stop: data leakage, inconsistent budgets, cache contamination, irreproducible
  results, or a correctness bug invalidates the comparison.

The user must approve the decision. That approval freezes architecture,
feature mode, head, loss, augmentation, optimizer, epochs, checkpoint rule,
seeds `[42, 43, 44]`, and threshold policy.

### Model freeze

Occurs at the approved end of Part 6. Later Parts may fix a correctness bug
only after stopping, documenting impact, and obtaining user approval. No new
structure or tuned variant is allowed.

### Number freeze

Occurs in Part 10 after provenance, completeness, statistics, and plots/tables
are audited. Afterwards there is no retraining, checkpoint reselection,
threshold adjustment, or selective exclusion. Corrections require reopening
the freeze with an explicit audit trail and user approval.

## Formal claims and negative results

- Clean AUROC improvement: claim improved cross-generator/cross-dataset ranking
  only if seed-level evidence supports it.
- Similar AUROC with better robustness: claim a robustness trade-off, not
  general superiority.
- Better balanced accuracy, class recall balance, or worst-generator score:
  claim improved operating-point or subgroup behavior.
- No stable improvement: report a controlled negative result explaining what
  final versus penultimate CLIP features contribute and where external
  generalization remains limited.
- Never report only the best seed or tune on Defactify.

## Time budget

Target total is about five working days: Parts 1–3 (0.8 day), Parts 4–6 (1.2
days plus GPU pilots), Parts 7–8 (1.5 days plus queued GPU time), Parts 9–10
(0.8 day), Parts 11–12 (0.7 day). If time is constrained, paired bootstrap is
the first optional item to drop; core metrics, three seeds, formal evaluation,
one compact robustness analysis, error cases, report, and reproducibility are
not optional.

## Global experiment and Git rules

- Naming: `{stage}_{model}_{training_scope}_{evaluation_scope}_seed{seed}_v2`.
- Stages are exactly `smoke`, `pilot`, and `final`; never reuse their outputs.
- Every formal result must map to a config, commit, seed, command, checkpoint
  hash, prediction hash, and metric file.
- Commit source, tests, configs, lightweight summaries, figures, documentation,
  and manifests. Do not commit data, caches, checkpoints, feature tensors, full
  predictions, copied error images, or large logs.
- Never overwrite an existing run directory except an explicitly documented
  resume.
- Continue on `feature/defactify-external-eval` unless a Part or user explicitly
  authorizes another branch. Never merge or push automatically.

## Standard completion summary

```text
Part XX completion summary

Status:
Branch:
Start commit:
Commit:
Tests:
Commands executed:
Files changed:
Files added:
Artifacts generated:
Key metrics:
Decision:
Known issues:
STATUS.md updated:
Ready for next Part:
```
