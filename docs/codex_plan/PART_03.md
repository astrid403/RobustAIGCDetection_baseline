# Part 03 — Verify CLIP Feature Contract and Cache Safety

## Goal and type

Inspect `open_clip` and establish a tested contract for extracting normalized
final and penultimate image features without implementing a classifier or
running a pilot. Type: read-only inspection plus focused code/tests.

## Preconditions

- Part 2 commit `fdc149acb1e203a66bb30d3ffc21d39be36dd12a`.
- Protocol v2 and STATUS agree that Part 3 is next.
- Existing CLIP weights/environment are locally available; no download is
  authorized.

## Allowed and forbidden scope

Read model, feature cache, train/evaluate code, installed `open_clip` source,
configs, and tests. Modify only CLIP feature extraction, cache fingerprint
metadata/path validation, narrowly necessary callers/config schema, and tests.
Run unit tests plus a tiny deterministic extraction smoke on one or a few
existing images/tensors.

Do not implement penultimate-only or fusion heads; train; run full dataset
extraction/evaluation; touch Defactify outputs; or change Protocol v2.

## Tasks and validation

1. Record installed `open_clip` version, ViT-B-32 visual module layout, and the
   exact semantic location of final and penultimate features.
2. Define feature modes (`final`, `penultimate`) and expected shapes/dtypes.
   The penultimate definition must be unambiguous and obtained without hooks
   whose ordering is unstable.
3. Extract both modes for identical inputs, verify batch dimension and finite
   values, and verify repeated eval-mode extraction is deterministic within a
   stated tolerance.
4. Ensure normalization is deliberate and documented for both modes.
5. Extend cache metadata/signature/path so model name, pretrained tag,
   feature mode, extraction version, manifest fingerprint, and relevant
   preprocessing cannot collide.
6. Test cache separation, stale/mismatched rejection, shapes, determinism,
   final backward compatibility, and no-grad/frozen behavior.

## Outputs and acceptance

- Focused implementation/tests and a lightweight extraction-contract note if
  needed.
- All tests pass.
- Same image repeated extraction matches tolerance.
- Final and penultimate shapes match the inspected architecture.
- Cache signatures differ by feature mode and reject wrong metadata.
- Existing final-feature cache remains either safely compatible or is rejected
  with a clear migration reason; it is never silently misread.
- No pilot/formal artifacts and no unrelated diff.

## Stop, repair, and rollback

Stop if installed architecture cannot provide a stable penultimate definition,
weights would need downloading, existing final cache could be silently
contaminated, or deterministic extraction fails. Allow two focused repair
rounds. Do not substitute another backbone/layer without user approval.

## Git and handoff

Commit after acceptance, suggested:
`feat: add verified CLIP feature modes and cache fingerprints`.
Do not push/merge. Handoff includes exact extraction locations, shapes,
normalization, determinism tolerance, cache schema/version, tests, commit, and
the approved feature-mode names required by Part 4.
