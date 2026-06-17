# Current Artifact Port Audit - 2026-06-17

This note records the current VEATIC v2 artifacts ported from the old local
checkout into the Neural Bridge repo. The goal was to restore the files active
scripts and future Codex sessions need, without dragging forward legacy clutter.

## Ported

- `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- `benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json`
- `benchmarks/veatic/veatic_manifest_124_validation_20260616.md`
- Current `benchmarks/veatic/veatic_124_*` alignment, event-conditioned, and
  spike/core sidecars.
- Current `benchmarks/veatic/veatic_neuro_benchmark_124video_*` v2 feature-mode
  summaries and JSON outputs.
- `outputs/veatic_124_temporal_context_v2_20260616_1557/`, including best
  windows, ablations, specificity checks, leakage audit, report, and summary.
- Selected current sidecars from
  `outputs/veatic_124_temporal_fairness_20260616_1509/` that are reused by the
  v2 temporal context workflow.

## Intentionally Not Ported

- Older VEATIC-89 device-audit outputs.
- Old transition artifacts from inactive validation branches.
- The large temporal fairness `balanced_event_stable_results.csv` file, because
  it is not needed by the active v2 context scripts and would add unnecessary
  repository weight.
- Raw TRIBE cache contents. Those stay outside git under the external asset
  root.

## Cache Root

The local Neural Bridge `.env` now points `NEURAL_BRIDGE_EXTERNAL_ROOT` at the
external asset tree. `backend/scripts/run_veatic_strict_benchmark.py` also reads
that local `.env` value when the variable is not exported in the shell, so dry
runs and future sessions resolve the live cache consistently.

Expected external cache:

```text
<external-assets-root>/benchmarks/veatic/tribe_cache
```

Former external asset paths may exist as compatibility symlinks on a local
machine, but new docs and reports should refer to the Neural Bridge external
asset root or the placeholder above.

## Follow-Up Note

An untracked temporal-head script from another active session still contains an
old-repo fallback path. It was not part of this artifact port and was not staged.
Before that temporal-head work is committed, remove the fallback or point it at
the Neural Bridge output directory now present in this repo.
