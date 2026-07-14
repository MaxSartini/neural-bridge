# Neural Bridge Evidence Bundle: Phase 0-5 (2026-06-25)

This lightweight bundle packages review evidence for Neural Bridge from VEATIC through dense AGAIN Phase 0-5. It is intentionally not a runnable cache bundle: heavy tensors, videos, checkpoints, model weights, PCA arrays, and per-video feature caches are excluded.

## Current Status Note

This bundle is a historical evidence snapshot through `2026-06-25`. It remains valid as foundational evidence, but the current repo-level story is now the paired VEATIC + AGAIN Phase 5.5 evidence ladder documented in `../../docs/neural_bridge_phase5_5_evidence_ladder.md`.

VEATIC-124 v2 established the original controlled future arousal spike/event-ranking signal. Later AGAIN work replicated and extended it at scale with dense 995-video V-JEPA 2.1 / TRIBE v2 features, frozen-AR residual repair, a redesigned washout-gap binary target, a 10-seed blocked temporal confirmation, and a repaired grouped-video compatibility verdict.  is contained in this bundle or promoted by this note.

## Phase Map

- VEATIC evidence: strict VEATIC-124 v2 benchmarks, timing/fairness checks, raw-vs-compressed representation audit, frozen tensor exports, and trained-head summaries.
- AGAIN Phase 0: dataset/H100/cache readiness audits, cleaned inventory, alignment/boundary handoff evidence, and dense-root metadata.
- AGAIN Phase 1: true 2Hz label alignment and target construction evidence.
- AGAIN Phase 2: AR-only 2Hz baseline evidence.
- AGAIN Phase 3: raw cortical/TRIBE-v2 versus AR benchmark evidence.
- AGAIN Phase 4: fold-safe train-only PCA bridge benchmark evidence.
- AGAIN Phase 5: MLX learned-head benchmark evidence plus compact label-permutation sanity run.

## Canonical Evidence Locations

- Top-level docs: `README.md`, `ROADMAP.md`, `REQUIREMENTS.md`, `AGENTS.md`, `.codexignore`.
- Project docs: `docs/PROJECT_MEMORY.md`, `docs/current_project_state.md`, `docs/again_dense_h100_cache.md`, `docs/external_assets_manifest.md`, and VEATIC evidence docs.
- VEATIC: `benchmarks/veatic/` and selected `outputs/veatic_*` metadata/report folders.
- AGAIN Phase 0: `outputs/again_cleaned_inventory_audit_20260621_123531/`, `outputs/again_alignment_offset_diagnosis_20260621_131041/`, `reports/again_dense_h100_local_audit_20260625.md`, and `again_dense_root_metadata/`.
- AGAIN Phase 1: `reports/again_labels_aligned_2hz_20260625_091209.md`, `again_dense_root_metadata/labels_aligned_2hz.parquet`, and label summaries.
- AGAIN Phase 2: `reports/again_dense_2hz_ar_baseline_20260625_093722.md` and `outputs/again_dense_2hz_ar_baseline_20260625_093714/`.
- AGAIN Phase 3: `reports/again_dense_2hz_raw_cortical_vs_ar_20260625_094242.md` and `outputs/again_dense_2hz_raw_cortical_benchmark_20260625_093733/`.
- AGAIN Phase 4: phase 4 reports plus `external_root/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/` lightweight metrics, promotion gates, diagnostics, and manifests.
- AGAIN Phase 5: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/` and `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`, excluding checkpoints.
- Compressed large evidence: `large_evidence_compressed/` contains gzip-compressed CSVs that were too large to keep as raw CSV in git.

## Headline Results

Historical results contained in this bundle:

- VEATIC best spike PR-AUC: `0.2536` vs AR `0.1969`.
- AGAIN Phase 3 AR+raw PR-AUC: `0.17030` vs AR `0.14725`.
- AGAIN Phase 4 best PR-AUC: `0.17165` vs AR `0.14725`.
- AGAIN Phase 5 main learned-head PR-AUC: `0.21913` vs Phase 4 `0.17165` and AR `0.14725`.
- AGAIN Phase 5 label-permutation follow-up best real grouped PR-AUC: `0.22458`; label permutation PR-AUC `0.10428`; blocked temporal real PR-AUC `0.20712`.

Later current-facing results are documented outside this historical bundle, especially the AGAIN blocked washout-gap confirmation (`0.2670735630` real PR-AUC vs `0.2602336231` frozen AR) and repaired grouped compatibility (`0.2313831909` real PR-AUC vs `0.2174953276` AR/frozen).

## Intentional Omissions

Hard-excluded payloads include `per_video/`, `features/`, `pca_components/`, `score_parts/`, `cache/`, `checkpoints/`, broad `.cache/` payload paths, NumPy/model binaries, videos, and audio. Critical evidence files under 70MB are allowed. The approved explicit exception is `again_dense_root_metadata/labels_aligned_2hz.parquet`, which preserves Phase 1 labels for review. Oversize CSV evidence is stored as deterministic gzip under `large_evidence_compressed/`. Larger non-CSV files are otherwise omitted. See `omitted_large_files_manifest.json`.

## Compressed Large Evidence

The following raw CSVs exceeded the normal evidence cap and are included as `.csv.gz` inside `large_evidence_compressed/`:

- `again_dense_root_metadata/row_index.csv`
- `outputs/again_boundary_aligned_1hz_manifest_20260621_205412/again_boundary_aligned_1hz_manifest.csv`
- `outputs/again_cleaned_inventory_audit_20260621_123531/again_manifest_proposal.csv`

To inspect one:

```bash
gzip -cd large_evidence_compressed/again_dense_root_metadata/row_index.csv.gz | head
```

## Verify Checksums

```bash
python3 - <<'PY'
import hashlib, json, pathlib
bundle = pathlib.Path('evidence/phase_0_to_5_historical_ladder_20260625')
rows = json.loads((bundle / 'checksum_manifest.json').read_text())['files']
for row in rows:
    path = bundle / row['bundle_path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row['sha256']:
        raise SystemExit(f"checksum mismatch: {path}")
print(f"verified {len(rows)} bundled files")
PY
```

Generated at `2026-06-25T20:36:27.459939+00:00`.
