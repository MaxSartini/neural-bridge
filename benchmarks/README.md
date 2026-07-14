# Benchmarks

This directory contains tracked benchmark evidence artifacts, not active heavy output roots.

## Current Contents

- `veatic/` - VEATIC-124 v2 benchmark reports, JSON summaries, manifests, and CSV audits used as the foundational half of the Neural Bridge evidence ladder.

## Where AGAIN Benchmark Evidence Lives

Current AGAIN evidence is organized under:

- `reports/`
- `evidence/phase_*`
- `evidence/current_phase_7_review/`
- `evidence/current_phase_5_5_review/` for the detailed historical ladder

The current strongest result is the Phase 7 grouped continuous checkpoint ensemble: `420/420`, `15/15` positive fold-groups versus AR and controls on both primary metrics, failed gates `[]`.

Heavy AGAIN output roots remain under ignored `outputs/` directories and are not force-added to git.

## Policy

Do not add checkpoint files, tensors, `.npy`, `.npz`, dense caches, or full generated output roots here. Benchmark evidence added to git should be small, inspectable CSV/JSON/Markdown artifacts with clear source reports.
