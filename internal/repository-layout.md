# Repository Layout

This repository separates the reader-facing scientific record from active work and heavy artifacts.

- `results/`: strongest concluded, defensible results.
- `studies/`: phase-closure packages for original VEATIC, AGAIN, zero-label, and VEATIC 2.1.
- `src/neural_bridge/`: reusable scientific implementation.
- `apps/`: active product applications that survive the audit.
- `tests/`: a small suite protecting live shared code and scientific contracts; not a historical test archive.
- `docs/`: concise overview, methods, rigor, reproducibility, and reference material.
- `registry/`: study, claim, and external-artifact indexes.
- `internal/`: active research, decisions, handoffs, and migration records; absent from front navigation.
- `artifacts`: ignored local symlink to the organized external artifact root.

Canonical names are scientific, not operational. Hardware, cloud provider, drive, bundle, and temporary job names are recorded in provenance but never used as destination taxonomy. Material found under an operational bundle is relocated according to its actual programme and phase.

Reports, compact results, and final experiment runners belong to their owning study. There are no generic top-level dumping grounds for reports, outputs, or benchmark results.

Tests migrate by value, not by ancestry. A retained test must protect a final runner, shared implementation, scientific invariant, provenance gate, or previously demonstrated failure. One-off smoke tests, redundant unit variants, exploratory hardware probes, and tests tied only to code left behind do not migrate.
