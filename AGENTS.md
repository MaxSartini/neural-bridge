# Neural Bridge Repository Contract

- This repository is the new canonical private Git repository under construction.
- The existing internal and external project roots are forensic source material until migration closes.
- Never infer absence from GitHub: audit local, ignored, untracked, and external material.
- Work newest-to-oldest: VEATIC 2.1, zero-label, AGAIN Phase 7 to 0, original VEATIC, earliest exploration.
- Do not move a file or verified directory collection until `internal/migration/move-manifest.csv` records its source, destination, size, SHA-256 tree digest, role, and verification state.
- Same-volume moves should be atomic renames. Cross-volume moves require destination checksum verification before source deletion.
- Heavy data, caches, features, PCA, scores, models, weights, checkpoints, tensors, and complete runs belong under the ignored `artifacts` boundary on external storage.
- Canonical paths describe scientific programme, phase, artifact role, and lifecycle. Never organize by incidental machine/provider labels such as H100, MLX, drive names, upload bundle names, or temporary job names; preserve those details only in provenance manifests.
- Do not globally ignore scientific extensions. A heavy artifact placed outside `artifacts` must remain visible in Git status.
- Front-facing paths contain concluded, defensible results only. Active work and handoffs stay under `internal/` and are not linked from the root README.
- Each concluded phase owns its final runner, configuration, compact CSV/JSON results, report, provenance, external-artifact manifest, and only the minimum tests needed to defend that endpoint.
- Do not migrate tests wholesale. Keep a test only when it protects migrated live code, a scientific invariant, reproducibility, or a demonstrated failure mode. Leave redundant smoke, probe, scaffolding, hardware-tryout, and superseded-runner tests behind.
- Preserve decisive wins and failures that explain phase transitions; keep implementation noise out of normal navigation.
- Build VEATIC 2.1 scientifically from scratch. AGAIN may contribute hypotheses and neutral rigor utilities only after semantic compatibility is proven; never inherit an AGAIN head, fitted PCA, threshold, target, window, checkpoint, training recipe, or model selection.
- Keep `src/neural_bridge/again/` and the fresh `src/neural_bridge/veatic21/` implementation separate. Factor code into `src/neural_bridge/science/` only after both sides independently demonstrate identical semantics and provenance requirements.
- Do not add a root README until the scientific packages and navigation are settled.
- Do not configure a remote, commit, push, delete a source root, or migrate an active study without explicit approval.
