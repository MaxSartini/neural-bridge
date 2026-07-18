# AGAIN Phase 6: Event Stabilization

Phase 6 records a necessary failure-to-win sequence.

The Optuna single-seed pilot was promising, but its selected configuration did not robustly improve the reference recipe under locked multi-seed confirmation. Trial 4 then failed fresh blocked confirmation, a fixed within-seed blend failed, and its three-checkpoint variant also failed. None is a promoted result.

The successful hypothesis was simpler: prospectively average independently trained checkpoints from the already validated reference recipe. That passed fresh blocked confirmation and grouped held-out-video closure. Final grouped event PR-AUC was `0.2344` versus frozen AR `0.2180` and best matched control `0.2180`, positive in `15/15` fold-groups with no failed gates.

`runs/` retains only each branch’s result, report, run manifest, and decisive audit where available. `plans/` preserves the prospective decisions. The ten phase-coupled entrypoints are superseded by the canonical engine; checkpoints, scores, predictions, databases, and fold material remain in the registered external collection.
