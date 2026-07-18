# AGAIN Phase 7: Continuous Movement Ranking

Phase 7 cracked a distinct problem from the earlier event/spike result: ranking future arousal movement and identifying its highest-movement tail.

The claim-bearing result is the grouped held-out-video confirmation:

1. Grouped held-out-video closure passed `420/420` comparison rows.
2. Neural Bridge beat frozen AR and matched controls in `15/15` fold-groups.
3. Spearman improved from `0.2405` to `0.2603` (`+8.22%`), while top-5% lift improved from `0.0896` to `0.0976` (`+8.97%`).

This confirms continuous future-movement ranking and top-tail selection on unseen videos. The earlier diagnostic and blocked-temporal work remains separately documented because those protocols answer different questions; neither is used to discount the grouped confirmation.

Each evidence directory contains only its audit, run manifest, compact metrics, rows, and report. Full checkpoints, fold matrices, training curves, and predictions remain in the registered external runs. The canonical engine and checksum-locked replay specs replace the two phase-coupled entrypoints.
