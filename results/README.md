# Concluded Results

This page contains the strongest claim-bearing results only. The complete development record and diagnostic branches remain with their study packages, keeping this scorecard focused on concluded wins.

## Original VEATIC: event signal established

For `arousal__future_spike_1_3s@0.05`, the strongest blocked row reached PR-AUC `0.2536` versus AR `0.1969` (`+28.80%`), shuffled `0.1840` (`+37.83%`), and random `0.1944` (`+30.45%`). The balanced event-vs-stable evaluation reached PR-AUC `0.3394`. VEATIC's confirmed scope was future-event ranking; continuous specialization was established later on AGAIN.

[Study closure](../studies/original-veatic/v2-closure/README.md) · [claim report](../studies/original-veatic/v2-closure/report.md) · [machine evidence](../studies/original-veatic/v2-closure/results/)

## AGAIN Phase 5: event confirmation

The selected temporal residual head passed the full `420/420` controlled matrix across strict blocked-temporal and grouped held-out-video protocols.

| Protocol | Real residual | Frozen AR | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Blocked temporal | `0.267074` PR-AUC | `0.260234` | `+0.006840` | `+2.63%` |
| Grouped held-out video | `0.231383` PR-AUC | `0.217495` | `+0.013888` | `+6.39%` |

[Phase 5 journey](../studies/again/phase-05-learned-bridge/README.md) · [selected-head evidence](../studies/again/phase-05-learned-bridge/evidence/phase_5_5_selected_head_420_confirmation_20260714_124953/README.md)

## AGAIN Phase 7: continuous future-movement ranking

The grouped held-out-video confirmation passed all gates. Neural Bridge beat the separately trained frozen AR and the strongest matched controls in every fold-group.

| Endpoint | Neural Bridge | Frozen AR | Absolute gain | Relative gain | Positive groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spearman | `0.260301` | `0.240537` | `+0.019764` | `+8.22%` | `15/15` |
| Top-5% movement lift | `0.097598` | `0.089566` | `+0.008032` | `+8.97%` | `15/15` |

The claim-bearing protocol is grouped held-out-video confirmation. Blocked-temporal and grouped evidence remain separate because they test different forms of generalization. The confirmed result supports ranking and top-tail lift, not exact continuous trajectory prediction.

[Phase 7 summary](../studies/again/phase-07-continuous/evidence-summary.md) · [grouped report](../studies/again/phase-07-continuous/grouped-confirmation/reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md) · [machine result](../studies/again/phase-07-continuous/grouped-confirmation/metrics/result.json)

## AGAIN zero-label-at-inference: locked confirmation

One prospectively locked video-only candidate was trained on 696 development videos and evaluated once on 299 untouched videos. At inference it used no observed arousal, response history, teacher score, or labeled warm start.

| Endpoint | Neural Bridge | Strongest control | Relative gain | Panel wins |
| --- | ---: | ---: | ---: | ---: |
| Spearman | `0.178513` | `0.100488` | `+77.65%` | `5/5` |
| Top-5% movement lift | `0.076608` | `0.044852` | `+70.80%` | `5/5` |
| Event PR-AUC | `0.171062` | `0.135230` | `+26.50%` | `5/5` |

All paired whole-video bootstrap lower bounds were positive, and the first-30-second cold-start tier passed. This remains supervised learning; “zero-label” describes inference inputs, not training.

[Zero-label evidence](../studies/again/zero-label/evidence-summary.md) · [locked report](../studies/again/zero-label/locked-confirmation/reports/again_dense_2hz_zero_label_direct_supervised_locked_confirmation_20260715.md) · [machine result](../studies/again/zero-label/locked-confirmation/metrics/locked_confirmation_result.json)
