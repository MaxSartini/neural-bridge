# VEATIC 124 Causal Window Audit

- `arousal labels current manifest`: observed/current; future feature leakage=False. Manifest rows read averaged labels at row timestamp.
- `future_change`: label-only future target construction; future feature leakage=False. Allowed target y(t+h)-y(t); not used as feature.
- `event_future_spike_1_3s`: label-only future target construction; future feature leakage=False. Allowed binary future-label target.
- `residual_future_p*_rolling3`: causal/past-only baseline plus future target; future feature leakage=False. History window uses current/past labels; future label is target only.
- `local target future_minus_rolling_baseline`: causal/past-only baseline plus future target; future feature leakage=False. Rolling baseline uses current/past values in retest code.
- `local target future_change_local_volatility`: causal/past-only denominator plus future target; future feature leakage=False. Local volatility uses current/past values; near-perfect local-volatility rows remain suspicious and should not be headline.
- `pca64_delta delta1`: causal/past-only; future feature leakage=False. Uses current minus previous within video.
- `pca64_delta accel`: causal/past-only; future feature leakage=False. Uses current/previous deltas only.
- `pca64_delta rollmean3`: causal/past-only; future feature leakage=False. Window is current plus prior two rows within video.
- `pca64_delta slope3`: causal/past-only; future feature leakage=False. Slope over current plus prior two rows.
- `pca64_delta slope5`: causal/past-only; future feature leakage=False. Slope over current plus prior four rows.
- `pre_event masks`: label-only future/onset diagnostic; future feature leakage=False. Masks filter test rows for diagnostics; thresholds remain train-selected.
- `event_plus_pre masks`: label-only future/onset diagnostic; future feature leakage=False. Positive-only masks have undefined PR-AUC; use balanced event-vs-stable for discrimination.