# AGAIN Dense 2Hz Label Alignment

## Scope

- This is a true 2Hz-on-2Hz supervised alignment for the dense H100 AGAIN cache.
- It uses saved dense-cache `time_seconds`; it does not collapse labels to 1Hz.
- No V-JEPA/TRIBE re-encoding, PCA, bridge training, or benchmark fitting was performed.

## Coverage

- dense rows: `243575`
- manifest rows: `243575`
- labeled rows: `243441`
- unlabeled rows: `134`
- videos total: `995`
- videos with labels: `995`
- videos with zero labeled rows: `0`
- videos with any unlabeled rows: `38`
- rows outside boundary: `0`
- rows outside tolerance/annotation: `134`
- rows missing AR context: `4153`
- first timestamp counts: `{'0.0': 864, '0.5': 131}`

## Targets

- `arousal_spike_rows_2_6_train_q90`: rows `237206`, videos `995`, source `future_arousal_max_delta_rows_2_6`
- `arousal_delta_p2rows_train_q90`: rows `241379`, videos `995`, source `future_arousal_delta_p2rows`
- `arousal_abs_delta_p4rows_train_q90`: rows `239358`, videos `995`, source `future_arousal_delta_p4rows`

Targets store continuous future movement values and masks. Binary event thresholds are selected inside each train fold during the benchmark; test labels do not set thresholds.

## 0.5s Movement Histogram

Absolute arousal movement over +1 dense row (+0.5s):

- `0.0` to `0.005`: `162075` rows
- `0.005` to `0.01`: `11038` rows
- `0.01` to `0.025`: `23255` rows
- `0.025` to `0.05`: `20189` rows
- `0.05` to `0.075`: `10054` rows
- `0.075` to `0.1`: `5616` rows
- `0.1` to `0.2`: `7675` rows
- `0.2` to `inf`: `2489` rows

## Guardrails

- between-second label movements preserved: `true`
- small-spike targets created: `true`
- primary row index source: dense cache `row_index.parquet`
- output manifest: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/labels_aligned_2hz.parquet`
- vjepa_encoding_run=`false`
- tribe_encoding_run=`false`
- pca_run=`false`
- benchmark_run=`false`
