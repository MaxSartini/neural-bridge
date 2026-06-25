# AGAIN Cleaned Inventory and VEATIC Compatibility Audit

## Executive Answers

1. Complete enough to benchmark: yes for source completeness, but not yet for strict VEATIC-contract benchmarking without an explicit alignment convention. Strict <=1s aligned videos: 1/995; alignment-review candidates: 995/995.
2. Usable videos: 1 strict aligned; 995 candidates after resolving the systematic duration offset.
3. Usable annotation rows: 492 strict aligned; 487176 candidate rows before alignment trimming.
4. Arousal is available and continuous; valence is not present in the cleaned files.
5. 1Hz resampling is feasible for 995 videos.
6. VEATIC-style future spike targets are feasible for 995 videos, subject to target-threshold selection and train-only thresholding.
7. Best first split strategy: grouped leave-video/session-out, with participant-disjoint analysis as a secondary stricter split because participant IDs exist.
8. First encoding/benchmark run: first inspect a small representative sample to decide whether the roughly 3s video-over-annotation offset is pre-roll/post-roll. After that, create a 1Hz aligned manifest and run a small encoding pilot before scaling to all candidates.
9. Licensing/data hazards: use the saved public description/README and any upstream AGAIN license terms before redistribution; cleaned data are session-normalized, so avoid comparing absolute arousal across sessions as if globally calibrated.
10. Estimated TRIBE video encoding footprint: about 34.63 video-hours; at 1Hz, about 124652 frame/time rows before target trimming.

## Dataset Root

`$NEURAL_BRIDGE_EXTERNAL_ROOT/data/external/AGAIN/cleaned`

## Videos

- video files: 995
- readable videos: 995
- unreadable/corrupt videos: 0
- missing expected video files: 0
- videos with no annotation: 0
- videos with no metadata: 0
- candidates ready for encoding before alignment review: 995
- total duration hours: 34.63

## Annotations

- annotation rows: 487176
- sessions: 995
- median sampling rate estimate: 3.874 Hz
- arousal range: 0.0 to 1.0
- label type: continuous arousal, participant/session-level, session-normalized.

## Alignment

- videos with duration mismatch > 1s: 994
- usable aligned videos: 1
- alignment-review candidates: 995
- mismatch distribution is systematic rather than random: most videos have about three extra video seconds relative to annotation span.
- start timestamps are present as `[control]time_stamp`; raw wall-clock epoch is also present as `[control]epoch`.

## VEATIC-Compatible Manifest Proposal

The proposed manifest uses one row per cleaned annotation sample and can be resampled to 1Hz for VEATIC-style contracts. `video_id` is the video filename stem; `participant_id`, `session_id`, game, and genre are retained for grouped splits.

## Recommended Benchmark Modes

- Aggregate arousal benchmark: possible by averaging/resampling session labels, but not the first recommendation because labels are participant/session-level.
- Participant-level benchmark: supported and recommended.
- Leave-video/session-out: supported and recommended first.
- Leave-participant-out: supported as stricter generalization.
- Train-on-AGAIN validate-on-VEATIC: useful after scale/target calibration checks.
- Train-on-VEATIC validate-on-AGAIN: useful as a transfer stress test after a 1Hz AGAIN manifest exists.
- Pretrain-on-AGAIN fine-tune-on-VEATIC: plausible later, but only after baseline/control parity on AGAIN.

## Guardrails

No TRIBE encoding, model training, tensor export, VEATIC benchmark modification, or video re-encoding was performed by this audit.
