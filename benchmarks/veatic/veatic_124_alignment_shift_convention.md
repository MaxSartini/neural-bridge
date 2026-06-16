# VEATIC 124 Alignment Shift Convention

`offset_seconds > 0` means feature rows are sampled later than the label-anchor row: `feature_time = label_anchor_time + offset_seconds`.

Equivalently, positive offset tests whether later cortical features align better with the target anchored at the current label row. Negative offset tests whether earlier cortical features align better and is the expected direction for a real early-warning signal.

Targets are constructed first from label rows at the label anchor time. The feature shift is then applied inside each split using only rows available in that split. Rows whose shifted feature time falls outside the same-video split region are trimmed.

PCA bases are fit once per feature mode and split/fold using train rows only; offset scans reuse that split-local basis.