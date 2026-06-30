# VEATIC 124 Alignment Candidate Fixes

| Candidate | Final? | Non-leaky? | Rationale |
|---|---|---|---|
| `keep_current_0s_as_primary_plus_report_offset_diagnostics` | True | True | 0s remains valid and nonzero best offsets vary by target/mode; use offset grid as diagnostic until global lag survives controls and grouped validation. |
| `diagnostic_global_offset_-1.75s` | False | False | Blocked PCA best offsets distribution negative=8, zero=1, positive=1; median=-1.75. |
| `target_framing_event_balanced_and_p3_movement` | True | True | Full-frame continuous MAE is dominated by stable zeros; balanced event-vs-stable and p3 movement rows better match the signal. |