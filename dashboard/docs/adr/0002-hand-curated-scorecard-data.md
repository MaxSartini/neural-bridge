# 0002 — Scorecard numbers are hand-curated, not extracted

**Status:** Accepted
**Date:** 2026-08-11 (recorded retroactively; decision taken during Phase A2)

## Context

The [Scorecard](../../CONTEXT.md#scorecard) compares Neural Bridge against Frozen AR and the
strongest matched control across ten research phases. The underlying evidence exists, but
each phase's `result.json` carries its own field names and its own `schema_version` — verified
directly against `phase-06`, `phase-07`, `zero-label`, and the VEATIC result files. There is no
unified shape to extract from.

A generic extractor would therefore need per-phase mapping code, and would break every time a
new phase lands with different keys. Meanwhile `results/README.md` already exists and the
repository itself treats it as authoritative: *"This page contains the strongest claim-bearing
results only."*

## Decision

Transcribe the scorecard numbers once, by hand, into `web/src/data/scorecard.ts` as a typed
dataset. Every entry carries a `sourceDoc` path so each card and chart links back to its own
evidence. The file header states that it is a transcription.

## Consequences

- **This is a real manual-sync risk and it is not mitigated.** If a phase concludes and the
  `results/README.md` table changes, `scorecard.ts` must be updated by hand or the dashboard
  will silently show stale numbers. There is no checker. This is flagged rather than pretended
  away.
- A live-sync validator (parse the Markdown table, diff against the typed dataset, fail loudly)
  is the obvious mitigation and is deliberately deferred — real work for a tool that is still
  internal-only. This is the most likely reason to reopen this ADR.
- The existing `/doc/results/README.md` Markdown view is kept, unchanged, alongside the
  scorecard. The charts are a second lens on the same curated numbers, never a replacement, so
  the raw table is always one click away.
