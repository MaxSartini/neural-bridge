# Architecture Decision Records

One file per decision, `NNNN-kebab-title.md`, numbered in the order they were taken. An ADR is
a record of a decision *and the constraints that forced it* — the point is that a later reader
(human or agent) can tell the difference between a deliberate trade-off and an accident.

Format:

```markdown
# NNNN — Title

**Status:** Accepted | Superseded by NNNN | Reopened
**Date:** YYYY-MM-DD

## Context
What was true that made this a decision rather than an obvious call.

## Decision
What we do.

## Consequences
What this costs, what it rules out, and what would make us revisit it.
```

Decisions recorded here are **not re-litigated** by routine architecture reviews. If a review
surfaces friction real enough to warrant reopening one, it must say so explicitly and argue
against the ADR's stated consequences — not merely propose the alternative the ADR already
rejected.
