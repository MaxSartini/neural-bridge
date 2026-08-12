# 0005 — Plain glosses lead, technical lines stay, and glosses carry no digits

**Status:** Accepted
**Date:** 2026-08-12

## Context

The dashboard was built for the people who ran the programme. Its hero read
"Turning predicted cortical activity into forward-looking response
intelligence" and its cards were labelled "Blocked event PR-AUC". Those are
correct and they are the right words for that audience.

They are the wrong first words for the audience the programme is now reaching
for. The user asked for "a whole lot more layman's language to make it easier to
understand for prospective users and investors when first using the platform",
and the work now spans an internal evidence surface and a client-facing product
that must both be legible to someone who has never met the term PR-AUC.

Three mechanisms were considered and declined: an audience toggle swapping copy
between registers (doubles the copy and lets the two drift), a glossary page
(sends the reader away at the exact moment they are deciding whether to care),
and per-card explainers (adds a third register to keep consistent). The risk in
all of this is not verbosity — it is that simplification quietly becomes
overclaiming, on the one surface where overclaiming does the most damage.

## Decision

Every card, chart and hero leads with a plain-English **gloss** and keeps the
precise metric directly beneath it as the **technical line**. Nothing precise is
deleted; the jargon stops being the first thing read.

- All glosses live in one module, `web/src/data/plainLanguage.ts`, so every
  outward claim the interface makes can be reviewed in a single sitting.
- **A gloss carries no digits.** `plainLanguage.test.ts` fails the build on one.
- **A gloss stays inside the root README's "Honest boundaries"** — no mind
  reading, no individual profiling, no medical inference, no universal emotion
  recognition, no exact second-by-second trajectory, no guaranteed outcome.
- Technical lines for landmarks and datasets are **composed from**
  `scorecard.ts` (`landmarkTechnicalLine`, `datasetTechnicalLine`), never
  restated.

## Consequences

- **The no-digits rule is the load-bearing part.** `+28.80%` means a gain in
  blocked event PR-AUC over a trained autoregressive baseline. Lifted out of
  that label into a friendly sentence — "29% better at finding the moments
  people react to" — it becomes a claim about something the evidence does not
  measure. Keeping figures welded to their metric is what lets the copy get much
  friendlier without drifting. The rule *will* feel obstructive when a sentence
  wants a number; spell it out ("fifteen") or move it to the technical half.
  Do not relax the test.
- That test was mutation-checked against a scratch copy: injecting `15` into one
  gloss failed exactly that gloss and nothing else. Per CONVENTIONS.md, a suite
  never seen failing is not known to test anything.
- The registry is checked for completeness in both directions — every landmark
  and dataset has a gloss, and no gloss survives its subject being deleted.
- **Keeping glosses out of `scorecard.ts` is deliberate.** ADR-0002 makes that
  file a hand-transcribed record of `results/README.md`, with a `sourceDoc` per
  entry. Mixing authored copy into it would blur transcribed evidence and
  written claim. Composing the technical lines from it rather than restating
  them also avoids doubling the manual-sync risk ADR-0002 already flags.
- Both surfaces read from the same registry, so the customer-facing Studio and
  the internal evidence dashboard cannot drift into telling different stories
  about the same numbers.
- The declined mechanisms remain available. A glossary is the most likely
  addition; it would supplement the glosses, not replace them.
