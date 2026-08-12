/**
 * The plain-gloss layer: every outward-facing claim the interface makes, in one
 * auditable file.
 *
 * Neural Bridge is no longer read only by the people who ran the programme.
 * A newcomer — a prospective customer, an investor — bounces off a card
 * labelled "Blocked event PR-AUC". So every card, chart and hero leads with a
 * plain-English sentence and keeps the precise metric directly beneath it.
 * Nothing precise is deleted; the jargon just stops being the first thing read.
 *
 * ## Two rules, and why
 *
 * **A gloss carries no digits.** Figures stay in the numeral and the technical
 * line, where the metric that defines them is named. `+28.80%` means a gain in
 * blocked event PR-AUC over a trained autoregressive baseline; lifted out of
 * that label and into a friendly sentence it reads as a far broader claim than
 * the evidence supports. `plainLanguage.test.ts` enforces this — it is the one
 * mechanical guard on the whole layer, so do not weaken it to fit a sentence.
 * Spell the number out, or move it to the technical half.
 *
 * **A gloss stays inside the README's "Honest boundaries".** That section
 * states what Neural Bridge does not claim: mind reading, individual
 * profiling, medical inference, universal emotion recognition, exact
 * second-by-second trajectories, guaranteed audience or commercial outcomes.
 * Simplifying language is not licence to overclaim. Prefer "where response is
 * likely to move" over "what viewers will feel", and never promise an outcome.
 *
 * ## Why this is not in scorecard.ts
 *
 * ADR-0002 makes `scorecard.ts` a hand-transcribed record of
 * `results/README.md`, each entry carrying a `sourceDoc`. Mixing written copy
 * into it would blur what is transcribed evidence and what is authored claim,
 * and reviewing the boundaries above would mean reading both concerns at once.
 *
 * Note the asymmetry below: landmark and dataset technical lines are *composed
 * from* `scorecard.ts` rather than restated here. ADR-0002 already flags the
 * manual-sync risk on those numbers; transcribing them a second time into this
 * file would double it.
 */

import type { Dataset, LandmarkComparison } from "./scorecard";

export interface PlainGloss {
  /** Leads the element. Never contains a digit. */
  plain: string;
  /** The precise line beneath it. Digits belong here, beside their metric. */
  technical: string;
}

/** The Overview hero. The technical half is the programme's own one-liner. */
export const heroGloss: PlainGloss = {
  plain: "See where a video will hold attention — before you test it on people.",
  technical:
    "Frozen V-JEPA 2.1 / TRIBE v2 predictions, converted by the Neural Bridge " +
    "temporal model into future spike, continuous-movement, and heat-map signal — " +
    "validated across VEATIC and AGAIN under matched controls.",
};

/**
 * Keyed by `LandmarkComparison.id`. Plain half only — the technical half comes
 * from `landmarkTechnicalLine` so the figures keep one owner.
 * `plainLanguage.test.ts` asserts this covers every landmark.
 */
export const landmarkPlain: Record<string, string> = {
  veatic: "Finds the moments that spike, better than predicting from history alone.",
  "again-event": "Holds up on a second, unrelated video set — in every one of fifteen checks.",
  "phase7-continuous": "Ranks the whole timeline, not just the spikes.",
  "zero-label": "Works on a video it has never seen, with no audience data at all.",
};

/** Keyed by `Dataset.id`. The counts live on the card as its numeral. */
export const datasetPlain: Record<string, string> = {
  again: "Gameplay sessions, sampled twice a second from start to finish.",
  veatic: "Film, television and documentary clips, each rated by dozens of people.",
};

/** Keyed by the scorecard's chart slots. */
export const chartGlosses: Record<string, PlainGloss> = {
  gain: {
    plain: "How much better than the next-best method.",
    technical:
      "Percentage gain of the real Neural Bridge lane over the frozen AR/baseline and " +
      "the strongest matched control, for each landmark's primary metric. Click a bar " +
      "to open that landmark's evidence. (Zero-label reports a single control, so both " +
      "bars read the same figure there.)",
  },
  progression: {
    plain: "Rich features weren't useful on their own — this is what the bridge added, step by step.",
    technical:
      "Grouped event PR-AUC as the system progressed from raw predicted cortical " +
      "features to the frozen-AR residual bridge.",
  },
  failure: {
    plain: "The failure that made the bridge necessary.",
    technical:
      "On the early blocked AGAIN benchmark, adding raw cortical features to a strong " +
      "AR baseline did not merely fail to help — it made the predictor worse. The " +
      "dashed line marks the AR-only baseline.",
  },
};

/**
 * The technical line under a landmark's plain headline, composed from the
 * transcribed record rather than restated. Reads e.g.
 * "Blocked event PR-AUC · +28.80% vs Trained AR · beats AR, shuffled, and random".
 */
export function landmarkTechnicalLine(landmark: LandmarkComparison): string {
  const gain = `+${landmark.baselineGainPct.toFixed(2)}%`;
  return `${landmark.metricLabel} · ${gain} vs ${landmark.baselineLabel} · ${landmark.positiveGroups}`;
}

/** The technical line under a dataset's plain headline. */
export function datasetTechnicalLine(dataset: Dataset): string {
  return dataset.statLabel;
}
