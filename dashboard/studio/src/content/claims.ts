/**
 * Everything Neural Bridge says to the outside world, in one file.
 *
 * This package is the customer- and investor-facing surface. It is deployed to
 * people who are not under NDA and who may work for, advise, or later fund a
 * competitor. So it operates under a **disclosure line**, and this module is
 * where that line is held.
 *
 * ## What may appear here
 *
 * Outcomes, and validated figures framed plainly. "Ranks attention shifts 78%
 * better than the strongest alternative we tested, across 299 videos it had
 * never seen" is fine: it is checkable, it is the claim an investor needs, and
 * it teaches a competitor nothing reproducible.
 *
 * ## What must never appear here
 *
 * - **Upstream model names.** Not the encoder, not the response model, not any
 *   version string. This is the single most valuable thing on the page to a
 *   competitor and the cheapest thing for us to withhold.
 * - **Metric names** — PR-AUC, Spearman, ranking-score names.
 * - **Dataset names** and corpus identities.
 * - **Baseline and control vocabulary** — frozen/trained AR, shuffled features,
 *   matched or false-signal controls, held-out folds, positive groups.
 * - **Internal repo paths** or study/phase identifiers.
 * - **Raw absolute metric values.** A percentage gain is a claim; the
 *   underlying score is a measurement someone can reverse-engineer a method
 *   from.
 *
 * `claims.test.ts` enforces the whole list mechanically, and
 * `scripts/verify-bundle.mjs` re-checks the built output — because the
 * regression that prompted this file was a single import, not a bad sentence.
 *
 * ## Two further rules, inherited
 *
 * - **A `plain` sentence carries no digits.** Same rule, same reason as the
 *   internal plain-gloss layer: a figure lifted away from what it measures
 *   reads as a much broader claim than the evidence supports. Put the number in
 *   `figure`, where `context` says what it is.
 * - **Stay inside the Honest boundaries** — the root README's list of what
 *   Neural Bridge does not claim. Simplifying is not licence to overclaim.
 */

export interface Claim {
  id: string;
  /** Leads the card. Never contains a digit. */
  plain: string;
  /** The validated figure, as displayed. */
  figure: string;
  /** What the figure is, in plain words. No metric or control vocabulary. */
  context: string;
}

export const heroClaim = {
  /** The one-sentence answer to "what is this?". */
  plain: "Find the moments your video earns attention — before you test it on people.",
  supporting:
    "Upload a cut and get back a map of where viewer response is most likely to move, " +
    "where it sags, and which sections are worth putting in front of a real audience. " +
    "No audience data needed.",
};

export const validatedClaims: Claim[] = [
  {
    id: "spikes",
    plain: "Finds the moments that spike, better than predicting from what came before.",
    figure: "+28.80%",
    context: "better at picking out reaction moments than the strongest alternative we tested",
  },
  {
    id: "repeatable",
    plain: "Holds up on a second, unrelated video set — in every one of fifteen checks.",
    figure: "+7.48%",
    context: "better on an independent set of videos, positive in every check we ran",
  },
  {
    id: "timeline",
    plain: "Ranks the whole timeline, not just the spikes.",
    figure: "+8.22%",
    context: "better at ordering a full clip from most to least likely to move response",
  },
  {
    id: "unseen",
    plain: "Works on a video it has never seen, with no audience data at all.",
    figure: "+77.65%",
    context:
      "better at ranking attention shifts than the strongest alternative we tested, " +
      "across 299 videos held back and never used to build the system",
  },
];

/** What lands in the customer's hands. The Home page's answer to "and then?". */
export const whatYouGet = [
  {
    id: "peaks",
    title: "Peaks",
    body: "The moments most likely to hold attention, ranked, with a timestamp you can jump to.",
  },
  {
    id: "weak",
    title: "Weak moments",
    body: "Where the cut sags — the stretches worth trimming, reordering, or reshooting.",
  },
  {
    id: "shortlist",
    title: "A shortlist to test",
    body: "The sections most worth putting in front of real people, so a full audience study spends its budget where it matters.",
  },
];

/** The pipeline, described by what it does rather than what it runs on. */
export const howItWorks = [
  {
    step: "01",
    title: "You upload a cut",
    body: "An advertisement, a trailer, a gameplay capture. No audience data, no prior test results, no response history.",
  },
  {
    step: "02",
    title: "We estimate the viewer response",
    body: "The system reads the video frame by frame and estimates how a typical viewer's attention responds as it plays.",
  },
  {
    step: "03",
    title: "The bridge turns that into a forecast",
    body: "A raw estimate is not useful on its own. The bridge converts it into a ranking of where response is most likely to move next.",
  },
  {
    step: "04",
    title: "You get a response map",
    body: "Likely peaks, weak moments, and the sections most worth testing with real people before you spend on a full audience study.",
  },
];

/** Straight from the root README's "Honest boundaries", in the same order. */
export const notClaimed = [
  "reading minds",
  "profiling individual people",
  "medical or diagnostic inference",
  "universal emotion recognition",
  "an exact reaction at a given second",
  "guaranteed audience or commercial outcomes",
];

export const boundariesIntro =
  "Neural Bridge surfaces where viewer response is likely to move, and by how much relative " +
  "to the rest of a cut. It is not doing any of the following, and every result above is " +
  "measured against deliberately misleading inputs so that stays checkable:";

export const boundariesOutro =
  "Those boundaries are what make the result defensible. They do not make it small.";

/** The report's credibility block. Composed, so the figure has one owner. */
export const reportMethodology = {
  kicker: "Why you can trust this",
  body: (figure: string) =>
    `This ranking runs on a system validated to rank attention shifts ${figure} better than ` +
    `the strongest alternative we tested, across 299 videos it had never seen — with no ` +
    `audience data needed to run it.`,
  /** The claim the figure above comes from. */
  figure: validatedClaims.find((c) => c.id === "unseen")?.figure ?? "",
};

export const confidence = {
  kicker: "Confidence",
  title: "Calibrated, not overpromised",
  body:
    "Ranking and lift are validated on video the system never saw during development, " +
    "measured against deliberately misleading inputs. Neural Bridge does not claim an exact " +
    "reaction at any single second — it surfaces where response is most likely to move, and " +
    "by how much relative to the rest of the cut.",
};
