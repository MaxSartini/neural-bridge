import { describe, expect, it } from "vitest";
import { datasets, landmarks } from "./scorecard";
import {
  chartGlosses,
  datasetPlain,
  datasetTechnicalLine,
  heroGloss,
  landmarkPlain,
  landmarkTechnicalLine,
} from "./plainLanguage";

/**
 * Every plain-half string in the layer, labelled so a failure names the gloss
 * rather than an array index.
 */
const everyPlainGloss: ReadonlyArray<readonly [string, string]> = [
  ["hero", heroGloss.plain],
  ...Object.entries(landmarkPlain).map(([k, v]) => [`landmark:${k}`, v] as const),
  ...Object.entries(datasetPlain).map(([k, v]) => [`dataset:${k}`, v] as const),
  ...Object.entries(chartGlosses).map(([k, v]) => [`chart:${k}`, v.plain] as const),
];

describe("plain glosses carry no digits", () => {
  // The one mechanical guard on the whole plain-language layer. A figure lifted
  // out of its metric label reads as a much broader claim than the evidence
  // supports: "+28.80%" means a gain in blocked event PR-AUC over a trained AR
  // baseline, not that the product is 28.8% better at anything a reader might
  // have in mind. If a sentence needs a quantity, spell it out ("fifteen") or
  // move it to the technical half — do not relax this test.
  it.each(everyPlainGloss)("%s", (_label, text) => {
    expect(text).not.toMatch(/\d/);
  });
});

describe("coverage", () => {
  it("glosses every landmark in the scorecard", () => {
    const missing = landmarks.filter((l) => !landmarkPlain[l.id]).map((l) => l.id);
    expect(missing).toEqual([]);
  });

  it("glosses every dataset in the scorecard", () => {
    const missing = datasets.filter((d) => !datasetPlain[d.id]).map((d) => d.id);
    expect(missing).toEqual([]);
  });

  it("has no gloss for a landmark or dataset that no longer exists", () => {
    const knownIds = new Set([...landmarks.map((l) => l.id), ...datasets.map((d) => d.id)]);
    const orphans = [...Object.keys(landmarkPlain), ...Object.keys(datasetPlain)].filter(
      (id) => !knownIds.has(id),
    );
    expect(orphans).toEqual([]);
  });

  it("gives every gloss a non-empty plain half", () => {
    for (const [label, text] of everyPlainGloss) {
      expect(text.trim(), label).not.toBe("");
    }
  });
});

describe("technical lines are composed from the transcribed record", () => {
  // Composition, not transcription: ADR-0002 already carries a manual-sync risk
  // on these figures, and restating them here would double it. These assertions
  // pin the composition, so a change to scorecard.ts flows through rather than
  // silently disagreeing with a hardcoded copy.
  it("names the metric, the gain, the baseline and the group count", () => {
    const veatic = landmarks.find((l) => l.id === "veatic");
    expect(veatic).toBeDefined();
    expect(landmarkTechnicalLine(veatic!)).toBe(
      "Blocked event PR-AUC · +28.80% vs Trained AR · beats AR, shuffled, and random",
    );
  });

  it("formats the gain to two decimal places", () => {
    for (const landmark of landmarks) {
      expect(landmarkTechnicalLine(landmark)).toContain(
        `+${landmark.baselineGainPct.toFixed(2)}%`,
      );
    }
  });

  it("carries the dataset's secondary figures", () => {
    const again = datasets.find((d) => d.id === "again");
    expect(again).toBeDefined();
    expect(datasetTechnicalLine(again!)).toBe("videos · 243,575 aligned 2Hz rows · ~33.8h");
  });
});
