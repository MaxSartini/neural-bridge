import type { AnalysisReport } from "../api/analysisClient";
import { interpolateSeries } from "../lib/heatStrip";

/**
 * The sample report — a thirty-second advertisement, "Holiday Sale — 30s cut".
 *
 * Every figure here is invented for demonstration. `isSample: true` travels
 * with the report so the UI says so on the page; a demo that looks like a
 * measurement is the same false precision the programme refuses everywhere
 * else. When a real client lands, this file stops being reachable.
 *
 * The design supplied twelve intensity anchors. They are stored as anchors and
 * expanded to one sample per 2 Hz row, because that is the shape a real run
 * produces — and it means the heat strip is folding real rows down rather than
 * being handed its twelve cells directly.
 */

const DURATION_SECONDS = 30;
const SAMPLE_RATE_HZ = 2;

/** Relative predicted movement across the cut, 0–1. From the design. */
const INTENSITY_ANCHORS = [0.15, 0.2, 0.3, 0.25, 0.35, 0.5, 0.9, 0.6, 0.4, 0.55, 0.85, 0.7];

export const holidaySaleReport: Omit<AnalysisReport, "analysisId" | "projectName"> = {
  durationSeconds: DURATION_SECONDS,
  series: interpolateSeries(INTENSITY_ANCHORS, DURATION_SECONDS * SAMPLE_RATE_HZ),
  peaks: [
    { atSeconds: 14, note: "Top ~4% predicted movement — offer reveal" },
    { atSeconds: 21, note: "Second lift — price callout" },
    { atSeconds: 27, note: "Closing beat — logo + CTA" },
  ],
  weakMoments: [
    { atSeconds: 5, note: "Slow open — momentum builds late" },
    { atSeconds: 18, note: "Dip between beats — consider trimming" },
  ],
  // Invented. An earlier value here was a real programme measurement rounded to
  // three places, which is worse than a leak: it presents a measurement as
  // demo data, so nobody reviewing the demo would think to scrub it.
  movementRankingScore: 0.164,
  responseWindowPercentile: 12,
  isSample: true,
};

export const SAMPLE_PROJECT_NAME = "Holiday Sale — 30s cut";

/** The id the Studio landing links to, so a visitor can read a finished report
 *  without uploading anything. */
export const SAMPLE_ANALYSIS_ID = "sample";
