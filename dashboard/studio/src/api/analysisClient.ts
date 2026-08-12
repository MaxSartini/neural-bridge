/**
 * The seam between the Studio UI and whatever eventually runs inference.
 *
 * ADR-0001 settled that the evidence server is read-only and that inference
 * "should be a separate service, and this dashboard should not become the place
 * inference lives". This interface is how the client product gets built without
 * reopening that: the UI is real, and the only thing that is mocked sits behind
 * these three methods. Swapping `MockAnalysisClient` for an HTTP client that
 * talks to a real service is a one-module change and touches no screen.
 *
 * The mock is deliberately in-process — it issues no `fetch`, and the selected
 * File never leaves the browser. That is what lets the Studio ship as a static
 * bundle under the existing `connect-src 'none'` CSP without a new ADR.
 */

export type ContentType = "advertisement" | "trailer" | "gameplay";

/** What the customer wants out of the run. Shapes the report's emphasis. */
export type Objective = "peaks" | "weak" | "full";

export interface AnalysisInput {
  projectName: string;
  contentType: ContentType;
  objective: Objective;
  notes: string;
  file: File;
}

export type PipelineStepId = "encoding" | "predicting" | "bridging" | "mapping";

export type StepState = "queued" | "active" | "done";

export interface PipelineStep {
  id: PipelineStepId;
  title: string;
  state: StepState;
  /**
   * Fraction complete, 0–1, and **only** present when the step can actually
   * measure itself. `undefined` means "running, no measurable progress" — the
   * UI must show no sub-bar rather than invent one. A real client should leave
   * this undefined unless the service genuinely reports it.
   */
  progress?: number;
}

export type AnalysisStatus = "queued" | "running" | "complete" | "failed";

export interface Analysis {
  id: string;
  projectName: string;
  status: AnalysisStatus;
  steps: PipelineStep[];
  /** Fraction complete across the whole run, 0–1. */
  overallProgress: number;
  /**
   * Seconds left, or `undefined` when no honest estimate exists. The design is
   * explicit that the line may be omitted rather than guessed.
   */
  etaSeconds?: number;
}

/** A moment worth looking at, located in the cut. */
export interface Moment {
  atSeconds: number;
  note: string;
}

export interface AnalysisReport {
  analysisId: string;
  projectName: string;
  durationSeconds: number;
  /**
   * Predicted response movement, one sample per 2 Hz row, normalised to 0–1.
   * Relative within this cut — not an absolute reaction strength.
   */
  series: number[];
  peaks: Moment[];
  weakMoments: Moment[];
  movementRankingScore: number;
  /** Where this cut sits in the predicted response window, as a percentile. */
  responseWindowPercentile: number;
  /**
   * True while these figures come from a fixture rather than a model run. The
   * report labels itself when this is set — showing sample numbers as if they
   * were measured is exactly the false precision the programme refuses.
   */
  isSample: boolean;
}

export interface AnalysisClient {
  createAnalysis(input: AnalysisInput): Promise<{ id: string }>;
  getAnalysis(id: string): Promise<Analysis>;
  getReport(id: string): Promise<AnalysisReport>;
}
