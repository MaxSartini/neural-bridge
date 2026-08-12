/**
 * The seam.
 *
 * Screens import from here — never from an implementation module. That is the
 * whole point: before this existed, `analysisClient` was exported from
 * `mockAnalysisClient.ts` and all three screens imported it by that name, so
 * ADR-0003's claim that swapping the mock is "a one-module change that touches
 * no screen" was false. Either you edited a file called `mockAnalysisClient`
 * and left it serving production, or you edited every screen.
 *
 * Now the swap is the one line below.
 */
export type {
  Analysis,
  AnalysisClient,
  AnalysisInput,
  AnalysisReport,
  AnalysisStatus,
  ContentType,
  Moment,
  Objective,
  PipelineStep,
  PipelineStepId,
  StepState,
} from "./analysisClient";

export { AnalysisNotFound } from "./analysisClient";

import type { AnalysisClient } from "./analysisClient";
import { MockAnalysisClient } from "./mockAnalysisClient";

/**
 * The client every screen uses. Replace this one line with an HTTP client when
 * a real inference service exists; nothing else changes.
 *
 * Note what a real client must still honour, because it is part of the
 * interface and not of this implementation: reject with `AnalysisNotFound` for
 * an unknown id, omit `PipelineStep.progress` for a step that cannot measure
 * itself, and omit `etaSeconds` when there is no honest estimate.
 */
export const analysisClient: AnalysisClient = new MockAnalysisClient();
