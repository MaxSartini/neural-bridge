import {
  AnalysisNotFound,
  type Analysis,
  type AnalysisClient,
  type AnalysisInput,
  type AnalysisReport,
} from "./analysisClient";
import { PIPELINE, stateAt } from "../lib/pipeline";
import {
  SAMPLE_ANALYSIS_ID,
  SAMPLE_PROJECT_NAME,
  holidaySaleReport,
} from "../fixtures/holidaySale";

/**
 * In-memory implementation of `AnalysisClient`.
 *
 * Two properties matter and are load-bearing, not incidental:
 *
 * 1. **No network.** It issues no `fetch`, so the Studio satisfies the deployed
 *    CSP's `connect-src 'none'` unchanged and needs no new ADR to ship as a
 *    static demo.
 * 2. **The file never leaves the browser.** Nothing about it is stored here at
 *    all; the page that read it keeps it in `lib/localVideo`, and the report's
 *    frame preview builds an object URL from it and revokes that on unmount.
 *
 * State lives in a module-level Map, so it survives client-side navigation
 * between the upload, processing and report screens but not a page reload.
 * That is the right lifetime for a demo: a reload should not resurrect a run
 * that was never really happening.
 */

interface Run {
  projectName: string;
  /** Milliseconds since the epoch, or `null` for a run that is already done. */
  startedAt: number | null;
}

const TOTAL_MS = PIPELINE.reduce((sum, s) => sum + s.durationMs, 0);

/**
 * The sample is seeded as an ordinary finished run rather than special-cased in
 * every method. It used to be a branch in both `getAnalysis` and `getReport`,
 * expressing "finished" as `stateAt(Number.MAX_SAFE_INTEGER)` — abusing a clamp
 * to mean completion. As data it needs no branch at all, and `SAMPLE_ANALYSIS_ID`
 * stops being a value three modules have to agree about.
 */
const runs = new Map<string, Run>([
  [SAMPLE_ANALYSIS_ID, { projectName: SAMPLE_PROJECT_NAME, startedAt: null }],
]);

let counter = 0;
function nextId(): string {
  counter += 1;
  return `a${Date.now().toString(36)}${counter}`;
}

/** Milliseconds of pipeline elapsed for a run — a finished run is simply past the end. */
function elapsedFor(run: Run): number {
  return run.startedAt === null ? TOTAL_MS : Date.now() - run.startedAt;
}

export class MockAnalysisClient implements AnalysisClient {
  async createAnalysis(input: AnalysisInput): Promise<{ id: string }> {
    const id = nextId();
    runs.set(id, {
      projectName: input.projectName.trim() || "Untitled analysis",
      startedAt: Date.now(),
    });
    return { id };
  }

  async getAnalysis(id: string): Promise<Analysis> {
    const run = runs.get(id);
    if (!run) throw new AnalysisNotFound(id);
    return { id, projectName: run.projectName, ...stateAt(elapsedFor(run)) };
  }

  async getReport(id: string): Promise<AnalysisReport> {
    const run = runs.get(id);
    if (!run) throw new AnalysisNotFound(id);
    // Every run returns the same fixture. That is the whole extent of the mock:
    // the shape is real, the numbers are not, and `isSample` says so on screen.
    return { analysisId: id, projectName: run.projectName, ...holidaySaleReport };
  }
}
