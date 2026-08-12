import type {
  Analysis,
  AnalysisClient,
  AnalysisInput,
  AnalysisReport,
} from "./analysisClient";
import { stateAt } from "../lib/pipeline";
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
 * 2. **The file never leaves the browser.** Only its name and size are kept.
 *    The `File` itself stays with the page that read it, and the report's frame
 *    preview builds an object URL from it and revokes that URL on unmount.
 *
 * State lives in a module-level Map, so it survives client-side navigation
 * between the upload, processing and report screens but not a page reload.
 * That is the right lifetime for a demo: a reload should not resurrect a run
 * that was never really happening.
 */

interface Record {
  id: string;
  projectName: string;
  startedAt: number;
  fileName: string;
  fileSize: number;
}

const runs = new Map<string, Record>();

let counter = 0;
function nextId(): string {
  counter += 1;
  return `a${Date.now().toString(36)}${counter}`;
}

export class MockAnalysisClient implements AnalysisClient {
  async createAnalysis(input: AnalysisInput): Promise<{ id: string }> {
    const id = nextId();
    runs.set(id, {
      id,
      projectName: input.projectName.trim() || "Untitled analysis",
      startedAt: Date.now(),
      fileName: input.file.name,
      fileSize: input.file.size,
    });
    return { id };
  }

  async getAnalysis(id: string): Promise<Analysis> {
    // The sample is always finished — it is there to be read, not watched.
    if (id === SAMPLE_ANALYSIS_ID) {
      const finished = stateAt(Number.MAX_SAFE_INTEGER);
      return { id, projectName: SAMPLE_PROJECT_NAME, ...finished };
    }

    const run = runs.get(id);
    if (!run) throw new Error(`No analysis ${id}`);
    return { id, projectName: run.projectName, ...stateAt(Date.now() - run.startedAt) };
  }

  async getReport(id: string): Promise<AnalysisReport> {
    const projectName =
      id === SAMPLE_ANALYSIS_ID ? SAMPLE_PROJECT_NAME : (runs.get(id)?.projectName ?? "Analysis");
    // Every run returns the same fixture. That is the whole extent of the mock:
    // the shape is real, the numbers are not, and `isSample` says so on screen.
    return { analysisId: id, projectName, ...holidaySaleReport };
  }
}

/** The client the Studio screens use. Swap this line for a real implementation. */
export const analysisClient: AnalysisClient = new MockAnalysisClient();
