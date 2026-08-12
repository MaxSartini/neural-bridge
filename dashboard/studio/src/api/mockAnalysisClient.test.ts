import { describe, expect, it } from "vitest";
import { MockAnalysisClient } from "./mockAnalysisClient";
import { AnalysisNotFound, type AnalysisInput } from "./analysisClient";
import { SAMPLE_ANALYSIS_ID } from "../fixtures/holidaySale";

/**
 * These tests exist because the seam moved.
 *
 * Before, the client was reachable only as a singleton exported from this
 * implementation module, so there was nothing to construct and nothing to
 * assert against — and `getReport` shipped for weeks returning a complete,
 * plausible report for an id that had never existed. The interface is the test
 * surface; now that it is one, that class of bug is catchable.
 */

/** Minimal stand-in — the client stores nothing about the file but its name. */
function inputWith(projectName: string): AnalysisInput {
  return {
    projectName,
    contentType: "advertisement",
    objective: "peaks",
    notes: "",
    file: { name: "cut.mp4", size: 1024, type: "video/mp4" } as File,
  };
}

describe("unknown ids reject rather than inventing a run", () => {
  it("getAnalysis rejects with AnalysisNotFound", async () => {
    const client = new MockAnalysisClient();
    await expect(client.getAnalysis("never-existed")).rejects.toBeInstanceOf(AnalysisNotFound);
  });

  // The live defect: /analyses/anything/report used to render a full report.
  it("getReport rejects with AnalysisNotFound", async () => {
    const client = new MockAnalysisClient();
    await expect(client.getReport("never-existed")).rejects.toBeInstanceOf(AnalysisNotFound);
  });

  it("names the id in the message, so a log says which one", async () => {
    const client = new MockAnalysisClient();
    await expect(client.getReport("abc123")).rejects.toThrow(/abc123/);
  });
});

describe("a created analysis is retrievable through the interface", () => {
  it("resolves the id it just handed out", async () => {
    const client = new MockAnalysisClient();
    const { id } = await client.createAnalysis(inputWith("Holiday Sale"));
    const analysis = await client.getAnalysis(id);
    expect(analysis.id).toBe(id);
    expect(analysis.projectName).toBe("Holiday Sale");
    expect(analysis.status).toBe("running");
  });

  it("falls back to a placeholder name rather than an empty heading", async () => {
    const client = new MockAnalysisClient();
    const { id } = await client.createAnalysis(inputWith("   "));
    expect((await client.getAnalysis(id)).projectName).toBe("Untitled analysis");
  });

  it("hands out distinct ids", async () => {
    const client = new MockAnalysisClient();
    const a = await client.createAnalysis(inputWith("one"));
    const b = await client.createAnalysis(inputWith("two"));
    expect(a.id).not.toBe(b.id);
  });
});

describe("the sample is seeded data, not a special case", () => {
  it("is already complete, with every step done", async () => {
    const client = new MockAnalysisClient();
    const analysis = await client.getAnalysis(SAMPLE_ANALYSIS_ID);
    expect(analysis.status).toBe("complete");
    expect(analysis.overallProgress).toBe(1);
    expect(analysis.steps.every((s) => s.state === "done")).toBe(true);
  });

  it("offers no estimate, because there is nothing left to estimate", async () => {
    const client = new MockAnalysisClient();
    expect((await client.getAnalysis(SAMPLE_ANALYSIS_ID)).etaSeconds).toBeUndefined();
  });

  it("returns a report that labels itself as sample data", async () => {
    const client = new MockAnalysisClient();
    const report = await client.getReport(SAMPLE_ANALYSIS_ID);
    expect(report.isSample).toBe(true);
    expect(report.analysisId).toBe(SAMPLE_ANALYSIS_ID);
    expect(report.series.length).toBeGreaterThan(0);
  });
});
