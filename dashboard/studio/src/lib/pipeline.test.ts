import { describe, expect, it } from "vitest";
import { PIPELINE, TOTAL_DURATION_MS, stateAt } from "./pipeline";

const idsWithState = (elapsed: number) =>
  stateAt(elapsed).steps.map((s) => [s.id, s.state] as const);

describe("stateAt", () => {
  it("starts with the first step already active and the rest queued", () => {
    const s = stateAt(0);
    // The clock starts when the analysis is created, so there is no moment
    // where the run exists but nothing is happening.
    expect(s.status).toBe("running");
    expect(idsWithState(0)).toEqual([
      ["encoding", "active"],
      ["predicting", "queued"],
      ["bridging", "queued"],
      ["mapping", "queued"],
    ]);
    expect(s.overallProgress).toBe(0);
  });

  it("ends with everything done and no estimate left to give", () => {
    const s = stateAt(TOTAL_DURATION_MS);
    expect(s.status).toBe("complete");
    expect(s.steps.every((step) => step.state === "done")).toBe(true);
    expect(s.overallProgress).toBe(1);
    expect(s.etaSeconds).toBeUndefined();
  });

  it("holds at complete past the end rather than overflowing", () => {
    const s = stateAt(TOTAL_DURATION_MS * 10);
    expect(s.status).toBe("complete");
    expect(s.overallProgress).toBe(1);
  });

  it("treats a negative or non-finite clock as not started", () => {
    for (const bad of [-5_000, Number.NaN, Number.POSITIVE_INFINITY * 0]) {
      expect(stateAt(bad).overallProgress).toBe(0);
    }
  });

  it("advances one step at a time, leaving earlier steps done", () => {
    // Land inside step three: one and two done, three active, four queued.
    const intoThird = PIPELINE[0].durationMs + PIPELINE[1].durationMs + 1_000;
    expect(idsWithState(intoThird)).toEqual([
      ["encoding", "done"],
      ["predicting", "done"],
      ["bridging", "active"],
      ["mapping", "queued"],
    ]);
  });

  it("never has more than one active step", () => {
    for (let t = 0; t <= TOTAL_DURATION_MS; t += 500) {
      const active = stateAt(t).steps.filter((s) => s.state === "active");
      expect(active.length).toBeLessThanOrEqual(1);
    }
  });

  it("never moves a step backwards as time advances", () => {
    const rank = { queued: 0, active: 1, done: 2 } as const;
    let previous = stateAt(0).steps.map((s) => rank[s.state]);
    for (let t = 250; t <= TOTAL_DURATION_MS; t += 250) {
      const current = stateAt(t).steps.map((s) => rank[s.state]);
      current.forEach((v, i) => expect(v).toBeGreaterThanOrEqual(previous[i]));
      previous = current;
    }
  });
});

describe("progress is only reported where it can be measured", () => {
  // The design's honesty rule, encoded: a step that cannot measure itself must
  // give the UI nothing to draw a bar from. Relaxing this is how a fake
  // percentage gets shipped.
  it("gives an active measurable step a fraction", () => {
    // Halfway through the bridge, expressed relative to the table so retuning
    // the demo timings cannot silently move this out of the step it targets.
    const intoBridge =
      PIPELINE[0].durationMs + PIPELINE[1].durationMs + PIPELINE[2].durationMs / 2;
    const bridging = stateAt(intoBridge).steps.find((s) => s.id === "bridging");
    expect(bridging?.state).toBe("active");
    expect(bridging?.progress).toBeGreaterThan(0);
    expect(bridging?.progress).toBeLessThan(1);
  });

  it("leaves an active unmeasurable step's progress undefined", () => {
    const encoding = stateAt(1_000).steps.find((s) => s.id === "encoding");
    expect(encoding?.state).toBe("active");
    expect(encoding?.progress).toBeUndefined();
  });

  it("never reports progress on a queued or done step", () => {
    for (let t = 0; t <= TOTAL_DURATION_MS; t += 1_000) {
      for (const step of stateAt(t).steps) {
        if (step.state !== "active") expect(step.progress).toBeUndefined();
      }
    }
  });
});

describe("overall progress and estimate", () => {
  it("increases monotonically", () => {
    let last = -1;
    for (let t = 0; t <= TOTAL_DURATION_MS; t += 1_000) {
      const p = stateAt(t).overallProgress;
      expect(p).toBeGreaterThanOrEqual(last);
      last = p;
    }
  });

  it("counts down the estimate as the run proceeds", () => {
    const early = stateAt(5_000).etaSeconds ?? 0;
    const late = stateAt(TOTAL_DURATION_MS - 5_000).etaSeconds ?? 0;
    expect(early).toBeGreaterThan(late);
  });
});
