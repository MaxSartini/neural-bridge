import type { Analysis, PipelineStep, PipelineStepId } from "../api/analysisClient";

/**
 * The pipeline as a pure function of elapsed time.
 *
 * Keeping this a reducer rather than a bag of timers is what makes the
 * processing screen testable without a clock or a DOM: `stateAt(3_000)` is a
 * plain assertion. The mock client owns the clock; this module owns the rules.
 */

interface StepPlan {
  id: PipelineStepId;
  title: string;
  durationMs: number;
  /**
   * Whether the step can report its own progress. Only the bridge can — the
   * others are opaque calls that either have returned or have not. This is the
   * honesty rule from the design encoded as data: a step that cannot measure
   * itself must not render a progress bar.
   */
  measurable: boolean;
}

/**
 * Durations are **demonstration timings**, not a claim about what real
 * inference costs. They are short enough that someone meeting the product can
 * watch a run finish, and their relative weights follow the design's ordering.
 * A client fronting a real service takes its step states from that service and
 * ignores this table entirely.
 */
/* Step titles name what the stage does, never what it runs on. The upstream
   encoder and response model are the most valuable thing a competitor could
   read off this screen and the cheapest thing to withhold — see ADR-0006. */
export const PIPELINE: readonly StepPlan[] = [
  { id: "encoding", title: "Encoding video", durationMs: 4_000, measurable: false },
  {
    id: "predicting",
    title: "Estimating viewer response",
    durationMs: 5_000,
    measurable: false,
  },
  {
    id: "bridging",
    title: "Running the Neural Bridge temporal model",
    durationMs: 5_500,
    measurable: true,
  },
  { id: "mapping", title: "Generating the response map", durationMs: 2_500, measurable: false },
];

export const TOTAL_DURATION_MS = PIPELINE.reduce((sum, s) => sum + s.durationMs, 0);

/**
 * Resolve the whole run at `elapsedMs`. Clamped at both ends, so a negative
 * clock reads as "not started" and anything past the end reads as complete
 * rather than overflowing past 100%.
 */
export function stateAt(elapsedMs: number): Pick<
  Analysis,
  "status" | "steps" | "overallProgress" | "etaSeconds"
> {
  const elapsed = Number.isFinite(elapsedMs) ? Math.max(0, elapsedMs) : 0;
  const done = elapsed >= TOTAL_DURATION_MS;

  let consumed = 0;
  const steps: PipelineStep[] = PIPELINE.map((plan) => {
    const start = consumed;
    consumed += plan.durationMs;

    if (elapsed >= start + plan.durationMs) {
      return { id: plan.id, title: plan.title, state: "done" };
    }
    if (elapsed < start) {
      return { id: plan.id, title: plan.title, state: "queued" };
    }
    const step: PipelineStep = { id: plan.id, title: plan.title, state: "active" };
    // Only a measurable step carries a fraction. Everything else stays
    // undefined so the UI has nothing to draw a fake bar from.
    if (plan.measurable) {
      step.progress = (elapsed - start) / plan.durationMs;
    }
    return step;
  });

  const overallProgress = Math.min(1, elapsed / TOTAL_DURATION_MS);
  const remainingMs = Math.max(0, TOTAL_DURATION_MS - elapsed);

  return {
    // Never "queued": the mock's clock starts the moment the analysis is
    // created, so its first step is already active at t=0. The status stays in
    // the type for a real client, where work can genuinely sit behind a queue.
    status: done ? "complete" : "running",
    steps,
    overallProgress,
    // The mock knows its own schedule, so an estimate here is real rather than
    // guessed. A client fronting an actual service should omit this unless the
    // service reports it.
    etaSeconds: done ? undefined : Math.ceil(remainingMs / 1000),
  };
}
