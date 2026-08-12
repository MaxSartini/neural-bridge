import {
  againBridgeProgression,
  againBridgeProgressionSource,
  rawFeatureFailure,
  rawFeatureFailureSource,
  type OrderedStage,
} from "../data/scorecard";

/**
 * The scroll story is told in two ACTS, and they are not on the same scale.
 *
 * `rawFeatureFailure` comes from the early blocked AGAIN benchmark; the six
 * `againBridgeProgression` stages come from the grouped benchmark. Putting an
 * AR bar of 0.203622 next to an AR bar of 0.147251 on one axis would imply the
 * two numbers are comparable. They are not. Each act therefore names its own
 * benchmark, and the axis is deliberately rebuilt at the act boundary rather
 * than tweened across it.
 */

export type StepTone = "neutral" | "critical" | "payoff";

export interface StoryStep {
  id: string;
  /** Which dataset the chart is showing. Changes exactly once, at the act break. */
  stages: OrderedStage[];
  /** Named on the chart so the scale change is explicit, never implied. */
  benchmark: string;
  /** Indices of `stages` drawn at this step; the rest render as gaps, so the
   *  category axis stays put and bars grow in place instead of re-flowing. */
  visible: number[];
  /** Index drawn in the emphasis colour, if any. */
  highlight?: number;
  tone: StepTone;
  heading: string;
  body: string;
  /** Repo-relative evidence path for this act. */
  sourceDoc: string;
}

const FAILURE = rawFeatureFailure;
const BRIDGE = againBridgeProgression;

// rawFeatureFailure is stored ascending by value (see its comment in
// data/scorecard.ts) so colour encodes magnitude rather than array position.
// Index 2 is the AR baseline, 0 is raw-cortical-only, 1 is the fusion.
const AR_BASELINE = 2;
const RAW_ONLY = 0;
const FUSION = 1;

export const storySteps: StoryStep[] = [
  {
    id: "baseline",
    stages: FAILURE,
    benchmark: "Early blocked AGAIN benchmark · event PR-AUC",
    visible: [AR_BASELINE],
    tone: "neutral",
    heading: "Start with something that already works",
    body:
      "A trained autoregressive predictor reaches 0.203622 event PR-AUC on the early blocked " +
      "benchmark. This is the number everything else has to beat.",
    sourceDoc: rawFeatureFailureSource,
  },
  {
    id: "raw-alone",
    stages: FAILURE,
    benchmark: "Early blocked AGAIN benchmark · event PR-AUC",
    visible: [RAW_ONLY, AR_BASELINE],
    highlight: RAW_ONLY,
    tone: "critical",
    heading: "The rich features are worse on their own",
    body:
      "Predicted cortical features alone manage 0.124315 — 38.95% below the AR baseline. " +
      "On its own that is unsurprising: they carry different information.",
    sourceDoc: rawFeatureFailureSource,
  },
  {
    id: "fusion-fails",
    stages: FAILURE,
    benchmark: "Early blocked AGAIN benchmark · event PR-AUC",
    visible: [RAW_ONLY, FUSION, AR_BASELINE],
    highlight: FUSION,
    tone: "critical",
    heading: "Combining them made the strong predictor worse",
    body:
      "Fusing the cortical features into the AR baseline drops it to 0.167731 — 17.63% below " +
      "where it started. Adding information subtracted performance. That failure is the whole " +
      "reason a bridge exists.",
    sourceDoc: rawFeatureFailureSource,
  },
  {
    id: "act-break",
    stages: BRIDGE,
    benchmark: "Grouped AGAIN benchmark · event PR-AUC (different scale)",
    visible: [0, 1],
    tone: "neutral",
    heading: "So the features weren't useless — they were unusable",
    body:
      "The rest of this story runs on the grouped benchmark, so the numbers restart on a new " +
      "scale and are not comparable to the bars above. Here raw cortical sits at 0.136579 and " +
      "AR alone at 0.147251.",
    sourceDoc: againBridgeProgressionSource,
  },
  {
    id: "pca",
    stages: BRIDGE,
    benchmark: "Grouped AGAIN benchmark · event PR-AUC (different scale)",
    visible: [0, 1, 2, 3],
    highlight: 3,
    tone: "neutral",
    heading: "Bridging naively barely moved it",
    body:
      "Direct fusion reaches 0.170299 and a fold-safe PCA bridge 0.171648 — a gain of about " +
      "0.001. The features are being carried across, but nothing is learning how to use them.",
    sourceDoc: againBridgeProgressionSource,
  },
  {
    id: "learned",
    stages: BRIDGE,
    benchmark: "Grouped AGAIN benchmark · event PR-AUC (different scale)",
    visible: [0, 1, 2, 3, 4],
    highlight: 4,
    tone: "payoff",
    heading: "A learned bridge jumps it",
    body:
      "A deterministic learned bridge reaches 0.230064 — a 34% step over the PCA bridge from " +
      "exactly the same upstream features. Nothing new was measured; only the way across changed.",
    sourceDoc: againBridgeProgressionSource,
  },
  {
    id: "residual",
    stages: BRIDGE,
    benchmark: "Grouped AGAIN benchmark · event PR-AUC (different scale)",
    visible: [0, 1, 2, 3, 4, 5],
    highlight: 5,
    tone: "payoff",
    heading: "And the frozen-AR residual bridge lands it",
    body:
      "0.238341 — the best of the six, and the configuration the programme carried forward. " +
      "Rich upstream features were never automatically useful. The bridge is what made them useful.",
    sourceDoc: againBridgeProgressionSource,
  },
];

/** The act boundary, used to caption the scale change in the reduced-motion fallback. */
export const ACT_TWO_START = storySteps.findIndex((s) => s.id === "act-break");
