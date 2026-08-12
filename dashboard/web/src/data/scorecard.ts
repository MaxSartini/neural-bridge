/**
 * Hand-curated snapshot of results/README.md's scorecard tables — not a live read of that
 * file or of the underlying result.json evidence. If a new phase concludes and
 * results/README.md's numbers change, this file needs a matching manual update or the
 * dashboard will silently show stale figures (see Phase A2 plan, "Known limitation").
 * Every entry carries a sourceDoc so a reader can always jump to the primary evidence.
 */

export interface LandmarkComparison {
  id: string;
  romanNumeral: string;
  title: string;
  metricLabel: string;
  real: number;
  baseline: number;
  baselineLabel: string;
  baselineGainPct: number;
  control: number;
  controlLabel: string;
  controlGainPct: number;
  positiveGroups: string;
  sourceDoc: string;
}

export const landmarks: LandmarkComparison[] = [
  {
    id: "veatic",
    romanNumeral: "I",
    title: "Original VEATIC event/spike breakthrough",
    metricLabel: "Blocked event PR-AUC",
    real: 0.2536,
    baseline: 0.1969,
    baselineLabel: "Trained AR",
    baselineGainPct: 28.8,
    control: 0.184,
    controlLabel: "Shuffled features",
    controlGainPct: 37.83,
    positiveGroups: "beats AR, shuffled, and random",
    sourceDoc: "studies/original-veatic/v2-closure/README.md",
  },
  {
    id: "again-event",
    romanNumeral: "II",
    title: "AGAIN event/spike reconstruction",
    metricLabel: "Grouped event PR-AUC",
    real: 0.234368,
    baseline: 0.21805,
    baselineLabel: "Frozen AR",
    baselineGainPct: 7.48,
    control: 0.217972,
    controlLabel: "Best matched control",
    controlGainPct: 7.52,
    positiveGroups: "15/15",
    sourceDoc: "studies/again/phase-06-event-stabilization/README.md",
  },
  {
    id: "phase7-continuous",
    romanNumeral: "III",
    title: "Continuous response intelligence (Phase 7)",
    metricLabel: "Future-movement Spearman",
    real: 0.260301,
    baseline: 0.240537,
    baselineLabel: "Frozen AR",
    baselineGainPct: 8.22,
    control: 0.240252,
    controlLabel: "Best matched control",
    controlGainPct: 8.34,
    positiveGroups: "15/15",
    sourceDoc: "studies/again/phase-07-continuous/evidence-summary.md",
  },
  {
    id: "zero-label",
    romanNumeral: "IV",
    title: "Video-only inference (zero-label, locked)",
    metricLabel: "Future-movement ranking",
    real: 0.178513,
    baseline: 0.100488,
    baselineLabel: "Strongest false-signal control",
    baselineGainPct: 77.65,
    control: 0.100488,
    controlLabel: "Strongest false-signal control",
    controlGainPct: 77.65,
    positiveGroups: "5/5",
    sourceDoc: "studies/again/zero-label/locked-confirmation/README.md",
  },
];

/** A corpus the programme trains and evaluates on, as reported in README.md's
 * dataset section. Same hand-transcription caveat as the landmarks above. */
export interface Dataset {
  id: string;
  /** Kicker above the card — what kind of footage this is. */
  category: string;
  title: string;
  /** The headline count, rendered as the card's numeral. */
  stat: string;
  /** What the numeral counts, plus the secondary figures. */
  statLabel: string;
  sourceDoc: string;
}

export const datasets: Dataset[] = [
  {
    id: "again",
    category: "Gameplay",
    title: "AGAIN",
    stat: "995",
    statLabel: "videos · 243,575 aligned 2Hz rows · ~33.8h",
    sourceDoc: "README.md",
  },
  {
    id: "veatic",
    category: "Film · TV · Documentary",
    title: "VEATIC 2.1",
    stat: "124",
    statLabel: "videos · 257,601 frames · 192 annotators",
    sourceDoc: "README.md",
  },
];

export interface OrderedStage {
  label: string;
  fullLabel: string;
  value: number;
}

export const againBridgeProgression: OrderedStage[] = [
  { label: "Raw cortical", fullLabel: "Raw cortical only", value: 0.136579 },
  { label: "AR only", fullLabel: "Trained AR only", value: 0.147251 },
  { label: "AR + raw", fullLabel: "Direct AR + raw cortical", value: 0.170299 },
  { label: "PCA bridge", fullLabel: "Fold-safe PCA bridge", value: 0.171648 },
  { label: "Learned bridge", fullLabel: "Deterministic learned bridge", value: 0.230064 },
  { label: "Residual bridge", fullLabel: "Frozen-AR residual bridge", value: 0.238341 },
];

export const againBridgeProgressionSource = "studies/again/phase-05-learned-bridge/README.md";

/** Ascending by value (not narrative order) so the sequential ramp's light→dark index
 * mapping stays honest — color encodes magnitude, never array position. */
export const rawFeatureFailure: OrderedStage[] = [
  { label: "Raw cortical only", fullLabel: "Raw predicted cortical features", value: 0.124315 },
  { label: "AR + raw cortical", fullLabel: "AR + raw cortical features", value: 0.167731 },
  { label: "AR baseline", fullLabel: "Strong trained AR baseline", value: 0.203622 },
];

export const rawFeatureFailureSource = "studies/again/phase-03-raw-cortical/README.md";
