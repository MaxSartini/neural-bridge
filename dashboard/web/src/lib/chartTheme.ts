/**
 * Chart colors are CSS variable *references*, not resolved hex strings.
 *
 * Recharts passes `fill` straight through to the SVG, so handing it
 * `var(--viz-seq-3)` means the browser re-resolves the color when the theme
 * changes — the bars re-color on a theme flip with no JS listener and no
 * re-render. Resolving to hex here (what this module used to do) would freeze
 * the palette at mount time.
 *
 * This survives the move to `light-dark()`: the tokens now hold a pair rather
 * than a single value, and a presentation attribute still resolves them.
 * Verified by reading the painted `fill` off a bar while flipping the pin —
 * same DOM node, both colors.
 *
 * The ramp is a single hue, monotone in lightness, with a different range per
 * theme. Every step clears the 3:1 mark floor against its own ground; see the
 * measured ranges recorded beside the tokens in `global.css`.
 */
const RAMP_STEPS = 6;

/** Returns `count` sequential steps, faint→strong, for an ordinal bar ranking. */
export function sequentialRamp(count: number): string[] {
  if (count <= 1) return ["var(--viz-seq-4)"];
  const steps: string[] = [];
  for (let i = 0; i < count; i += 1) {
    // Spread the requested count evenly across the 6 available ramp steps, so a
    // 3-bar chart uses the extremes and the middle rather than only the top.
    const idx = Math.round((i * (RAMP_STEPS - 1)) / (count - 1)) + 1;
    steps.push(`var(--viz-seq-${idx})`);
  }
  return steps;
}

/**
 * Category bars are wide in this design — the mocks draw them as solid blocks
 * nearly filling their band, not the thin columns the previous theme used.
 */
export const BAR_MAX_SIZE = 120;

/** The horizontal gain bars are a fixed 8px rule, per the design. */
export const GAIN_BAR_SIZE = 8;

/** Square by design: nothing in this system is rounded. */
export const BAR_RADIUS: [number, number, number, number] = [0, 0, 0, 0];
