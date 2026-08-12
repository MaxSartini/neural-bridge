/**
 * Fold a 2 Hz prediction series down to the report's heat strip.
 *
 * The strip is a fixed number of cells regardless of clip length, so each cell
 * is the *mean* of the samples that land in it — not a sample taken every Nth
 * row. Averaging matters: picking every Nth row would let a one-sample spike
 * decide a cell's colour, or miss it entirely, and the strip is meant to read
 * as "how much movement is in this stretch of the cut".
 */
export const HEAT_STRIP_CELLS = 12;

export function bucketSeries(series: number[], cellCount = HEAT_STRIP_CELLS): number[] {
  if (cellCount <= 0) return [];
  if (series.length === 0) return new Array<number>(cellCount).fill(0);

  const cells: number[] = [];
  for (let i = 0; i < cellCount; i += 1) {
    // Proportional split, so a series shorter than cellCount repeats samples
    // across cells rather than leaving holes, and a long one averages cleanly.
    const start = Math.floor((i * series.length) / cellCount);
    const end = Math.max(start + 1, Math.floor(((i + 1) * series.length) / cellCount));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += series[j];
    cells.push(sum / (end - start));
  }
  return cells;
}

/**
 * Expand a short list of anchor intensities into a per-2Hz-row series by
 * linear interpolation. The fixture stores anchors; the report renders rows.
 */
export function interpolateSeries(anchors: number[], sampleCount: number): number[] {
  if (anchors.length === 0 || sampleCount <= 0) return [];
  if (anchors.length === 1) return new Array<number>(sampleCount).fill(anchors[0]);

  const out: number[] = [];
  for (let i = 0; i < sampleCount; i += 1) {
    const t = (i / (sampleCount - 1 || 1)) * (anchors.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(anchors.length - 1, lo + 1);
    out.push(anchors[lo] + (anchors[hi] - anchors[lo]) * (t - lo));
  }
  return out;
}
