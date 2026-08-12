import { describe, expect, it } from "vitest";
import { HEAT_STRIP_CELLS, bucketSeries, interpolateSeries } from "./heatStrip";

describe("bucketSeries", () => {
  it("returns the requested number of cells", () => {
    expect(bucketSeries([1, 2, 3, 4, 5, 6], 3)).toHaveLength(3);
    expect(bucketSeries(new Array(60).fill(0.5))).toHaveLength(HEAT_STRIP_CELLS);
  });

  it("averages within a cell rather than sampling one row", () => {
    // Sampling every Nth row would return 0 here and lose the spike entirely.
    expect(bucketSeries([0, 1, 0, 1], 2)).toEqual([0.5, 0.5]);
  });

  it("keeps a lone spike visible in its own cell", () => {
    const series = new Array(12).fill(0);
    series[6] = 1;
    const cells = bucketSeries(series, 12);
    expect(cells[6]).toBe(1);
    expect(cells.filter((c) => c > 0)).toHaveLength(1);
  });

  it("preserves a flat series exactly", () => {
    expect(bucketSeries(new Array(48).fill(0.42), 12).every((c) => Math.abs(c - 0.42) < 1e-9)).toBe(
      true,
    );
  });

  it("covers every sample across the cells, leaving no gaps", () => {
    const series = Array.from({ length: 60 }, (_, i) => i);
    const cells = bucketSeries(series, 12);
    // Mean of all cells equals the mean of the series when buckets are even.
    const cellMean = cells.reduce((a, b) => a + b, 0) / cells.length;
    const seriesMean = series.reduce((a, b) => a + b, 0) / series.length;
    expect(cellMean).toBeCloseTo(seriesMean, 6);
  });

  it("handles a series shorter than the cell count without holes", () => {
    const cells = bucketSeries([0, 1], 6);
    expect(cells).toHaveLength(6);
    expect(cells.every((c) => Number.isFinite(c))).toBe(true);
  });

  it("degrades safely on empty input", () => {
    expect(bucketSeries([], 4)).toEqual([0, 0, 0, 0]);
    expect(bucketSeries([1, 2, 3], 0)).toEqual([]);
  });
});

describe("interpolateSeries", () => {
  it("hits both anchors at the ends", () => {
    const out = interpolateSeries([0.2, 0.8], 10);
    expect(out[0]).toBeCloseTo(0.2, 9);
    expect(out[out.length - 1]).toBeCloseTo(0.8, 9);
  });

  it("produces the requested sample count", () => {
    expect(interpolateSeries([0.1, 0.5, 0.9], 60)).toHaveLength(60);
  });

  it("moves monotonically between rising anchors", () => {
    const out = interpolateSeries([0, 1], 20);
    for (let i = 1; i < out.length; i += 1) expect(out[i]).toBeGreaterThanOrEqual(out[i - 1]);
  });

  it("round-trips anchors back through bucketing, approximately", () => {
    // The fixture stores 12 anchors, expands to 60 rows, and the strip folds
    // back to 12 cells. Those cells should resemble the anchors they came from.
    const anchors = [0.15, 0.2, 0.3, 0.25, 0.35, 0.5, 0.9, 0.6, 0.4, 0.55, 0.85, 0.7];
    const cells = bucketSeries(interpolateSeries(anchors, 60), 12);
    cells.forEach((c, i) => expect(Math.abs(c - anchors[i])).toBeLessThan(0.1));
  });

  it("degrades safely", () => {
    expect(interpolateSeries([], 10)).toEqual([]);
    expect(interpolateSeries([0.5], 3)).toEqual([0.5, 0.5, 0.5]);
    expect(interpolateSeries([0, 1], 0)).toEqual([]);
  });
});
