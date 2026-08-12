import { useEvidenceNavigate } from "../../hooks/useEvidenceNavigate";
import {
  Bar,
  BarChart,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { LandmarkComparison } from "../../data/scorecard";
import ChartTooltip from "./ChartTooltip";
import { GAIN_BAR_SIZE, BAR_RADIUS } from "../../lib/chartTheme";

interface Props {
  data: LandmarkComparison[];
}

interface ChartRow {
  category: string;
  title: string;
  sourceDoc: string;
  baselineGainPct: number;
  baselineLabel: string;
  controlGainPct: number;
  controlLabel: string;
}

const pct = (v: number) => `+${v.toFixed(2)}%`;

/** Room to the right of the plot for the mono captions that label each bar. */
const CAPTION_GUTTER = 170;

/**
 * Horizontal paired bars, one row per landmark: the baseline gain in green over
 * the control gain in steel, each captioned with its own figure.
 *
 * Horizontal because the design reads these as rows of a table rather than a
 * column chart — the roman numeral sits in a narrow left column and the bar
 * runs out from it. It also lets each bar carry its value as text, which is why
 * there is no legend and no y-axis: the caption says both the number and which
 * lane it belongs to.
 */
export default function GainBarChart({ data }: Props) {
  const openEvidence = useEvidenceNavigate();

  const chartData: ChartRow[] = data.map((d) => ({
    category: d.romanNumeral,
    title: d.title,
    sourceDoc: d.sourceDoc,
    baselineGainPct: d.baselineGainPct,
    baselineLabel: `vs ${d.baselineLabel}`,
    controlGainPct: d.controlGainPct,
    controlLabel: `vs ${d.controlLabel}`,
  }));

  // Round the axis up to the next 10% so the longest bar leaves a little air
  // rather than running to the very edge of the plot.
  const largest = Math.max(...data.flatMap((d) => [d.baselineGainPct, d.controlGainPct]));
  const axisMax = Math.ceil(largest / 10) * 10;

  function handleBarClick(item: { payload?: ChartRow }) {
    const sourceDoc = item.payload?.sourceDoc;
    if (sourceDoc) openEvidence(sourceDoc);
  }

  // Recharts hands a label formatter `string | number | undefined`, so narrow
  // rather than assuming a number and rendering "+NaN%" if the key is missing.
  const caption = (suffix: string) => (value: unknown) =>
    typeof value === "number" ? `${pct(value)} ${suffix}` : "";

  return (
    <div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart
          layout="vertical"
          data={chartData}
          barGap={5}
          margin={{ top: 4, right: CAPTION_GUTTER, left: 0, bottom: 4 }}
          style={{ cursor: "pointer" }}
        >
          <XAxis type="number" domain={[0, axisMax]} hide />
          <YAxis
            type="category"
            dataKey="category"
            width={36}
            tick={{
              fill: "var(--viz-text-secondary)",
              fontSize: 15,
              fontFamily: "var(--font-heading)",
            }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "var(--viz-gridline)", opacity: 0.4 }}
            content={
              <ChartTooltip
                formatValue={pct}
                titleFormatter={(_label, payload) => (payload[0]?.payload?.title as string) ?? ""}
              />
            }
          />
          <Bar
            dataKey="baselineGainPct"
            name="vs AR / raw baseline"
            fill="var(--viz-series-1)"
            barSize={GAIN_BAR_SIZE}
            radius={BAR_RADIUS}
            isAnimationActive={false}
            onClick={handleBarClick}
          >
            <LabelList
              dataKey="baselineGainPct"
              position="right"
              formatter={caption("vs baseline")}
              fill="var(--viz-text-secondary)"
              fontSize={11}
              fontFamily="var(--font-mono)"
            />
          </Bar>
          <Bar
            dataKey="controlGainPct"
            name="vs strongest matched control"
            fill="var(--viz-series-2)"
            barSize={GAIN_BAR_SIZE}
            radius={BAR_RADIUS}
            isAnimationActive={false}
            onClick={handleBarClick}
          >
            <LabelList
              dataKey="controlGainPct"
              position="right"
              formatter={caption("vs control")}
              fill="var(--viz-text-secondary)"
              fontSize={11}
              fontFamily="var(--font-mono)"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <details className="chart-table-toggle">
        <summary>View as table</summary>
        <table>
          <thead>
            <tr>
              <th>Landmark</th>
              <th>vs AR / raw baseline</th>
              <th>vs strongest matched control</th>
            </tr>
          </thead>
          <tbody>
            {data.map((d) => (
              <tr key={d.id}>
                <td>
                  {d.romanNumeral}. {d.title}
                </td>
                <td>{pct(d.baselineGainPct)}</td>
                <td>{pct(d.controlGainPct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
