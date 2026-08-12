import { useEvidenceNavigate } from "../../hooks/useEvidenceNavigate";
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { OrderedStage } from "../../data/scorecard";
import ChartTooltip from "./ChartTooltip";
import { sequentialRamp, BAR_MAX_SIZE, BAR_RADIUS } from "../../lib/chartTheme";

interface Props {
  data: OrderedStage[];
  sourceDoc: string;
  seriesName: string;
  baselineValue?: number;
  baselineLabel?: string;
  valueFormat?: (v: number) => string;
}

const defaultFormat = (v: number) => v.toFixed(4);

/** Ordered single-series bars, light→dark sequential ramp by rank. Reused for the AGAIN
 * bridge-development progression (ascending) and the motivating-failure comparison (with a
 * dashed reference line at the AR baseline). */
export default function ProgressionBarChart({
  data,
  sourceDoc,
  seriesName,
  baselineValue,
  baselineLabel,
  valueFormat = defaultFormat,
}: Props) {
  const openEvidence = useEvidenceNavigate();
  const colors = sequentialRamp(data.length);
  // Recharts hands a label formatter `string | number | undefined`.
  const labelFormat = (value: unknown) => (typeof value === "number" ? valueFormat(value) : "");

  return (
    <div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          onClick={() => openEvidence(sourceDoc)}
          /* Top margin clears the value label sitting above the tallest bar. */
          margin={{ top: 22, right: 8, left: 8, bottom: 0 }}
          style={{ cursor: "pointer" }}
        >
          {/* No grid and no value axis: the design labels each bar with its own
              figure instead, which reads better than making someone trace a bar
              back to a tick. The category axis stays — it names the stages. */}
          <XAxis
            dataKey="label"
            tick={{ fill: "var(--viz-muted)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            interval={0}
            height={36}
          />
          <YAxis hide />
          <Tooltip
            cursor={{ fill: "var(--viz-gridline)", opacity: 0.4 }}
            content={
              <ChartTooltip
                formatValue={valueFormat}
                titleFormatter={(_label, payload) => (payload[0]?.payload?.fullLabel as string) ?? ""}
              />
            }
          />
          {baselineValue !== undefined && (
            <ReferenceLine
              y={baselineValue}
              stroke="var(--viz-baseline)"
              strokeDasharray="4 4"
              label={{
                value: baselineLabel ?? "baseline",
                position: "insideTopRight",
                fill: "var(--viz-text-secondary)",
                fontSize: 11,
              }}
            />
          )}
          <Bar
            dataKey="value"
            name={seriesName}
            maxBarSize={BAR_MAX_SIZE}
            radius={BAR_RADIUS}
            isAnimationActive={false}
          >
            <LabelList
              dataKey="value"
              position="top"
              formatter={labelFormat}
              fill="var(--viz-text-secondary)"
              fontSize={11}
              fontFamily="var(--font-mono)"
            />
            {data.map((entry, i) => (
              <Cell key={entry.label} fill={colors[i]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <details className="chart-table-toggle">
        <summary>View as table</summary>
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th>{seriesName}</th>
            </tr>
          </thead>
          <tbody>
            {data.map((d) => (
              <tr key={d.label}>
                <td>{d.fullLabel}</td>
                <td>{valueFormat(d.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
