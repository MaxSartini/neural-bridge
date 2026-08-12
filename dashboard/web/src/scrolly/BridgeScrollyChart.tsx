import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import type { StoryStep } from "./steps";
import { BAR_MAX_SIZE, BAR_RADIUS, sequentialRamp } from "../lib/chartTheme";

interface Props {
  step: StoryStep;
  animate: boolean;
  height?: number;
}

interface Row {
  label: string;
  fullLabel: string;
  /** null renders as a gap: the category keeps its slot on the axis so bars
   *  grow in place instead of the whole chart re-flowing on every step. */
  value: number | null;
  index: number;
}

const format = (v: number) => v.toFixed(4);

/**
 * One chart, driven entirely by the active story step.
 *
 * The category axis always carries every stage in the current act; only the
 * values appear and disappear. That is what makes the transitions read as bars
 * rising rather than as a different chart being swapped in.
 */
export default function BridgeScrollyChart({ step, animate, height = 340 }: Props) {
  const rows: Row[] = step.stages.map((stage, index) => ({
    label: stage.label,
    fullLabel: stage.fullLabel,
    value: step.visible.includes(index) ? stage.value : null,
    index,
  }));

  const ramp = sequentialRamp(step.stages.length);

  // A fixed domain per act keeps bar heights honest between steps — an
  // auto-domain would rescale as bars appear and make earlier stages look like
  // they shrank.
  const max = Math.max(...step.stages.map((s) => s.value));
  const domainTop = Math.ceil(max * 11) / 10;

  function fillFor(row: Row): string {
    if (step.highlight === row.index) {
      return step.tone === "critical" ? "var(--viz-critical)" : "var(--accent)";
    }
    if (step.tone === "critical") return "var(--ink-3)";
    return ramp[row.index];
  }

  return (
    <div className="scrolly-chart">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 16, right: 12, left: 0, bottom: 28 }}>
          <CartesianGrid stroke="var(--viz-gridline)" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: "var(--viz-text-secondary)", fontSize: 11 }}
            axisLine={{ stroke: "var(--viz-baseline)" }}
            tickLine={false}
            interval={0}
            angle={-15}
            textAnchor="end"
            height={52}
          />
          <YAxis
            domain={[0, domainTop]}
            tick={{ fill: "var(--viz-muted)", fontSize: 11 }}
            tickFormatter={(v: number) => v.toFixed(2)}
            axisLine={false}
            tickLine={false}
            width={46}
          />
          <Bar
            dataKey="value"
            maxBarSize={BAR_MAX_SIZE}
            radius={BAR_RADIUS}
            isAnimationActive={animate}
            animationDuration={520}
            animationEasing="ease-out"
          >
            <LabelList
              dataKey="value"
              position="top"
              // Recharts types this as RenderableText (which includes undefined),
              // and hidden bars arrive here as null — both render as no label.
              formatter={(v: unknown) => (typeof v === "number" ? format(v) : "")}
              style={{ fill: "var(--ink-2)", fontSize: 10, fontVariantNumeric: "tabular-nums" }}
            />
            {rows.map((row) => (
              <Cell key={row.label} fill={fillFor(row)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="scrolly-benchmark">{step.benchmark}</p>
    </div>
  );
}
