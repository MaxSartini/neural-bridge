interface TooltipPayloadEntry {
  dataKey?: string | number;
  name?: string;
  value?: number | string;
  color?: string;
  payload?: Record<string, unknown>;
}

interface Props {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string | number;
  formatValue?: (value: number) => string;
  titleFormatter?: (label: string, payload: TooltipPayloadEntry[]) => string;
}

const defaultFormat = (v: number) => v.toString();

/** Shared tooltip: value leads (Strong), series name follows, line-key swatch per row. */
export default function ChartTooltip({ active, payload, label, formatValue, titleFormatter }: Props) {
  if (!active || !payload || payload.length === 0) return null;
  const format = formatValue ?? defaultFormat;
  const title = titleFormatter ? titleFormatter(String(label ?? ""), payload) : String(label ?? "");

  return (
    <div className="chart-tooltip">
      {title && <div className="tooltip-title">{title}</div>}
      {payload.map((entry, i) => (
        <div className="tooltip-row" key={entry.dataKey ?? i}>
          <span className="tooltip-swatch" style={{ background: entry.color }} />
          <span className="tooltip-value">
            {typeof entry.value === "number" ? format(entry.value) : entry.value}
          </span>
          <span className="tooltip-name">{entry.name}</span>
        </div>
      ))}
    </div>
  );
}
