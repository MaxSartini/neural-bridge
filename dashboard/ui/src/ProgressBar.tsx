interface Props {
  /** Fraction complete, 0–1. Clamped; a non-finite value reads as 0. */
  value: number;
  /** Track height in px — 8 for an overall bar, 5 for a step's sub-bar. */
  height?: number;
  /** Fixed track width in px. Omit to fill the container. */
  width?: number;
  /** Announced name. Required: a bare progressbar tells a screen reader nothing. */
  label: string;
}

export default function ProgressBar({ value, height = 8, width, label }: Props) {
  const fraction = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
  const pct = Math.round(fraction * 100);
  return (
    <div
      className="progress-track"
      role="progressbar"
      aria-label={label}
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      style={{ height, width }}
    >
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}
