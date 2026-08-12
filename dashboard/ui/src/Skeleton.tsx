interface Props {
  /** Number of shimmer bars to stack. */
  lines?: number;
  /** Height of each bar. */
  height?: number;
}

/**
 * Loading placeholder. Replaces the bare "Loading…" text so the page doesn't
 * jump from one line to a full document. Widths vary so the block reads as
 * text rather than a solid slab.
 *
 * The variation is a deterministic hash of the row index rather than a cycled
 * list. A fixed five-width cycle visibly repeated its pattern at the stack
 * heights actually in use (`lines={8}` on three pages), which reads as a
 * graphic rather than as prose.
 */
export default function Skeleton({ lines = 5, height = 14 }: Props) {
  return (
    <div className="skeleton-stack" role="status" aria-label="Loading">
      {Array.from({ length: lines }, (_, i) => (
        <div
          key={i}
          className="skeleton"
          // 72–100%, no visible period at any realistic stack height.
          style={{ height, width: `${72 + ((i * 37) % 29)}%` }}
        />
      ))}
    </div>
  );
}
