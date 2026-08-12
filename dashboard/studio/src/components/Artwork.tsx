/**
 * On-brand artwork, generated rather than photographed.
 *
 * The site needs visual weight and there is no licensed photography to ship —
 * and the deployed CSP is `img-src 'self' data:`, so nothing can be pulled from
 * a stock service at runtime anyway. These draw in the Industry language
 * instead: hairlines, registration marks, square cells, one green. They need no
 * licence, they theme correctly, and they are replaceable — `ImageSlot` renders
 * a real photograph the moment one is dropped into `public/media/`.
 *
 * Every piece is inline SVG using `currentColor` and the token palette, so it
 * follows the theme with no second asset for dark mode.
 */

interface ArtProps {
  className?: string;
}

/**
 * A strip of film frames with a response trace running through them — the
 * product's whole idea in one mark: a cut, read left to right, with attention
 * rising and falling across it.
 */
export function FrameStripArt({ className }: ArtProps) {
  const cells = [0.18, 0.24, 0.32, 0.28, 0.44, 0.62, 0.95, 0.7, 0.48, 0.6, 0.88, 0.72];
  const w = 480;
  const h = 200;
  const cellW = w / cells.length;
  const points = cells
    .map((v, i) => `${i * cellW + cellW / 2},${h - 24 - v * (h - 70)}`)
    .join(" ");

  return (
    <svg
      className={className}
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="A strip of video frames with a predicted attention trace rising and falling across it."
    >
      {cells.map((v, i) => (
        <rect
          key={i}
          x={i * cellW + 2}
          y={h - 22}
          width={cellW - 4}
          height={16}
          fill="var(--mark-solid)"
          opacity={0.15 + v * 0.85}
        />
      ))}
      {cells.map((_, i) => (
        <rect
          key={`f${i}`}
          x={i * cellW + 2}
          y={14}
          width={cellW - 4}
          height={h - 44}
          fill="none"
          stroke="var(--divider)"
          strokeWidth="1"
        />
      ))}
      <polyline
        points={points}
        fill="none"
        stroke="var(--mark-solid)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      {cells.map((v, i) =>
        v > 0.85 ? (
          <circle
            key={`p${i}`}
            cx={i * cellW + cellW / 2}
            cy={h - 24 - v * (h - 70)}
            r="4"
            fill="var(--mark-solid)"
          />
        ) : null,
      )}
    </svg>
  );
}

/**
 * A field of registration marks — the design system's own grammar used as
 * texture. Sits behind a hero without competing with the type.
 */
export function RegistrationFieldArt({ className }: ArtProps) {
  const cols = 9;
  const rows = 4;
  const gap = 54;
  const marks = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const x = 20 + c * gap;
      const y = 20 + r * gap;
      marks.push(
        <g key={`${r}-${c}`} stroke="var(--divider)" strokeWidth="1">
          <line x1={x - 5} y1={y} x2={x + 5} y2={y} />
          <line x1={x} y1={y - 5} x2={x} y2={y + 5} />
        </g>,
      );
    }
  }
  return (
    <svg
      className={className}
      viewBox={`0 0 ${20 + cols * gap} ${20 + rows * gap}`}
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      {marks}
    </svg>
  );
}

/** The response map as a large graphic — used as the Results page's opener. */
export function ResponseMapArt({ className }: ArtProps) {
  const rows = 5;
  const cols = 16;
  const cells = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      // Deterministic, so the artwork is stable across renders and builds.
      const v = (Math.sin(c * 0.7 + r * 1.3) + 1) / 2;
      cells.push(
        <rect
          key={`${r}-${c}`}
          x={c * 30 + 1}
          y={r * 30 + 1}
          width={28}
          height={28}
          fill="var(--mark-solid)"
          opacity={0.1 + v * 0.7}
        />,
      );
    }
  }
  return (
    <svg
      className={className}
      viewBox={`0 0 ${cols * 30} ${rows * 30}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="A grid of cells shaded by predicted response intensity."
    >
      {cells}
    </svg>
  );
}
