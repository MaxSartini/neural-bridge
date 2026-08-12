import type { ReactNode } from "react";
import { Blueprint } from "@dashboard/ui";

interface Props {
  /**
   * Path under `public/media/`, e.g. `/media/hero.jpg`. Omit to render the
   * `fallback` artwork instead — which is the state the site ships in until
   * real photography exists.
   */
  src?: string;
  /** Required when `src` is set. Artwork carries its own label. */
  alt?: string;
  /** Generated artwork shown when there is no photograph yet. */
  fallback: ReactNode;
  /** CSS aspect-ratio, e.g. "16 / 9". */
  ratio?: string;
  className?: string;
}

/**
 * A framed image position.
 *
 * Two things the design system insists on, applied here once so no call site
 * has to remember them: the blueprint frame with its registration marks, and
 * the `.duotone` wrapper — "every content photograph goes through it", which is
 * what stops a stock photo from dragging its own palette into a two-colour
 * system.
 *
 * Adding real photography is a content change, not a code change: drop a file
 * into `public/media/` and pass its path. Nothing else moves.
 */
export default function ImageSlot({ src, alt, fallback, ratio = "16 / 9", className }: Props) {
  const classes = ["image-slot", "duotone"];
  if (className) classes.push(className);

  return (
    <Blueprint as="figure" className={classes.join(" ")} style={{ aspectRatio: ratio }}>
      {src ? (
        <img src={src} alt={alt ?? ""} className="image-slot-img" loading="lazy" />
      ) : (
        <div className="image-slot-art">{fallback}</div>
      )}
    </Blueprint>
  );
}
