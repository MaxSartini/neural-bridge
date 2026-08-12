import type { ReactNode } from "react";
import { useReveal } from "../lib/useReveal";

interface Props {
  children: ReactNode;
  /** Stagger within a group, in ms. Keep small — this is a lift, not a show. */
  delay?: number;
  className?: string;
}

/**
 * Wraps a section so it rises into place the first time it is seen. The actual
 * movement is two CSS properties (see `.reveal` in studio.css); this only
 * decides when to flip the attribute.
 */
export default function Reveal({ children, delay = 0, className }: Props) {
  const { ref, shown } = useReveal<HTMLDivElement>();
  return (
    <div
      ref={ref}
      className={className ? `reveal ${className}` : "reveal"}
      data-shown={shown}
      style={delay ? { transitionDelay: `${delay}ms` } : undefined}
    >
      {children}
    </div>
  );
}
