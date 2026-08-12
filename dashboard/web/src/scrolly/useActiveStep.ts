import { useCallback, useEffect, useRef, useState } from "react";

/** True when the reader has asked for reduced motion. The scroll story swaps to
 *  a static layout rather than degrading the animated one. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return reduced;
}

export interface ActiveStep {
  active: number;
  /** Attach to each step element: `ref={registerStep(i)}`. */
  registerStep: (index: number) => (el: HTMLElement | null) => void;
}

/**
 * Tracks which step element sits nearest the middle of the viewport.
 *
 * Uses a centre band (`rootMargin` shrinking the root to a horizontal strip)
 * rather than a per-step threshold: steps are taller than the strip, so a plain
 * threshold would leave the final step never fully intersecting. When several
 * steps overlap the band, the one closest to its centre wins — so fast
 * scrolling can't leave two steps fighting over the chart.
 */
export function useActiveStep(count: number, enabled: boolean): ActiveStep {
  const [active, setActive] = useState(0);
  const elements = useRef<(HTMLElement | null)[]>([]);

  const registerStep = useCallback(
    (index: number) => (el: HTMLElement | null) => {
      elements.current[index] = el;
    },
    [],
  );

  useEffect(() => {
    if (!enabled) return;

    const present = () =>
      elements.current.slice(0, count).filter(Boolean) as HTMLElement[];

    const pick = () => {
      const middle = window.innerHeight / 2;
      let best = 0;
      let bestDistance = Infinity;
      present().forEach((el, i) => {
        const rect = el.getBoundingClientRect();
        const distance = Math.abs(rect.top + rect.height / 2 - middle);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = i;
        }
      });
      setActive(best);
    };

    const observed = present();
    if (observed.length === 0) return;

    // The observer only wakes us when a step crosses the band; `pick` does the
    // choosing. Cheaper than a scroll listener, and correct at rest (first
    // paint, anchor jump) because we also run it once up front.
    const observer = new IntersectionObserver(pick, {
      rootMargin: "-45% 0px -45% 0px",
      threshold: 0,
    });
    observed.forEach((el) => observer.observe(el));

    pick();
    window.addEventListener("resize", pick);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", pick);
    };
  }, [count, enabled]);

  return { active, registerStep };
}
