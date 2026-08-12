import { useEffect, useRef, useState } from "react";

/**
 * Animates 0 → `target` over `duration` ms on mount, easing out.
 *
 * Returns `target` immediately (no animation frames scheduled at all) when the
 * user has asked for reduced motion, so the number is correct and static rather
 * than correct-after-a-delay.
 */
export function useCountUp(target: number, duration = 900): number {
  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const [value, setValue] = useState(reduced ? target : 0);
  const frame = useRef<number>();

  useEffect(() => {
    if (reduced) {
      setValue(target);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      // easeOutExpo — fast start, long settle, matching --ease-expo
      const eased = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
      setValue(target * eased);
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);

    // Browsers pause rAF in backgrounded or non-compositing tabs. Without this
    // fallback a scorecard opened in a background tab would sit at 0.00% until
    // the tab is focused — the animation is decoration, the number is not.
    const settle = setTimeout(() => setValue(target), duration + 100);

    return () => {
      if (frame.current !== undefined) cancelAnimationFrame(frame.current);
      clearTimeout(settle);
    };
  }, [target, duration, reduced]);

  return value;
}
