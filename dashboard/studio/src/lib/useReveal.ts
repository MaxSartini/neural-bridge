import { useEffect, useRef, useState } from "react";

/**
 * Reveal-on-scroll, for the one gesture the Industry language can carry:
 * a section arriving rather than a section decorated.
 *
 * Motion here is deliberately small — an 8px rise and a fade, plus the
 * registration marks drawing themselves in. ADR-0004 still stands: no
 * gradients, no elevation, no rounded corners, no second accent. "A bit of
 * flash" is delivered by things *moving in* the wireframe grammar, not by
 * abandoning it.
 *
 * Reduced motion is honoured by never arming the observer at all, so the
 * content is simply present. That is stronger than animating to a final state
 * quickly: nothing transitions, so nothing can be caught mid-transition.
 */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/**
 * How long to wait for the observer before showing the content anyway.
 * Long enough that a normal scroll-triggered reveal still feels intentional,
 * short enough that a visitor never notices the fallback.
 */
const FAIL_OPEN_MS = 1200;

/**
 * Returns a ref to attach and whether the element has been seen. Once shown it
 * stays shown — re-animating on scroll-back is the thing that makes reveal
 * effects tiresome.
 *
 * **This fails open, deliberately.** The hidden state lives in CSS, so if the
 * observer never fires the content stays invisible permanently — a decorative
 * effect silently eating the page. That is not hypothetical: verifying this
 * work in a non-compositing browser pane produced exactly that, with a fresh
 * observer on a fully visible element never firing once. Embedded webviews,
 * some extensions and prerenderers do the same thing.
 *
 * So a timer runs alongside the observer and shows the content regardless.
 * Motion is an enhancement; the words are the product.
 */
export function useReveal<T extends HTMLElement = HTMLDivElement>() {
  const ref = useRef<T>(null);
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState(reduced);

  useEffect(() => {
    if (reduced) {
      setShown(true);
      return;
    }
    const el = ref.current;
    if (!el) {
      setShown(true);
      return;
    }

    // Already in view at mount (above the fold): show without waiting for a
    // scroll that may never come.
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight * 0.92) {
      setShown(true);
      return;
    }

    if (typeof IntersectionObserver === "undefined") {
      setShown(true);
      return;
    }

    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setShown(true);
            io.disconnect();
          }
        }
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    io.observe(el);

    const failOpen = window.setTimeout(() => setShown(true), FAIL_OPEN_MS);

    return () => {
      io.disconnect();
      window.clearTimeout(failOpen);
    };
  }, [reduced]);

  return { ref, shown };
}
