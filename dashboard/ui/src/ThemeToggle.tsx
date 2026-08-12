import { useEffect, useState } from "react";

type Pin = "light" | "dark" | null;

const STORAGE_KEY = "theme";

function readPin(): Pin {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null; // private-mode localStorage throws on access
  }
}

function prefersDark(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

/**
 * Sun/moon toggle. No pin stored means "follow the OS" — we leave `data-theme`
 * off the root entirely, which leaves `color-scheme: light dark` on `:root` to
 * resolve every `light-dark()` token pair from the OS preference. A pin sets
 * the attribute and beats the OS in both directions.
 *
 * The `matchMedia` tracking below is **only** to choose which icon to draw. The
 * colours themselves are resolved by CSS, not here — do not reintroduce a JS
 * theme calculation on the back of it.
 *
 * The pre-paint half lives outside this component, once per bundle:
 * `web/index.html` (inline) and each bundle's own `theme-init.js` (external, so
 * the deployed CSP can forbid inline script). Without it a pinned theme flashes
 * the OS theme on every reload.
 */
export default function ThemeToggle() {
  const [pin, setPin] = useState<Pin>(() => readPin());
  const [osDark, setOsDark] = useState<boolean>(() => prefersDark());

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e: MediaQueryListEvent) => setOsDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (pin) {
      root.setAttribute("data-theme", pin);
    } else {
      root.removeAttribute("data-theme");
    }
    try {
      if (pin) localStorage.setItem(STORAGE_KEY, pin);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Non-fatal: the theme still applies for this session.
    }
  }, [pin]);

  const isDark = pin ? pin === "dark" : osDark;

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => setPin(isDark ? "light" : "dark")}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
    >
      {isDark ? (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="1.7" />
          <g stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
            <path d="M12 2.4v2.2M12 19.4v2.2M2.4 12h2.2M19.4 12h2.2" />
            <path d="M5.2 5.2l1.6 1.6M17.2 17.2l1.6 1.6M18.8 5.2l-1.6 1.6M6.8 17.2l-1.6 1.6" />
          </g>
        </svg>
      ) : (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M20.5 14.6A8.6 8.6 0 1 1 9.4 3.5a6.9 6.9 0 0 0 11.1 11.1Z"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinejoin="round"
          />
        </svg>
      )}
    </button>
  );
}
