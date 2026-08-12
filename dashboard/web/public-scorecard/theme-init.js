// Stamps a pinned theme before first paint. Kept as a real file rather than an
// inline <script> so the Content-Security-Policy can forbid inline script
// entirely instead of carrying 'unsafe-inline' or a brittle hash.
// No pin in localStorage means no attribute, which leaves prefers-color-scheme
// in charge.
try {
  var t = localStorage.getItem("theme");
  if (t === "light" || t === "dark") {
    document.documentElement.setAttribute("data-theme", t);
  }
} catch (e) {
  /* private-mode localStorage throws; fall back to following the OS */
}
