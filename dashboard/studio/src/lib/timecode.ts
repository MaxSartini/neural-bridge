/**
 * Seconds to `m:ss`, the form every timestamp in the report uses ("0:14").
 * Minutes are not zero-padded, seconds always are.
 */
export function formatTimecode(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const whole = Math.floor(seconds);
  const mins = Math.floor(whole / 60);
  const secs = whole % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}
