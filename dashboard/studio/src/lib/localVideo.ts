/**
 * The uploaded video, held in this tab and nowhere else.
 *
 * Kept out of `AnalysisClient` on purpose. That interface is the seam to a real
 * service, and a real service would never hand a `File` back — it would return
 * a frame URL. Putting the local file here keeps the seam honest and makes the
 * browser-only lifetime of the video an explicit, separate concern.
 *
 * Nothing is persisted: a reload drops the map, which is correct. The whole
 * point is that the file never leaves the page that read it.
 */

/**
 * At most one video is held at a time.
 *
 * This used to be an unbounded Map with a `forgetVideo` export that nothing
 * ever called, so every upload in a session stayed resident for the life of the
 * tab. The report revoked its object URL on unmount but the `File` behind it
 * was never released — so the comment claiming a 2GB pick would not be pinned
 * was only half true. Evicting on write bounds it to one file and removes the
 * teardown function nobody was calling.
 */
let held: { analysisId: string; file: File } | null = null;

export function rememberVideo(analysisId: string, file: File): void {
  held = { analysisId, file };
}

export function getVideo(analysisId: string): File | undefined {
  return held?.analysisId === analysisId ? held.file : undefined;
}
