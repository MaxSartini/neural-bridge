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
const files = new Map<string, File>();

export function rememberVideo(analysisId: string, file: File): void {
  files.set(analysisId, file);
}

export function getVideo(analysisId: string): File | undefined {
  return files.get(analysisId);
}

export function forgetVideo(analysisId: string): void {
  files.delete(analysisId);
}
