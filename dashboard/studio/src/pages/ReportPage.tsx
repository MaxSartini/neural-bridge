import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Blueprint, Skeleton, Tag } from "@dashboard/ui";
import { analysisClient } from "../api/mockAnalysisClient";
import type { AnalysisReport, Moment } from "../api/analysisClient";
import { confidence, reportMethodology } from "../content/claims";
import { HEAT_STRIP_CELLS, bucketSeries } from "../lib/heatStrip";
import { getVideo } from "../lib/localVideo";
import { formatTimecode } from "../lib/timecode";

interface MomentListProps {
  kicker: string;
  moments: Moment[];
  variant: "accent" | "outline";
  onSelect: (seconds: number) => void;
}

function MomentList({ kicker, moments, variant, onSelect }: MomentListProps) {
  return (
    <Blueprint className="moment-card">
      <div className="card-kicker">{kicker}</div>
      <ul className="moment-list">
        {moments.map((m) => (
          <li key={`${m.atSeconds}-${m.note}`}>
            <button type="button" className="moment-row" onClick={() => onSelect(m.atSeconds)}>
              <Tag variant={variant}>{formatTimecode(m.atSeconds)}</Tag>
              <span className="moment-note">{m.note}</span>
            </button>
          </li>
        ))}
      </ul>
    </Blueprint>
  );
}

interface Props {
  /**
   * Fixed analysis to render, overriding the route param. `/sample` uses this
   * so the shareable URL stays readable instead of redirecting to an opaque
   * analysis id.
   */
  analysisId?: string;
}

export default function ReportPage({ analysisId }: Props = {}) {
  const params = useParams();
  const id = analysisId ?? params.id ?? "";
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [selectedSecond, setSelectedSecond] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // The uploaded cut, if this run has one. The sample report has no file, so
  // the frame falls back to a flat field rather than inventing an image.
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  useEffect(() => {
    const file = getVideo(id);
    if (!file) return;
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    // Revoking on unmount is what keeps a 2GB pick from being pinned in memory
    // for the rest of the session.
    return () => {
      URL.revokeObjectURL(url);
      setVideoUrl(null);
    };
  }, [id]);

  // Seek the preview to whichever moment is selected, so clicking a peak or a
  // heat cell actually moves the frame.
  useEffect(() => {
    const el = videoRef.current;
    if (!el || selectedSecond === null) return;
    const seek = () => {
      if (Number.isFinite(el.duration)) {
        el.currentTime = Math.min(selectedSecond, Math.max(0, el.duration - 0.05));
      }
    };
    if (el.readyState >= 1) seek();
    else el.addEventListener("loadedmetadata", seek, { once: true });
  }, [selectedSecond, videoUrl]);

  useEffect(() => {
    let cancelled = false;
    void analysisClient.getReport(id).then((r) => {
      if (!cancelled) {
        setReport(r);
        setSelectedSecond(r.peaks[0]?.atSeconds ?? null);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const cells = useMemo(
    () => (report ? bucketSeries(report.series, HEAT_STRIP_CELLS) : []),
    [report],
  );

  /**
   * Stretch the strip across its full contrast range.
   *
   * Predicted movement rarely spans 0–1 within a single cut — this fixture sits
   * roughly 0.17–0.84 — so mapping raw values straight to opacity wastes most of
   * the range and the strip reads as one flat colour. Normalising per report
   * means the quietest moment in *this* cut is always the faintest cell and the
   * strongest is always the boldest, which is what the strip is for: relative
   * movement within one video, never an absolute scale across videos.
   */
  const cellOpacity = useMemo(() => {
    if (cells.length === 0) return () => 0.5;
    const lo = Math.min(...cells);
    const hi = Math.max(...cells);
    const span = hi - lo;
    // A genuinely flat cut has no relative story to tell; show it evenly.
    if (span < 1e-6) return () => 0.55;
    return (v: number) => 0.16 + ((v - lo) / span) * 0.84;
  }, [cells]);

  const timeline = useMemo(() => {
    if (!report) return [];
    const step = report.durationSeconds / Math.max(1, report.series.length - 1);
    return report.series.map((value, i) => ({ at: i * step, value }));
  }, [report]);

  if (!report) {
    return (
      <div className="studio-column">
        <Skeleton lines={8} height={26} />
      </div>
    );
  }

  const secondsPerCell = report.durationSeconds / HEAT_STRIP_CELLS;

  return (
    <div className="studio-column">
      <div className="studio-header report-header">
        <div>
          <div className="card-kicker">Report</div>
          <h1 className="hero-title studio-title">{report.projectName}</h1>
        </div>
        <div className="report-actions">
          <Tag variant="accent">Complete</Tag>
          <Blueprint as="button" type="button" className="btn btn-secondary" onClick={() => window.print()}>
            Export report
          </Blueprint>
        </div>
      </div>

      {report.isSample && (
        <p className="sample-notice" role="note">
          These figures are sample output for demonstration. The methodology and the programme
          results below are real; the per-video numbers on this page are not measurements.
        </p>
      )}

      <div className="report-top">
        <Blueprint as="figure" className="frame-preview duotone">
          {videoUrl ? (
            <video ref={videoRef} src={videoUrl} className="frame-video" muted playsInline preload="metadata" />
          ) : (
            <div className="frame-placeholder" aria-hidden="true" />
          )}
          <Tag variant="accent" className="frame-tag">
            {selectedSecond !== null ? `Peak · ${formatTimecode(selectedSecond)}` : "Preview"}
          </Tag>
          <figcaption className="visually-hidden">
            {videoUrl
              ? "Frame from your video at the selected moment."
              : "No video attached to this report."}
          </figcaption>
        </Blueprint>

        <div className="methodology">
          <div className="card-kicker">{reportMethodology.kicker}</div>
          <p className="methodology-body">{reportMethodology.body(reportMethodology.figure)}</p>
          <div className="methodology-figures">
            <div>
              <div className="methodology-stat">{report.movementRankingScore.toFixed(3)}</div>
              <div className="methodology-label">attention movement score</div>
            </div>
            <div>
              <div className="methodology-stat">Top {report.responseWindowPercentile}%</div>
              <div className="methodology-label">of the predicted response range</div>
            </div>
          </div>
        </div>
      </div>

      <Blueprint className="chart-card timeline-card">
        <div className="card-kicker">Predicted response — full timeline</div>
        <div className="timeline-chart">
          <ResponsiveContainer width="100%" height={80}>
            <LineChart data={timeline} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
              <XAxis dataKey="at" hide type="number" domain={[0, report.durationSeconds]} />
              <YAxis hide domain={[0, 1]} />
              <Tooltip
                cursor={{ stroke: "var(--viz-gridline)" }}
                labelFormatter={(v) => formatTimecode(Number(v))}
                formatter={(v: unknown) => [
                  typeof v === "number" ? v.toFixed(2) : "—",
                  "predicted movement",
                ]}
                contentStyle={{
                  background: "var(--surface)",
                  border: "1px solid var(--divider)",
                  borderRadius: 0,
                  fontSize: 12,
                }}
              />
              {/* The draw-on animation lives in CSS against
                  .recharts-line-curve. Setting strokeDasharray/Offset here as
                  props froze the line invisible, because Recharts puts
                  className on its layer group rather than the path, so nothing
                  ever animated the offset back to zero. */}
              <Line
                type="linear"
                dataKey="value"
                stroke="var(--mark-solid)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="heat-strip">
          {cells.map((intensity, i) => {
            const at = i * secondsPerCell;
            return (
              <button
                type="button"
                key={i}
                className="heat-cell heat-cell-animate"
                style={
                  {
                    opacity: cellOpacity(intensity),
                    "--cell-delay": `${i * 45}ms`,
                  } as React.CSSProperties
                }
                onClick={() => setSelectedSecond(Math.round(at))}
                aria-label={`${formatTimecode(at)}, predicted movement ${intensity.toFixed(2)}`}
              />
            );
          })}
        </div>

        <div className="timeline-axis">
          <span>{formatTimecode(0)}</span>
          <span>{formatTimecode(report.durationSeconds / 2)}</span>
          <span>{formatTimecode(report.durationSeconds)}</span>
        </div>
      </Blueprint>

      <div className="moment-grid">
        <MomentList
          kicker="Peaks"
          moments={report.peaks}
          variant="accent"
          onSelect={setSelectedSecond}
        />
        <MomentList
          kicker="Weak moments"
          moments={report.weakMoments}
          variant="outline"
          onSelect={setSelectedSecond}
        />
      </div>

      {/* The honesty contract. It mirrors the README's "Honest boundaries" and
          is not optional — a report that shows peaks without saying what it does
          not claim is the overclaim the whole programme is careful to avoid. */}
      <Blueprint className="chart-card confidence-card">
        <div className="card-kicker">{confidence.kicker}</div>
        <h3 className="confidence-title">{confidence.title}</h3>
        <p className="confidence-body">{confidence.body}</p>
      </Blueprint>
    </div>
  );
}
