import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Check } from "lucide-react";
import { Blueprint, ProgressBar, Skeleton, Tag } from "@dashboard/ui";
import { AnalysisNotFound, analysisClient } from "../api";
import type { Analysis, PipelineStep } from "../api";

const POLL_MS = 400;

/** Consecutive non-fatal errors tolerated before giving up, with a linear backoff. */
const MAX_CONSECUTIVE_ERRORS = 4;

function StepGlyph({ step, index }: { step: PipelineStep; index: number }) {
  if (step.state === "done") {
    return (
      <span className="step-glyph step-glyph-done">
        <Check size={14} strokeWidth={2} aria-hidden="true" />
      </span>
    );
  }
  return (
    <span className={`step-glyph step-glyph-${step.state}`}>{index + 1}</span>
  );
}

function StepTag({ state }: { state: PipelineStep["state"] }) {
  if (state === "done") return <Tag variant="accent">Done</Tag>;
  if (state === "active") return <Tag variant="outline">In progress</Tag>;
  return <Tag variant="neutral">Queued</Tag>;
}

export default function AnalyzingPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [failure, setFailure] = useState<"missing" | "unreachable" | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let consecutiveErrors = 0;

    async function poll() {
      try {
        const next = await analysisClient.getAnalysis(id);
        if (cancelled) return;
        consecutiveErrors = 0;
        setAnalysis(next);
        if (next.status === "complete") {
          // Let the finished state paint before moving on, so the run does not
          // just vanish the instant the last step lands.
          timer = window.setTimeout(() => {
            if (!cancelled) navigate(`/analyses/${id}/report`, { replace: true });
          }, 600);
          return;
        }
        timer = window.setTimeout(poll, POLL_MS);
      } catch (err: unknown) {
        if (cancelled) return;
        // A missing id will never appear, so stop immediately. Anything else
        // might be a blip — this used to treat every error as fatal and print
        // "that analysis isn't here", which is a lie about a dropped request.
        if (err instanceof AnalysisNotFound) {
          setFailure("missing");
          return;
        }
        consecutiveErrors += 1;
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          setFailure("unreachable");
          return;
        }
        timer = window.setTimeout(poll, POLL_MS * consecutiveErrors);
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [id, navigate]);

  if (failure) {
    return (
      <div className="studio-column studio-column-narrow">
        <section className="hero">
          <div className="card-kicker">Analyzing</div>
          <h1 className="hero-title">
            {failure === "missing"
              ? "That analysis isn't here"
              : "Lost track of that analysis"}
          </h1>
          <p className="hero-technical studio-subtitle">
            {failure === "missing"
              ? "Runs live in the browser tab that started them, so a reload clears them. Start a new analysis to see the pipeline again."
              : "It stopped responding after several attempts. Starting a new analysis usually clears it."}
          </p>
          <div className="hero-actions">
            <Link to="/analyze" className="btn btn-secondary">
              Analyze a video
            </Link>
            <Link to="/sample" className="btn btn-ghost">
              See a sample report
            </Link>
          </div>
        </section>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="studio-column studio-column-narrow">
        <Skeleton lines={6} height={30} />
      </div>
    );
  }

  return (
    <div className="studio-column studio-column-narrow">
      <div className="studio-header">
        <div>
          <div className="card-kicker">Analyzing</div>
          <h1 className="hero-title studio-title">{analysis.projectName}</h1>
        </div>
        <Tag variant="outline">Processing</Tag>
      </div>

      {/* Omitted rather than guessed when the client cannot estimate honestly. */}
      {analysis.etaSeconds !== undefined && (
        <p className="studio-eta">Estimated {analysis.etaSeconds} seconds remaining</p>
      )}

      <div className="overall-progress">
        <ProgressBar value={analysis.overallProgress} height={8} label="Overall progress" />
      </div>

      <ol className="pipeline">
        {analysis.steps.map((step, i) => (
          <li key={step.id}>
            <Blueprint className="pipeline-row">
              <StepGlyph step={step} index={i} />
              <div className="pipeline-body">
                <div className="pipeline-title">{step.title}</div>
                {/* Only a step that measures itself gets a bar. An active step
                    with no measurable progress shows nothing — never a fake
                    percentage. */}
                {step.state === "active" && step.progress !== undefined && (
                  <div className="pipeline-subprogress">
                    <ProgressBar
                      value={step.progress}
                      height={5}
                      width={220}
                      label={`${step.title} progress`}
                    />
                  </div>
                )}
              </div>
              <StepTag state={step.state} />
            </Blueprint>
          </li>
        ))}
      </ol>

      <p className="studio-footnote">
        No audience data needed — Neural Bridge reads the video itself.
      </p>
    </div>
  );
}
