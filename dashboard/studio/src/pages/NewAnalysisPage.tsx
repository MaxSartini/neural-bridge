import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Upload } from "lucide-react";
import { Blueprint } from "@dashboard/ui";
import { analysisClient } from "../api/mockAnalysisClient";
import { rememberVideo } from "../lib/localVideo";
import type { ContentType, Objective } from "../api/analysisClient";

const ACCEPTED_TYPES = ["video/mp4", "video/quicktime"];
const MAX_BYTES = 2 * 1024 * 1024 * 1024; // 2GB

const CONTENT_TYPES: { value: ContentType; label: string }[] = [
  { value: "advertisement", label: "Advertisement" },
  { value: "trailer", label: "Trailer" },
  { value: "gameplay", label: "Gameplay" },
];

const OBJECTIVES: { value: Objective; label: string }[] = [
  { value: "peaks", label: "Peak engagement" },
  { value: "weak", label: "Avoid weak moments" },
  { value: "full", label: "Full diagnostic report" },
];

function formatSize(bytes: number): string {
  const kb = bytes / 1024;
  if (kb < 1024) return `${Math.max(1, Math.round(kb))} KB`;
  const mb = kb / 1024;
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(1)} MB`;
}

/**
 * Validation runs before anything else touches the file. The size cap matters
 * even though nothing is uploaded — reading a 2GB file into an object URL is
 * still 2GB of the visitor's memory.
 */
function validate(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return `${file.name} is not an MP4 or MOV file.`;
  }
  if (file.size > MAX_BYTES) {
    return `${file.name} is ${formatSize(file.size)} — the limit is 2 GB.`;
  }
  return null;
}

export default function NewAnalysisPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [projectName, setProjectName] = useState("");
  const [contentType, setContentType] = useState<ContentType>("advertisement");
  const [objective, setObjective] = useState<Objective>("peaks");
  const [notes, setNotes] = useState("");

  function accept(candidate: File | undefined) {
    if (!candidate) return;
    const problem = validate(candidate);
    if (problem) {
      setError(problem);
      setFile(null);
      return;
    }
    setError(null);
    setFile(candidate);
    // Give the project a sensible default rather than making someone type the
    // filename they just picked.
    if (!projectName.trim()) {
      setProjectName(candidate.name.replace(/\.[^.]+$/, ""));
    }
  }

  async function handleSubmit() {
    if (!file || submitting) return;
    setSubmitting(true);
    const { id } = await analysisClient.createAnalysis({
      projectName,
      contentType,
      objective,
      notes,
      file,
    });
    // Held in this tab so the report can show a real frame from the cut. It is
    // never sent anywhere — see lib/localVideo.
    rememberVideo(id, file);
    navigate(`/analyses/${id}`, { state: { fileName: file.name } });
  }

  return (
    <div className="studio-column">
      <section className="hero">
        <div className="card-kicker">New analysis</div>
        <h1 className="hero-title">Upload your video</h1>
        <p className="hero-technical studio-subtitle">
          We&apos;ll turn it into a forward-looking response map — likely peaks, weak moments,
          and the sections most worth testing.
        </p>
      </section>

      <div className="upload-grid">
        <Blueprint
          className={`dropzone${dragging ? " dropzone-active" : ""}`}
          onDragOver={(e: React.DragEvent) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e: React.DragEvent) => {
            e.preventDefault();
            setDragging(false);
            accept(e.dataTransfer.files[0]);
          }}
        >
          <Upload size={30} strokeWidth={1.5} aria-hidden="true" className="dropzone-icon" />
          {file ? (
            <>
              <div className="dropzone-title">{file.name}</div>
              <div className="dropzone-hint">{formatSize(file.size)} · ready to analyze</div>
            </>
          ) : (
            <>
              <div className="dropzone-title">Drop your advertisement here</div>
              <div className="dropzone-hint">MP4 or MOV, up to 2GB</div>
            </>
          )}
          <Blueprint
            as="button"
            type="button"
            className="btn btn-secondary"
            onClick={() => inputRef.current?.click()}
          >
            {file ? "Choose a different file" : "Browse files"}
          </Blueprint>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(",")}
            className="visually-hidden"
            onChange={(e) => accept(e.target.files?.[0])}
          />
          {error && (
            <p className="dropzone-error" role="alert">
              {error}
            </p>
          )}
        </Blueprint>

        <div className="upload-form">
          <div className="field">
            <label htmlFor="project-name">Project name</label>
            <input
              id="project-name"
              className="input"
              value={projectName}
              placeholder="Holiday Sale — 30s cut"
              onChange={(e) => setProjectName(e.target.value)}
            />
          </div>

          <div className="field">
            <span className="field-label" id="content-type-label">
              Content type
            </span>
            <div className="seg" role="radiogroup" aria-labelledby="content-type-label">
              {CONTENT_TYPES.map((opt) => (
                <label className="seg-opt" key={opt.value}>
                  <input
                    type="radio"
                    name="content-type"
                    checked={contentType === opt.value}
                    onChange={() => setContentType(opt.value)}
                  />
                  {opt.label}
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <span className="field-label" id="objective-label">
              Optimize for
            </span>
            <div className="radio-group" role="radiogroup" aria-labelledby="objective-label">
              {OBJECTIVES.map((opt) => (
                <label className="radio" key={opt.value}>
                  <input
                    type="radio"
                    name="objective"
                    checked={objective === opt.value}
                    onChange={() => setObjective(opt.value)}
                  />
                  <span className="dot" />
                  {opt.label}
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <label htmlFor="notes">Notes</label>
            <textarea
              id="notes"
              className="input"
              rows={2}
              placeholder="Target audience, prior test results, brand constraints…"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </div>
        </div>
      </div>

      <Blueprint
        as="button"
        type="button"
        className="btn btn-primary btn-block submit-analysis"
        disabled={!file || submitting}
        onClick={handleSubmit}
      >
        {submitting ? "Starting…" : "Analyze video"}
        <ArrowRight size={14} strokeWidth={1.5} aria-hidden="true" />
      </Blueprint>

      <p className="studio-footnote">
        Your file is read in this browser and is not uploaded anywhere. This build runs a
        demonstration of the pipeline against sample output.
      </p>
    </div>
  );
}
