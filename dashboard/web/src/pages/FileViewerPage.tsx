import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchFile, rawUrl, ApiError, type FileResponse } from "../api/client";
import JsonView from "../components/JsonView";
import CsvTable from "../components/CsvTable";
import OversizedFileNotice from "../components/OversizedFileNotice";
import Breadcrumbs from "../components/Breadcrumbs";
import { Skeleton } from "@dashboard/ui";
import { fileKindForPath } from "@dashboard/shared";

type State =
  | { status: "loading" }
  | { status: "ok"; data: FileResponse }
  | { status: "unsupported" }
  | { status: "error"; message: string };

export default function FileViewerPage() {
  const params = useParams();
  const path = params["*"] ?? "";
  const isImage = fileKindForPath(path) === "image";

  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    if (isImage) return;
    let cancelled = false;
    setState({ status: "loading" });
    fetchFile(path)
      .then((data) => {
        if (!cancelled) setState({ status: "ok", data });
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 415) {
          setState({ status: "unsupported" });
        } else {
          setState({ status: "error", message: err.message });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [path, isImage]);

  return (
    <div>
      <Breadcrumbs path={path} />
      {isImage && <img src={rawUrl(path)} alt={path} className="preview-image" />}
      {!isImage && state.status === "loading" && <Skeleton lines={8} />}
      {!isImage && state.status === "error" && (
        <p className="error">Error loading {path}: {state.message}</p>
      )}
      {!isImage && state.status === "unsupported" && (
        <p>
          No inline preview for this file type.{" "}
          <a href={rawUrl(path)} target="_blank" rel="noopener noreferrer">
            Open / download it directly
          </a>
          .
        </p>
      )}
      {!isImage && state.status === "ok" && (
        <>
          {state.data.truncated && <OversizedFileNotice path={path} size={state.data.size} />}
          {state.data.kind === "json" && <JsonView content={state.data.content} />}
          {state.data.kind === "jsonl" && <JsonView content={state.data.content} jsonl />}
          {state.data.kind === "csv" && <CsvTable content={state.data.content} />}
          {(state.data.kind === "code" || state.data.kind === "text") && (
            <pre className="raw-fallback">{state.data.content}</pre>
          )}
          {state.data.kind === "markdown" && (
            <p>
              This is a Markdown file — <Link to={`/doc/${path}`}>view it rendered</Link> instead.
            </p>
          )}
        </>
      )}
    </div>
  );
}
