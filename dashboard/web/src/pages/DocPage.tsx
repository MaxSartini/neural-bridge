import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchFile, ApiError, type FileResponse } from "../api/client";
import Markdown from "../components/Markdown";
import OversizedFileNotice from "../components/OversizedFileNotice";
import Breadcrumbs from "../components/Breadcrumbs";
import { Skeleton } from "@dashboard/ui";

interface Props {
  path?: string;
}

type State =
  | { status: "loading" }
  | { status: "ok"; data: FileResponse }
  | { status: "notfound" }
  /** The path resolves, but /api/file won't render it — a directory, or empty. */
  | { status: "notadoc" }
  | { status: "error"; message: string };

export default function DocPage({ path: explicitPath }: Props) {
  const params = useParams();
  const path = explicitPath ?? params["*"] ?? "";
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    fetchFile(path)
      .then((data) => {
        if (!cancelled) setState({ status: "ok", data });
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setState({ status: "notfound" });
        } else if (err instanceof ApiError && (err.status === 415 || err.status === 400)) {
          // A hand-typed /doc/<directory> URL. Nothing links here any more —
          // breadcrumbs point at /browse — but the API error string should not
          // be what the reader sees when it happens.
          setState({ status: "notadoc" });
        } else {
          setState({ status: "error", message: err.message });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  if (state.status === "loading") return <Skeleton lines={8} />;

  if (state.status === "notfound") {
    const dir = path.split("/").slice(0, -1).join("/");
    return (
      <div>
        <Breadcrumbs path={path} />
        <p>
          No document at <code>{path}</code>.
        </p>
        <Link to={`/browse/${dir}`}>Browse this directory instead</Link>
      </div>
    );
  }

  if (state.status === "notadoc") {
    return (
      <div>
        <Breadcrumbs path={path} />
        <p>
          <code>{path || "/"}</code> is a directory, not a document.
        </p>
        <Link to={`/browse/${path}`}>Browse it instead</Link>
      </div>
    );
  }

  if (state.status === "error") {
    return <p className="error">Error loading {path}: {state.message}</p>;
  }

  return (
    <div>
      <Breadcrumbs path={path} />
      {state.data.truncated && <OversizedFileNotice path={path} size={state.data.size} />}
      {state.data.kind === "markdown" ? (
        <Markdown content={state.data.content} docPath={path} />
      ) : (
        <pre className="raw-fallback">{state.data.content}</pre>
      )}
    </div>
  );
}
