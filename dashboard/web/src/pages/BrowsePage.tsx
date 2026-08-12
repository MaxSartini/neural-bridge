import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { fetchTree, type TreeEntry } from "../api/client";
import FileTree from "../components/FileTree";
import Breadcrumbs from "../components/Breadcrumbs";
import { Skeleton } from "@dashboard/ui";

export default function BrowsePage() {
  const params = useParams();
  const path = params["*"] ?? "";
  const [entries, setEntries] = useState<TreeEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setEntries(null);
    setError(null);
    fetchTree(path)
      .then((res) => {
        if (!cancelled) setEntries(res.entries);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return (
    <div>
      <Breadcrumbs path={path} />
      {error && <p className="error">Error: {error}</p>}
      {!error && !entries && <Skeleton lines={6} height={30} />}
      {entries && <FileTree entries={entries} />}
    </div>
  );
}
