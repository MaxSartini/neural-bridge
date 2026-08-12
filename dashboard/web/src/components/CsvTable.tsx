import { useMemo, useState } from "react";
import Papa from "papaparse";

interface Props {
  content: string;
}

const PAGE_SIZE = 50;

export default function CsvTable({ content }: Props) {
  const parsed = useMemo(
    () => Papa.parse<Record<string, string>>(content, { header: true, skipEmptyLines: true }),
    [content],
  );
  const columns = parsed.meta.fields ?? [];
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<1 | -1>(1);
  const [page, setPage] = useState(0);

  const rows = useMemo(() => {
    if (!sortCol) return parsed.data;
    const copy = [...parsed.data];
    copy.sort((a, b) => {
      const av = a[sortCol] ?? "";
      const bv = b[sortCol] ?? "";
      const an = Number(av);
      const bn = Number(bv);
      const cmp = av !== "" && bv !== "" && !Number.isNaN(an) && !Number.isNaN(bn)
        ? an - bn
        : av.localeCompare(bv);
      return cmp * sortDir;
    });
    return copy;
  }, [parsed.data, sortCol, sortDir]);

  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function onSort(col: string) {
    if (sortCol === col) {
      setSortDir((d) => (d === 1 ? -1 : 1));
    } else {
      setSortCol(col);
      setSortDir(1);
    }
    setPage(0);
  }

  if (columns.length === 0) {
    return <pre className="raw-fallback">{content}</pre>;
  }

  return (
    <div className="csv-table-wrap">
      <div className="csv-meta">
        {rows.length} rows × {columns.length} columns
      </div>
      <div className="csv-scroll">
        <table className="csv-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col} onClick={() => onSort(col)}>
                  {col}
                  {sortCol === col ? (sortDir === 1 ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => (
              <tr key={page * PAGE_SIZE + i}>
                {columns.map((col) => (
                  <td key={col}>{row[col]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="csv-pagination">
        <button disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
          Prev
        </button>
        <span>
          Page {page + 1} / {pageCount}
        </span>
        <button disabled={page >= pageCount - 1} onClick={() => setPage((p) => p + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}
