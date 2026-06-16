"""
File Parser Utility
Supports text extraction from PDF, Markdown, TXT files
"""

import os
from pathlib import Path
from typing import List, Optional


def _read_text_with_fallback(file_path: str) -> str:
    """
    Read text file with automatic encoding detection if UTF-8 fails.

    Uses multi-level fallback strategy:
    1. First try UTF-8 decoding
    2. Use charset_normalizer for encoding detection
    3. Fall back to chardet for encoding detection
    4. Finally use UTF-8 + errors='replace' as fallback

    Args:
        file_path: File path

    Returns:
        Decoded text content
    """
    data = Path(file_path).read_bytes()
    
    # First try UTF-8
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass

    # Try charset_normalizer for encoding detection
    encoding = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            encoding = best.encoding
    except Exception:
        pass

    # Fall back to chardet
    if not encoding:
        try:
            import chardet
            result = chardet.detect(data)
            encoding = result.get('encoding') if result else None
        except Exception:
            pass

    # Final fallback: use UTF-8 + replace
    if not encoding:
        encoding = 'utf-8'

    return data.decode(encoding, errors='replace')


class FileParser:
    """File Parser"""

    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt', '.csv', '.xlsx', '.xls'}

    @classmethod
    def extract_text(cls, file_path: str, display_name: Optional[str] = None) -> str:
        """
        Extract text from file

        Args:
            file_path: File path
            display_name: Optional human-readable name for the asset (e.g., ticker like 'PRX.JO').

        Returns:
            Extracted text content
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        suffix = path.suffix.lower()

        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {suffix}")

        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)
        elif suffix == '.csv':
            # Prefer provided display name (e.g., original filename without extension)
            return cls._extract_from_csv(file_path, asset_name=(display_name or path.stem))
        elif suffix in {'.xlsx', '.xls'}:
            return cls._extract_from_excel(file_path, asset_name=(display_name or path.stem))

        raise ValueError(f"Cannot handle file format: {suffix}")

    # Raw numeric dumps defeat NER; produce a prose summary that names each
    # column as an Asset/Metric so entities land in the knowledge graph.
    _SAMPLE_ROWS = 5

    @classmethod
    def _build_metric_data_summary(cls, series, col_name: str, asset_name: str):
        """Build structured JSON summary for a metric column. Used for agent context."""
        try:
            import pandas as pd
        except ImportError:
            pd = None
            return None
        
        s = series.dropna()
        if len(s) == 0:
            return None
        
        if pd is not None and pd.api.types.is_numeric_dtype(s):
            first, last = s.iloc[0], s.iloc[-1]
            direction = "rising" if last > first else "falling" if last < first else "flat"
            pct_change = ((last - first) / first * 100) if first not in (0, 0.0) else 0
            
            # Collect recent sample values (last 3)
            recent_samples = []
            for idx in range(max(0, len(s) - 3), len(s)):
                recent_samples.append({"index": idx, "value": float(s.iloc[idx])})
            
            return {
                "metric_name": f"{asset_name}.{col_name}",
                "type": "numeric",
                "data_points": len(s),
                "latest_value": float(last),
                "previous_value": float(s.iloc[-2]) if len(s) > 1 else float(last),
                "min_value": float(s.min()),
                "max_value": float(s.max()),
                "mean": float(s.mean()),
                "std_dev": float(s.std()),
                "trend": direction,
                "pct_change_total": round(pct_change, 2),
                "recent_samples": recent_samples,
                "range_display": f"{s.min():.4g} to {s.max():.4g}"
            }
        return None

    @classmethod
    def _summarize_dataframe(cls, df, asset_name: str, sheet_label: str = "") -> str:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        total = len(df)
        cols = list(df.columns)
        scope = f"{asset_name}{('.' + sheet_label) if sheet_label else ''}"
        lines = [
            f"=== Asset: {asset_name} ===",
            f"The asset '{asset_name}' is a financial time series with {total} records across {len(cols)} columns: {', '.join(str(c) for c in cols)}.",
        ]

        for col in cols:
            series = df[col]
            qualified = f"{scope}.{col}"
            try:
                if pd is not None and pd.api.types.is_numeric_dtype(series):
                    s = series.dropna()
                    if len(s) == 0:
                        continue
                    first, last = s.iloc[0], s.iloc[-1]
                    direction = "rising" if last > first else "falling" if last < first else "flat"
                    pct = ((last - first) / first * 100) if first not in (0, 0.0) else 0
                    previous = s.iloc[-2] if len(s) > 1 else last
                    
                    # Enhanced summary with concrete values agents can cite
                    lines.append(
                        f"Metric '{qualified}' of asset '{asset_name}': {len(s)} data points. "
                        f"Latest={last:.4g}, Previous={previous:.4g}, "
                        f"Range {s.min():.4g}–{s.max():.4g}, Mean {s.mean():.4g}. "
                        f"Trend: {direction} ({pct:+.2f}% from start={first:.4g}). "
                        f"Volatility (std): {s.std():.4g}"
                    )
                else:
                    s = series.dropna().astype(str)
                    uniq = s.unique()
                    sample = ", ".join(repr(v) for v in uniq[:5])
                    lines.append(
                        f"Field '{qualified}' of asset '{asset_name}' is categorical "
                        f"with {len(uniq)} unique values (e.g. {sample})."
                    )
            except Exception:
                lines.append(f"Field '{qualified}' of asset '{asset_name}': summary unavailable.")

        sample = df.head(cls._SAMPLE_ROWS)
        lines.append(f"\nSample rows from asset '{asset_name}' (first {len(sample)} of {total}):")
        lines.append(sample.to_csv(index=False).strip())
        return "\n".join(lines)

    @classmethod
    def _extract_from_csv(cls, file_path: str, asset_name: str = "Dataset") -> str:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
        return cls._summarize_dataframe(df, asset_name=asset_name)

    @classmethod
    def _extract_from_excel(cls, file_path: str, asset_name: str = "Dataset") -> str:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas openpyxl")
        sheets = pd.read_excel(file_path, sheet_name=None)
        return "\n\n".join(
            cls._summarize_dataframe(df, asset_name=asset_name, sheet_label=str(name))
            for name, df in sheets.items()
        )

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required: pip install PyMuPDF")

        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        """Extract text from Markdown with automatic encoding detection"""
        return _read_text_with_fallback(file_path)

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Extract text from TXT with automatic encoding detection"""
        return _read_text_with_fallback(file_path)

    @classmethod
    def extract_from_multiple(cls, file_paths: List[str]) -> str:
        """
        Extract text from multiple files and merge

        Args:
            file_paths: List of file paths

        Returns:
            Merged text
        """
        all_texts = []

        for i, file_path in enumerate(file_paths, 1):
            try:
                text = cls.extract_text(file_path)
                filename = Path(file_path).name
                all_texts.append(f"=== Document {i}: {filename} ===\n{text}")
            except Exception as e:
                all_texts.append(f"=== Document {i}: {file_path} (extraction failed: {str(e)}) ===")

        return "\n\n".join(all_texts)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into chunks

    Args:
        text: Original text
        chunk_size: Characters per chunk
        overlap: Overlapping characters

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to split at sentence boundaries
        if end < len(text):
            window = text[start:end]
            # Find nearest sentence ending
            for sep in ['。', '！', '？', '.\n', '!\n', '?\n', '\n\n', '. ', '! ', '? ']:
                last_sep = window.rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Next chunk starts at overlap position
        start = end - overlap if end < len(text) else len(text)

    return chunks
