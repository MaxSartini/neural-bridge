"""
Market Data Consolidator

Consolidates metric/OHLCV entities into asset-level entities with rich temporal context.
Extracts time-series statistics and price history for agent reference.

Purpose: Transform granular metric nodes (TICKER.Open, TICKER.High, etc.) into
single asset entities (TICKER) with consolidated price history and context.

Loads complete OHLCV data from CSV files so agents get full historical data for
serious analysis (not just snapshots).

IMPORTANT: This module is fully topic-agnostic. No hardcoded tickers, price ranges,
sectors, or asset names. All identity comes from the filenames the user uploaded and
the project.json mapping that links original filenames to on-disk hash filenames.
"""

import re
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger('neural_bridge.market_data_consolidator')


@dataclass
class AssetTimeSeriesContext:
    """Time-series context for an asset (stock, index, commodity, or any CSV-backed entity)"""
    asset_id: str          # The filename stem, e.g. "BHG.JO" or "weather_london"
    asset_label: str       # Human-readable label (same as asset_id unless overridden)
    source_file: str       # Original filename the user uploaded
    latest_close: Optional[float] = None
    latest_date: Optional[str] = None
    price_history: List[Dict[str, Any]] = None
    trend_direction: Optional[str] = None    # "up", "down", "sideways"
    trend_pct_change: Optional[float] = None
    volatility_regime: Optional[str] = None  # "high", "normal", "low"
    key_levels: Optional[Dict[str, float]] = None
    volume_regime: Optional[str] = None      # "elevated", "normal", "compressed"
    column_names: Optional[List[str]] = None # All columns found in the CSV
    record_count: int = 0
    
    # Universal Statistical Metrics
    rolling_mean_short: Optional[float] = None
    rolling_mean_long: Optional[float] = None
    historical_z_score: Optional[float] = None
    all_time_high: Optional[float] = None
    all_time_low: Optional[float] = None

    def to_context_string(self) -> str:
        """Generate comprehensive, mathematically strict context for LLM agent reference."""
        lines = [
            f"=== STATISTICAL DATA REPORT: {self.asset_label} (source: {self.source_file}) ===",
            f"Total Historical Data Span: {self.record_count} data points",
            ""
        ]

        # Data regime and absolute truth summary
        lines.append("--- CURRENT STATE (AS OF FINAL RECORD) ---")
        if self.latest_date:
            lines.append(f"Date/Time: {self.latest_date}")
        if self.latest_close is not None:
            lines.append(f"Primary Value: {self.latest_close:.4g}")
        if self.trend_direction and self.trend_pct_change is not None:
            lines.append(f"Primary Trajectory: {self.trend_direction.upper()} ({self.trend_pct_change:+.2f}%)")
        
        lines.append("\n--- DERIVED STATISTICAL METRICS (ABSOLUTE REALITY) ---")
        if self.historical_z_score is not None:
            deviation_state = "Extreme Positive" if self.historical_z_score > 2 else "Extreme Negative" if self.historical_z_score < -2 else "Within Normal Variance"
            lines.append(f"Historical Z-Score (vs mean): {self.historical_z_score:+.2f} [{deviation_state}]")
            
        if self.rolling_mean_short is not None and self.rolling_mean_long is not None:
            trend_state = "Accelerating Upward" if self.rolling_mean_short > self.rolling_mean_long and self.latest_close and self.latest_close > self.rolling_mean_short else "Accelerating Downward" if self.rolling_mean_short < self.rolling_mean_long and self.latest_close and self.latest_close < self.rolling_mean_short else "Mean-Reverting / Mixed"
            lines.append(f"Short-Term Mean (20-pd): {self.rolling_mean_short:.4g} | Long-Term Mean (50-pd): {self.rolling_mean_long:.4g} [{trend_state}]")
            
        if self.all_time_high is not None and self.all_time_low is not None:
            lines.append(f"All-Time Range: Min={self.all_time_low:.4g} | Max={self.all_time_high:.4g}")
            
        if self.volatility_regime:
            lines.append(f"Volatility/Variance Regime: {self.volatility_regime.upper()}")
        if self.volume_regime:
            lines.append(f"Activity/Volume Regime: {self.volume_regime.upper()}")
            
        if self.key_levels:
            if "support" in self.key_levels:
                lines.append(f"Lower Bound (Support Tier): {self.key_levels['support']:.4g}")
            if "resistance" in self.key_levels:
                lines.append(f"Upper Bound (Resistance Tier): {self.key_levels['resistance']:.4g}")

        # Provide strict recent momentum (Last 30 days maximum to prevent token bloat)
        if self.price_history and len(self.price_history) > 0:
            lines.append("\n--- RECENT ACTIVITY LOG (LAST 30 POINTS) ---")
            recent_points = self.price_history[-30:]
            header_cols = [k for k in recent_points[0].keys()][:6] # Take up to 6 columns to prevent width bloat
            lines.append(",".join(header_cols))
            for point in recent_points:
                row_vals = [str(point.get(col, "N/A")) for col in header_cols]
                lines.append(",".join(row_vals))

        return "\n".join(lines)


# Keep old name as alias for backward compat within this module
StockTimeSeriesContext = AssetTimeSeriesContext


class MarketDataConsolidator:
    """
    Consolidate metric entities into asset-level entities with context.

    Fully topic-agnostic: derives all identity from the user's uploaded filenames
    and the project.json mapping. Never guesses or uses heuristics.
    """

    def __init__(self, project_dir: Optional[str] = None):
        """Initialize with optional project directory for CSV loading"""
        self.project_dir = project_dir
        self._file_mapping: Optional[Dict[str, str]] = None  # stem → saved_filename

    # ------------------------------------------------------------------
    #  File mapping: original_filename ←→ saved (hash) filename on disk
    # ------------------------------------------------------------------

    def _load_file_mapping(self) -> Dict[str, str]:
        """
        Load the original_filename → saved_filename mapping from project.json.

        Returns:
            Dict mapping filename_stem (no extension) → saved_filename on disk.
            Empty dict if project.json doesn't exist or has no file entries.
        """
        if self._file_mapping is not None:
            return self._file_mapping

        self._file_mapping = {}

        if not self.project_dir:
            return self._file_mapping

        project_json_path = os.path.join(self.project_dir, "project.json")
        if not os.path.exists(project_json_path):
            logger.warning(f"project.json not found at {project_json_path}")
            return self._file_mapping

        try:
            with open(project_json_path, 'r') as f:
                project_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read project.json: {e}")
            return self._file_mapping

        files_list = project_data.get("files", [])
        if not files_list:
            logger.warning("project.json has empty files list — CSV matching will fail")
            return self._file_mapping

        for entry in files_list:
            original = entry.get("filename", "")
            saved = entry.get("saved_filename", "")

            if not original:
                continue

            stem = os.path.splitext(original)[0]

            if saved:
                # Direct mapping available (new-style project.json)
                self._file_mapping[stem] = saved
                logger.debug(f"File mapping: '{stem}' → '{saved}'")
            else:
                # Legacy project.json without saved_filename — try size-based matching
                size = entry.get("size")
                if size is not None:
                    matched = self._match_by_size(original, size)
                    if matched:
                        self._file_mapping[stem] = matched
                        logger.info(f"File mapping (size-match): '{stem}' → '{matched}' (size={size})")
                    else:
                        logger.warning(f"Cannot map '{original}' — no saved_filename and size-match failed")
                else:
                    logger.warning(f"Cannot map '{original}' — no saved_filename and no size info")

        logger.info(f"Loaded {len(self._file_mapping)} file mappings from project.json")
        return self._file_mapping

    def _match_by_size(self, original_filename: str, expected_size: int) -> Optional[str]:
        """
        Legacy fallback: match a file by its exact byte size.
        Only used when project.json has filename+size but no saved_filename.

        Returns the matching on-disk filename, or None if ambiguous/not found.
        """
        files_dir = os.path.join(self.project_dir, "files")
        if not os.path.isdir(files_dir):
            return None

        ext = os.path.splitext(original_filename)[1].lower()
        candidates = [
            f for f in os.listdir(files_dir)
            if f.lower().endswith(ext)
        ]

        exact_matches = []
        for candidate in candidates:
            candidate_path = os.path.join(files_dir, candidate)
            try:
                actual_size = os.path.getsize(candidate_path)
                if actual_size == expected_size:
                    exact_matches.append(candidate)
            except OSError:
                continue

        if len(exact_matches) == 1:
            return exact_matches[0]
        elif len(exact_matches) > 1:
            # Ambiguous — multiple files with same size. Refuse to guess.
            logger.warning(
                f"Ambiguous size match for '{original_filename}' (size={expected_size}): "
                f"{exact_matches} — refusing to guess"
            )
            return None
        else:
            return None

    # ------------------------------------------------------------------
    #  Entity name parsing (generic, not hardcoded to any naming scheme)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_asset_id_from_entity(entity_name: str) -> Optional[str]:
        """
        Extract the asset identifier (filename stem) from an entity name.

        Entity names are structured as: {filename_stem}.{column_name}
        e.g. "BHG.JO.Close" → "BHG.JO", "weather_london.Temperature" → "weather_london"

        The filename stem is everything up to (but not including) the last dot-segment,
        where the last dot-segment matches a known OHLCV/data column name pattern.

        For robustness, we split on the LAST dot and check whether the trailing part
        looks like a column name (alpha characters). If not, the whole string is treated
        as the asset ID.
        """
        if not entity_name:
            return None

        # Split from the right on '.' — the last segment is the column name
        # e.g. "BHG.JO.Close" → ("BHG.JO", "Close")
        # e.g. "weather_london.Temperature" → ("weather_london", "Temperature")
        last_dot = entity_name.rfind(".")
        if last_dot <= 0:
            # No dot or dot at position 0 — entity name IS the asset id
            return entity_name

        prefix = entity_name[:last_dot]
        suffix = entity_name[last_dot + 1:]

        # The suffix should look like a column name (alphabetic, possibly with underscores)
        # If it looks like part of a domain/ticker (e.g. "JO" in "BHG.JO"), we need
        # to check whether the full name minus the suffix matches a known file stem.
        # But since we don't hardcode knowledge here, we rely on the caller to
        # validate against the file mapping.
        if suffix and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', suffix):
            return prefix

        return entity_name

    @staticmethod
    def extract_field_type(entity_name: str) -> Optional[str]:
        """
        Extract field/column type from entity name.
        e.g. "BHG.JO.Close" → "close", "weather_london.Temperature" → "temperature"
        """
        last_dot = entity_name.rfind(".")
        if last_dot <= 0:
            return None
        suffix = entity_name[last_dot + 1:]
        if suffix and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', suffix):
            return suffix.lower()
        return None

    # ------------------------------------------------------------------
    #  Main consolidation pipeline
    # ------------------------------------------------------------------

    def consolidate_entities(
        self,
        entities: List[Any]  # List of EntityNode
    ) -> Dict[str, AssetTimeSeriesContext]:
        """
        Consolidate metric entities into asset-level contexts.
        Also incorporates full CSV data history if available.

        Args:
            entities: List of EntityNode objects (from entity_reader)

        Returns:
            Dict mapping asset_id → AssetTimeSeriesContext (with full data history)
        """
        # Load file mapping FIRST so we know which stems are valid
        file_mapping = self._load_file_mapping()

        # Group entities by asset
        asset_data: Dict[str, Dict[str, Any]] = {}

        for entity in entities:
            # Try to extract asset_id, validating against the file mapping
            candidate_id = self.extract_asset_id_from_entity(entity.name)

            if candidate_id and candidate_id in file_mapping:
                asset_id = candidate_id
            else:
                # The simple rfind('.') split didn't produce a known stem.
                # Try progressively shorter prefixes to handle multi-dot names
                # e.g. "A.B.C.D" → try "A.B.C", then "A.B", then "A"
                asset_id = None
                parts = entity.name.split(".")
                for i in range(len(parts) - 1, 0, -1):
                    prefix = ".".join(parts[:i])
                    if prefix in file_mapping:
                        asset_id = prefix
                        break

                if not asset_id:
                    # No prefix matches a known file stem — skip this entity
                    logger.debug(
                        f"Entity '{entity.name}' does not match any uploaded file stem "
                        f"(known stems: {list(file_mapping.keys())})"
                    )
                    continue

            if asset_id not in asset_data:
                asset_data[asset_id] = {
                    "metrics": {},
                    "data_summaries": []
                }

            field_type = self.extract_field_type(entity.name)
            if field_type:
                asset_data[asset_id]["metrics"][field_type] = entity

            if entity.data_summary:
                asset_data[asset_id]["data_summaries"].append(entity.data_summary)

        logger.info(f"Grouped entities into {len(asset_data)} assets: {list(asset_data.keys())}")

        # Load CSV data for matched assets
        csv_data = self._load_csv_for_assets(asset_data.keys(), file_mapping)

        # Build context objects
        consolidated = {}
        for asset_id, data in asset_data.items():
            csv_records = csv_data.get(asset_id)
            original_filename = self._get_original_filename(asset_id, file_mapping)
            context = self._build_asset_context(asset_id, original_filename, data, csv_records)
            consolidated[asset_id] = context

            csv_count = len(csv_records) if csv_records else 0
            logger.info(
                f"Consolidated '{asset_id}': {len(data['metrics'])} metric fields, "
                f"{csv_count} CSV records, source='{original_filename}'"
            )

        return consolidated

    def _get_original_filename(self, asset_id: str, file_mapping: Dict[str, str]) -> str:
        """Recover the original filename from the asset_id (stem) + known extensions."""
        if not self.project_dir:
            return asset_id

        project_json_path = os.path.join(self.project_dir, "project.json")
        try:
            with open(project_json_path, 'r') as f:
                project_data = json.load(f)
            for entry in project_data.get("files", []):
                stem = os.path.splitext(entry.get("filename", ""))[0]
                if stem == asset_id:
                    return entry["filename"]
        except Exception:
            pass

        return asset_id

    # ------------------------------------------------------------------
    #  CSV loading — deterministic, uses project.json mapping only
    # ------------------------------------------------------------------

    def _load_csv_for_assets(
        self,
        asset_ids,
        file_mapping: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load CSV files for the given assets using the deterministic file mapping.

        No guessing. No heuristics. If the mapping doesn't exist, the asset
        gets no CSV data and we log an explicit warning.
        """
        if not self.project_dir:
            return {}

        files_dir = os.path.join(self.project_dir, "files")
        if not os.path.isdir(files_dir):
            logger.warning(f"Files directory not found: {files_dir}")
            return {}

        result = {}
        for asset_id in asset_ids:
            saved_filename = file_mapping.get(asset_id)
            if not saved_filename:
                logger.warning(
                    f"No file mapping for asset '{asset_id}' — CSV data unavailable. "
                    f"This means project.json is missing the saved_filename for this file."
                )
                continue

            csv_path = os.path.join(files_dir, saved_filename)
            if not os.path.isfile(csv_path):
                logger.error(
                    f"Mapped file '{saved_filename}' for asset '{asset_id}' does not exist at {csv_path}"
                )
                continue

            records = self._load_csv_file(csv_path, asset_id)
            if records:
                result[asset_id] = records
                logger.info(f"Loaded {len(records)} records for '{asset_id}' from '{saved_filename}'")
            else:
                logger.warning(f"No valid records in '{saved_filename}' for asset '{asset_id}'")

        return result

    def _load_csv_file(self, csv_path: str, asset_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load a single CSV file and extract all records. Returns all columns found."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)

            # Standardize column names (lowercase, stripped)
            df.columns = [col.lower().strip() for col in df.columns]

            if len(df) == 0:
                return None

            # Extract all columns into records (not just OHLCV) without the
            # slow pandas row iterator.
            records = df.astype(object).where(pd.notna(df), None).to_dict("records")
            for record in records:
                for col, val in list(record.items()):
                    if hasattr(val, "item"):
                        try:
                            record[col] = val.item()
                        except Exception:
                            record[col] = str(val)

            return records if records else None

        except Exception as e:
            logger.error(f"Failed to load CSV '{csv_path}' for asset '{asset_id}': {e}")
            return None

    # ------------------------------------------------------------------
    #  Context building — derives everything from the actual data
    # ------------------------------------------------------------------

    def _build_asset_context(
        self,
        asset_id: str,
        original_filename: str,
        data: Dict[str, Any],
        csv_records: Optional[List[Dict]] = None
    ) -> AssetTimeSeriesContext:
        """Build context for a single asset. All stats derived from data, nothing hardcoded."""
        context = AssetTimeSeriesContext(
            asset_id=asset_id,
            asset_label=asset_id,
            source_file=original_filename,
        )

        if csv_records and len(csv_records) > 0:
            context.price_history = csv_records
            context.record_count = len(csv_records)
            context.column_names = list(csv_records[0].keys()) if csv_records else []

            # Find the primary value column: prefer "close", then "value", then first numeric
            value_col = self._find_primary_value_column(csv_records)
            date_col = self._find_date_column(csv_records)
            volume_col = self._find_volume_column(csv_records)

            if value_col:
                values = [r[value_col] for r in csv_records if r.get(value_col) is not None]
                if values:
                    context.latest_close = values[-1] if isinstance(values[-1], (int, float)) else None

                    if len(values) >= 2 and isinstance(values[0], (int, float)) and isinstance(values[-1], (int, float)):
                        first_val = values[0]
                        last_val = values[-1]
                        if first_val != 0:
                            pct_change = ((last_val - first_val) / first_val) * 100
                            context.trend_pct_change = round(pct_change, 2)
                            if last_val > first_val:
                                context.trend_direction = "up"
                            elif last_val < first_val:
                                context.trend_direction = "down"
                            else:
                                context.trend_direction = "flat"
                                
                    # --- Universal Statistical Calculations via Pandas ---
                    try:
                        import pandas as pd
                        import numpy as np
                        
                        series = pd.Series([v for v in values if isinstance(v, (int, float))])
                        if not series.empty:
                            context.all_time_high = float(series.max())
                            context.all_time_low = float(series.min())
                            
                            # Z-score (Distance from all-time mean in standard deviations)
                            mean = series.mean()
                            std = series.std()
                            if std and std > 0:
                                current_val = float(series.iloc[-1])
                                context.historical_z_score = float((current_val - mean) / std)
                            
                            # Rolling Means (Agnostic short/long term momentum)
                            # Using 20 and 50 periods, safe-guarding against short datasets
                            if len(series) >= 20:
                                context.rolling_mean_short = float(series.rolling(window=20).mean().iloc[-1])
                            if len(series) >= 50:
                                context.rolling_mean_long = float(series.rolling(window=50).mean().iloc[-1])
                    except Exception as e:
                        logger.warning(f"Failed to calculate advanced stats for {asset_id}: {e}")

                    # Volatility from last 30 points
                    self._compute_volatility(context, values)

                    # Key levels from full history
                    self._compute_key_levels(context, values)

            if date_col:
                last_record = csv_records[-1]
                context.latest_date = str(last_record.get(date_col, ""))

            # Volume regime
            if volume_col:
                self._compute_volume_regime(context, csv_records, volume_col)

            return context

        # Fallback to entity data summaries if no CSV data
        return self._build_context_from_entity_data(context, data)

    @staticmethod
    def _find_primary_value_column(records: List[Dict]) -> Optional[str]:
        """Find the primary numeric value column. Prefer 'close', 'value', then first numeric."""
        if not records:
            return None
        sample = records[0]
        # Priority order
        for name in ["close", "value", "price", "amount", "total"]:
            if name in sample and isinstance(sample[name], (int, float)):
                return name
        # Fall back to first numeric column that isn't date/volume
        skip = {"date", "datetime", "time", "timestamp", "volume", "vol"}
        for key, val in sample.items():
            if key.lower() not in skip and isinstance(val, (int, float)):
                return key
        return None

    @staticmethod
    def _find_date_column(records: List[Dict]) -> Optional[str]:
        """Find the date/time column."""
        if not records:
            return None
        sample = records[0]
        for name in ["date", "datetime", "time", "timestamp", "period"]:
            if name in sample:
                return name
        return None

    @staticmethod
    def _find_volume_column(records: List[Dict]) -> Optional[str]:
        """Find a volume column."""
        if not records:
            return None
        sample = records[0]
        for name in ["volume", "vol", "quantity", "count"]:
            if name in sample and isinstance(sample[name], (int, float)):
                return name
        return None

    @staticmethod
    def _compute_volatility(context: AssetTimeSeriesContext, values: List):
        """Compute volatility regime from recent values."""
        try:
            import statistics
            recent = [v for v in values[-30:] if isinstance(v, (int, float))]
            if len(recent) > 1:
                mean = statistics.mean(recent)
                if mean != 0:
                    volatility = (statistics.stdev(recent) / abs(mean)) * 100
                    if volatility > 3:
                        context.volatility_regime = "high"
                    elif volatility > 1.5:
                        context.volatility_regime = "normal"
                    else:
                        context.volatility_regime = "low"
        except Exception:
            pass

    @staticmethod
    def _compute_key_levels(context: AssetTimeSeriesContext, values: List):
        """Compute support/resistance from historical data."""
        numeric = [v for v in values if isinstance(v, (int, float))]
        if len(numeric) >= 5:
            sorted_vals = sorted(numeric)
            idx_20pct = len(sorted_vals) // 5
            idx_80pct = len(sorted_vals) - idx_20pct
            context.key_levels = {
                "support": sorted_vals[idx_20pct],
                "resistance": sorted_vals[min(idx_80pct, len(sorted_vals) - 1)],
            }

    @staticmethod
    def _compute_volume_regime(context: AssetTimeSeriesContext, records: List[Dict], volume_col: str):
        """Compute volume regime."""
        try:
            volumes = [r[volume_col] for r in records if isinstance(r.get(volume_col), (int, float))]
            if len(volumes) >= 2:
                import statistics
                avg = statistics.mean(volumes)
                current = volumes[-1]
                if avg > 0:
                    if current > avg * 1.3:
                        context.volume_regime = "elevated"
                    elif current < avg * 0.7:
                        context.volume_regime = "compressed"
                    else:
                        context.volume_regime = "normal"
        except Exception:
            pass

    @staticmethod
    def _build_context_from_entity_data(
        context: AssetTimeSeriesContext,
        data: Dict[str, Any]
    ) -> AssetTimeSeriesContext:
        """Fallback: build context from entity data_summaries when CSV isn't available."""
        close_entity = data["metrics"].get("close") or data["metrics"].get("value")
        if close_entity and close_entity.data_summary:
            ds = close_entity.data_summary
            context.latest_close = ds.get("latest_value")
            context.latest_date = close_entity.summary

        if data["data_summaries"]:
            first_summary = data["data_summaries"][0]
            if isinstance(first_summary, dict):
                context.trend_direction = first_summary.get("trend", "unknown")
                context.trend_pct_change = first_summary.get("pct_change_total", 0)

        volume_entity = data["metrics"].get("volume")
        if volume_entity and volume_entity.data_summary:
            vol_summary = volume_entity.data_summary
            current_vol = vol_summary.get("latest_value", 0)
            avg_vol = vol_summary.get("mean", 0)
            if avg_vol > 0:
                if current_vol > avg_vol * 1.3:
                    context.volume_regime = "elevated"
                elif current_vol < avg_vol * 0.7:
                    context.volume_regime = "compressed"
                else:
                    context.volume_regime = "normal"

        low_entity = data["metrics"].get("low")
        high_entity = data["metrics"].get("high")
        if low_entity and high_entity:
            try:
                low_val = low_entity.data_summary.get("latest_value") if low_entity.data_summary else None
                high_val = high_entity.data_summary.get("latest_value") if high_entity.data_summary else None
                if low_val is not None and high_val is not None:
                    context.key_levels = {"support": low_val, "resistance": high_val}
            except Exception:
                pass

        if data["data_summaries"]:
            history = []
            for summary in data["data_summaries"][:20]:
                if isinstance(summary, dict):
                    history.append({
                        "date": summary.get("metric_name", "Unknown"),
                        "close": summary.get("latest_value"),
                        "volume": summary.get("volume")
                    })
            context.price_history = history if history else None

        if data["data_summaries"]:
            first = data["data_summaries"][0]
            if isinstance(first, dict) and first.get("std_dev"):
                std_dev = first.get("std_dev", 0)
                mean = first.get("mean", 1)
                if mean != 0:
                    ratio = (std_dev / abs(mean)) * 100
                    if ratio > 3:
                        context.volatility_regime = "high"
                    elif ratio > 1.5:
                        context.volatility_regime = "normal"
                    else:
                        context.volatility_regime = "low"

        return context

    # ------------------------------------------------------------------
    #  Entity node creation
    # ------------------------------------------------------------------

    def create_stock_entity_node(
        self,
        ticker: str,
        context: AssetTimeSeriesContext,
        uuid: Optional[str] = None
    ) -> 'EntityNode':
        """
        Create an EntityNode for an asset with consolidated context.

        Args:
            ticker: Asset identifier (kept as 'ticker' param for backward compat)
            context: AssetTimeSeriesContext
            uuid: Optional UUID

        Returns:
            EntityNode with consolidated asset context
        """
        from .entity_reader import EntityNode
        import uuid as uuid_lib

        if not uuid:
            uuid = str(uuid_lib.uuid4())

        entity = EntityNode(
            uuid=uuid,
            name=ticker,
            labels=["Asset", "DataSource"],
            summary=f"{context.asset_label} — {context.record_count} data points from {context.source_file}",
            attributes={
                "asset_id": context.asset_id,
                "source_file": context.source_file,
                "latest_value": context.latest_close,
                "latest_date": context.latest_date,
                "trend": context.trend_direction,
                "trend_pct_change": context.trend_pct_change,
                "volatility": context.volatility_regime,
                "volume_regime": context.volume_regime,
                "record_count": context.record_count,
                "columns": context.column_names,
            },
            data_summary=context.to_context_string(),
        )

        return entity

    # ------------------------------------------------------------------
    #  Legacy convenience: old public method name
    # ------------------------------------------------------------------

    def load_csv_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy method — loads all CSV files via the project.json mapping."""
        file_mapping = self._load_file_mapping()
        return self._load_csv_for_assets(file_mapping.keys(), file_mapping)
