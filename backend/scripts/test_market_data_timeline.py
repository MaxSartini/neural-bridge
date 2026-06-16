"""Smoke-test chronology-safe CSV timeline rendering."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_parallel_simulation import MarketDataTimeline  # noqa: E402
from app.services.simulation_manager import _contains_ohlcv_csv  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        project_dir = Path(temporary)
        files_dir = project_dir / "files"
        files_dir.mkdir()
        saved_name = "hashed.csv"
        rows = ["Date,Open,High,Low,Close,Volume"]
        for day in range(1, 11):
            rows.append(f"2026-01-{day:02d},{day},{day},{day},{day},100")
        (files_dir / saved_name).write_text("\n".join(rows), encoding="utf-8")
        (project_dir / "project.json").write_text(
            json.dumps(
                {"files": [{"saved_filename": saved_name, "filename": "REAL.JO.csv"}]}
            ),
            encoding="utf-8",
        )
        assert _contains_ohlcv_csv(str(files_dir))
        social_dir = project_dir / "social"
        social_dir.mkdir()
        (social_dir / "posts.csv").write_text("timestamp,text\n1,hello\n", encoding="utf-8")
        assert not _contains_ohlcv_csv(str(social_dir))

        timeline = MarketDataTimeline(str(files_dir))
        block = timeline.get_terminal(0, 1)
        assert timeline.available
        assert "REAL.JO" in block and "hashed" not in block
        assert "2026-01-01" in block and "2026-01-10" in block
        assert "displayed 8 evenly spaced rows from 10 total" in block
    print({"market_data_timeline_ok": True})


if __name__ == "__main__":
    main()
