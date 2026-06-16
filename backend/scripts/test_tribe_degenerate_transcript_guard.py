"""Regression tests for pathological transcript handling in TRIBE adapter."""

import pandas as pd

from app.services.tribe_adapter import TribeAdapter


def test_repeated_zero_duration_transcript_is_dropped_but_video_audio_remain() -> None:
    adapter = TribeAdapter()
    rows = [
        {
            "type": "Video",
            "start": 0.0,
            "duration": 35.0,
            "timeline": "default",
            "subject": "default",
            "filepath": "clip.webm",
        },
        {
            "type": "Audio",
            "start": 0.0,
            "duration": 35.0,
            "timeline": "default",
            "subject": "default",
            "filepath": "clip.wav",
        },
        {
            "type": "Text",
            "start": 13.0,
            "duration": 22.0,
            "timeline": "default",
            "subject": "default",
            "text": " ".join(["True"] * 120),
            "context": " ".join(["True"] * 120),
        },
    ]
    rows.extend(
        {
            "type": "Word",
            "text": " True",
            "start": 13.0 + index * 0.01,
            "duration": 0.0,
            "timeline": "default",
            "subject": "default",
            "sentence": " ".join(["True"] * 120),
            "context": " ".join(["True"] * 120),
        }
        for index in range(120)
    )

    repaired = adapter._repair_text_context(pd.DataFrame(rows))

    assert set(repaired["type"]) == {"Audio", "Video"}
    quality = repaired.attrs["neural_bridge_quality"]
    assert quality["degenerate_text_dropped"] is True
    assert quality["degenerate_text_events_removed"] == 121
    assert quality["missing_text"] is True
    assert quality["missing_audio"] is False
    assert quality["missing_video"] is False


def test_normal_short_transcript_is_preserved_and_repaired() -> None:
    adapter = TribeAdapter()
    events = pd.DataFrame(
        [
            {
                "type": "Video",
                "start": 0.0,
                "duration": 10.0,
                "timeline": "default",
                "subject": "default",
            },
            {
                "type": "Audio",
                "start": 0.0,
                "duration": 10.0,
                "timeline": "default",
                "subject": "default",
            },
            {
                "type": "Word",
                "text": "markets",
                "start": 1.0,
                "duration": 0.0,
                "timeline": "default",
                "subject": "default",
                "sentence": "markets react quickly",
                "context": "",
            },
            {
                "type": "Word",
                "text": "react",
                "start": 1.4,
                "duration": 0.2,
                "timeline": "default",
                "subject": "default",
                "sentence": "markets react quickly",
                "context": "",
            },
        ]
    )

    repaired = adapter._repair_text_context(events)

    assert "Word" in set(repaired["type"])
    quality = repaired.attrs["neural_bridge_quality"]
    assert quality["degenerate_text_dropped"] is False
    assert quality["word_duration_repairs"] == 1
    assert quality["null_word_durations_after_repair"] == 0
