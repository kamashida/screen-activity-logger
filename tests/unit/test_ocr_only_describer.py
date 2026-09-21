from pathlib import Path

from screen_activity_logger.domain.models import Frame, OcrText, VideoTimestamp
from screen_activity_logger.infrastructure.ocr_only_describer import (
    OcrOnlySceneDescriber,
)


def test_ocr_only_describer_keeps_a_short_ocr_preview() -> None:
    frame = Frame(VideoTimestamp(12.0), Path("frame.png"), True)
    result = OcrOnlySceneDescriber().describe(
        frame, OcrText(frame.timestamp, ("Excel", "見積書.xlsx", "長い補足"))
    )

    assert result.action == "OCR記録: Excel / 見積書.xlsx / 長い補足"
    assert result.app_guess is None


def test_ocr_only_describer_handles_empty_ocr() -> None:
    frame = Frame(VideoTimestamp(12.0), Path("frame.png"), True)

    result = OcrOnlySceneDescriber().describe(frame, OcrText(frame.timestamp, ()))

    assert result.action == "画面を記録"
