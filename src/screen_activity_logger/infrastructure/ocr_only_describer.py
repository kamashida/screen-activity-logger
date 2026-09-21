"""VLMを使わずOCRだけで画面ログを作る明示的な基線アダプタ。"""

from __future__ import annotations

from screen_activity_logger.domain.models import ActivityDescription, Frame, OcrText


class OcrOnlySceneDescriber:
    """外部送信もローカルVLMも行わず、OCR結果を作業ログに残す。"""

    def describe(
        self, frame: Frame, ocr: OcrText, speech: tuple[str, ...] = ()
    ) -> ActivityDescription:
        lines = ocr.normalized_lines()
        if lines:
            preview = " / ".join(lines[:3])
            if len(preview) > 160:
                preview = preview[:157].rstrip() + "..."
            action = f"OCR記録: {preview}"
        else:
            action = "画面を記録"
        return ActivityDescription(
            timestamp=frame.timestamp,
            action=action,
            app_guess=None,
        )
