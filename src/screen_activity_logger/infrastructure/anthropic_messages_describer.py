"""Anthropic Messages APIによる画像説明アダプタ。"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

from screen_activity_logger.domain.models import (
    ActivityDescription,
    Frame,
    OcrText,
)
from screen_activity_logger.infrastructure.vlm_common import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_ATTEMPTS,
    SLOW_CALL_THRESHOLD_SECONDS,
    build_anthropic_headers,
    build_prompt,
    extract_anthropic_text,
    is_retryable,
    parse_fields,
)

DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1"
_MAX_TOKENS = 512


class AnthropicMessagesSceneDescriber:
    """Anthropic Messages APIでフレームを説明する。"""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_ANTHROPIC_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        api_key: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Anthropic APIキーが必要です")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._headers = build_anthropic_headers(api_key)

    def describe(
        self, frame: Frame, ocr: OcrText, speech: tuple[str, ...] = ()
    ) -> ActivityDescription:
        image_path = Path(frame.path)
        image_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": build_prompt(ocr, speech)},
                    ],
                }
            ],
        }
        content = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                content = self._post_messages(payload)
            except Exception as error:  # noqa: BLE001 — バッチ継続を優先
                elapsed = time.monotonic() - started
                print(
                    f"VLM呼び出し失敗 t={frame.timestamp}"
                    f" attempt={attempt}/{MAX_ATTEMPTS} {elapsed:.1f}s:"
                    f" {type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt < MAX_ATTEMPTS and is_retryable(error):
                    continue
                return ActivityDescription(
                    timestamp=frame.timestamp,
                    action=f"（VLM呼び出し失敗: {type(error).__name__}）",
                    app_guess=None,
                )
            if not content and attempt < MAX_ATTEMPTS:
                print(
                    f"VLM空応答（リトライ） t={frame.timestamp}"
                    f" attempt={attempt}/{MAX_ATTEMPTS}",
                    flush=True,
                )
                continue
            elapsed = time.monotonic() - started
            slow = " SLOW" if elapsed > SLOW_CALL_THRESHOLD_SECONDS else ""
            print(
                f"VLM推論 t={frame.timestamp} attempt={attempt}"
                f" {elapsed:.1f}s{slow}",
                flush=True,
            )
            break

        fields = parse_fields(content)
        return ActivityDescription(
            timestamp=frame.timestamp,
            action=fields["action"],
            app_guess=fields["app_guess"],
            resource=fields["resource"],
            location=fields["location"],
            focus=fields["focus"],
        )

    def _post_messages(self, payload: dict[str, Any]) -> str:
        import httpx  # 遅延import（ローカル実行時の追加依存を避ける）

        response = httpx.post(
            f"{self._base_url}/messages",
            json=payload,
            headers=self._headers,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return extract_anthropic_text(response.json())
