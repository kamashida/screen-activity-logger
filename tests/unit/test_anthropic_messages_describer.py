"""Anthropic Messages APIの送信形式を実サーバーなしで検証する。"""

import base64
from pathlib import Path

from screen_activity_logger.domain.models import Frame, OcrText, VideoTimestamp
from screen_activity_logger.infrastructure.anthropic_messages_describer import (
    AnthropicMessagesSceneDescriber,
)


class FakeResponse:
    def raise_for_status(self) -> None: ...

    def json(self) -> dict:
        return {
            "content": [
                {"type": "text", "text": '{"app_guess":"Excel","action":"編集中"}'},
            ]
        }


def _frame(tmp_path: Path) -> Frame:
    png = tmp_path / "frame.png"
    png.write_bytes(b"\x89PNG-fake")
    return Frame(
        timestamp=VideoTimestamp(seconds=12.0), path=png, is_keyframe=True
    )


def test_sends_anthropic_messages_image_and_headers(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    describer = AnthropicMessagesSceneDescriber(
        model="claude-sonnet-test",
        base_url="https://api.anthropic.com/v1",
        api_key="secret",
    )

    result = describer.describe(_frame(tmp_path), OcrText(
        timestamp=VideoTimestamp(seconds=12.0), lines=("Excel",)
    ))

    call = calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"] == {
        "x-api-key": "secret",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    content = call["json"]["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["data"] == base64.b64encode(b"\x89PNG-fake").decode()
    assert content[1]["type"] == "text"
    assert result.app_guess == "Excel"
    assert result.action == "編集中"
