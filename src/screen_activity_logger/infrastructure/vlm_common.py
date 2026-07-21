"""VLMバックエンド共通部（プロンプト・応答パース・リトライ判定、Issue #21）。

Ollama / OpenAI互換（vllm-mlx）の両アダプタで、プロンプトと出力解釈の仕様を
共有し、バックエンド間の挙動差を構造的に排除する（asr_segmentsと同じ発想）。
"""

from __future__ import annotations

import json
import re
from ipaddress import ip_address
from urllib.parse import urlsplit

from screen_activity_logger.domain.models import OcrText

PROMPT_TEMPLATE = """あなたはPC作業の記録係です。このスクリーンショットについて日本語で記録します:
1) app_guess: 使用中のアプリ（Excel/PowerPoint/Chrome/VS Code等）
2) resource: 開いているファイル名・ページタイトル・文書名。タイトルバーやタブに明確に読み取れる場合のみ。読み取れない・確信がない場合はnull。画面の説明文（"Web page titled..."等）や意味不明な文字断片は書かない
3) location: リソース内の位置（シート名・スライド番号・ページ番号・見出し・URLパス等）
4) focus: ユーザーが画面のどこを見て何を判断していそうか（カーソル位置・選択状態・強調から推測）
5) action: 今している操作の説明（1〜2文）

品質基準（Issue #24）: actionは、本人が後から読み返して作業内容を思い出せ、
第三者が業務の流れを追える水準で書く。主語（「ユーザーは」等）を省き、常体（「〜している」）で統一する。

参考: この画面からOCRで抽出されたテキスト:
{ocr_text}

参考: この時間帯にユーザーが話していた内容（音声認識）:
{speech_text}

次のJSONのみを出力してください（説明文・コードフェンス不要。不明な項目はnull）:
{{"app_guess": "...", "resource": "...", "location": "...", "focus": "...", "action": "..."}}"""

_CODE_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\n|\n?```$")

FALLBACK_ACTION = "（この画面の説明を生成できませんでした）"

# VLM呼び出しのタイムアウト（秒）。実測: 応答が宙に浮くとsock_recvで
# 無限待ちになりバッチ全体が停止するため必須（実会議10本バッチで発覚）
DEFAULT_TIMEOUT_SECONDS = 300.0

# Anthropic Messages APIのバージョン固定値。API仕様変更時に一箇所だけ更新する。
ANTHROPIC_API_VERSION = "2023-06-01"

# タイムアウト時のリトライ上限（Issue #15。ハング・一時的負荷の救済）
MAX_ATTEMPTS = 2

# 推論時間テレメトリのSLOW警告閾値
SLOW_CALL_THRESHOLD_SECONDS = 60.0


def build_prompt(ocr: OcrText, speech: tuple[str, ...] = ()) -> str:
    lines = ocr.normalized_lines()
    ocr_text = "\n".join(lines) if lines else "(テキストなし)"
    speech_text = "\n".join(speech) if speech else "(発話なし)"
    return PROMPT_TEMPLATE.format(ocr_text=ocr_text, speech_text=speech_text)


def parse_fields(content: str) -> dict[str, str | None]:
    """VLM応答テキストを構造化フィールドに解釈する（コードフェンス耐性）。"""
    empty: dict[str, str | None] = {
        "app_guess": None,
        "resource": None,
        "location": None,
        "focus": None,
    }
    cleaned = _CODE_FENCE_PATTERN.sub("", content.strip()).strip()
    if not cleaned:
        return {**empty, "action": FALLBACK_ACTION}
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        if cleaned.startswith("{"):
            # 閉じ括弧欠け等の不正JSON: 生テキストをactionにせず
            # フィールドを正規表現で救出する（Issue #26 E2Eで顕在化）
            return _salvage_fields(cleaned)
        return {**empty, "action": cleaned}
    fields: dict[str, str | None] = {
        key: normalize_json_value(payload.get(key)) for key in empty
    }
    # actionも空値表現（null/"null"/"none"）を正規化する（Issue #3の実バグ:
    # 文字列"null"が素通りしてMarkdownに「null」と表示されていた）
    fields["action"] = (
        normalize_json_value(payload.get("action")) or FALLBACK_ACTION
    )
    return fields


_FIELD_PATTERN = re.compile(r'"(app_guess|resource|location|focus|action)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _salvage_fields(broken_json: str) -> dict[str, str | None]:
    """JSONとして壊れた応答からフィールド値を正規表現で拾う。"""
    fields: dict[str, str | None] = {
        "app_guess": None,
        "resource": None,
        "location": None,
        "focus": None,
    }
    action = None
    for match in _FIELD_PATTERN.finditer(broken_json):
        key, value = match.group(1), match.group(2)
        if key == "action":
            action = normalize_json_value(value)
        else:
            fields[key] = normalize_json_value(value)
    fields["action"] = action or FALLBACK_ACTION
    return fields


def normalize_json_value(value: object) -> str | None:
    """JSONの空値表現（null/"null"/"none"/空文字）をNoneに正規化する。"""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in ("null", "none"):
        return None
    return stripped


def build_auth_headers(api_key: str | None) -> dict[str, str] | None:
    """OpenAI互換API用のAuthorizationヘッダを構築する。

    api_key未指定（None/空文字）ならNoneを返す＝ヘッダを一切付けない
    （既存のローカルVLM無認証動作を不変に保つ）。
    """
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def build_anthropic_headers(api_key: str | None) -> dict[str, str] | None:
    """Anthropic Messages API用ヘッダを構築する。"""
    if not api_key:
        return None
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
    }


def extract_anthropic_text(payload: object) -> str:
    """Messages APIのcontentブロックからテキストだけを連結する。"""
    if not isinstance(payload, dict):
        return ""
    blocks = payload.get("content", ())
    if not isinstance(blocks, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local_url(url: str) -> bool:
    """ループバック宛URLかを判定する（外部送信警告・ガード用）。"""
    hostname = urlsplit(url).hostname
    if not hostname:
        return False
    normalized = hostname.lower().rstrip(".")
    if normalized in _LOCAL_HOSTS:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_vlm_url(url: str, *, allow_external: bool = False) -> None:
    """VLM接続先の形式と送信境界を検証する。

    ループバック以外はHTTPSかつ明示的な許可が必要。クエリ文字列へのAPIキー
    混入も防ぐため、query/fragment付きURLは拒否する。
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(
            f"VLM URLはhttp(s)://ホストの形式で指定してください: {url}"
        )
    if parts.username or parts.password:
        raise ValueError("VLM URLにユーザー名・パスワードは指定できません")
    if parts.query or parts.fragment:
        raise ValueError("VLM URLにquery/fragmentは指定できません（APIキー漏洩防止）")
    if is_local_url(url):
        return
    if parts.scheme != "https":
        raise ValueError("外部VLM接続はHTTPSが必須です")
    if not allow_external:
        raise ValueError(
            "外部VLMへのフレーム送信は既定で無効です。"
            " --allow-external-vlm を明示してください"
        )


def is_retryable(error: BaseException) -> bool:
    """タイムアウト系例外か（Issue #15のリトライ対象判定）。

    httpxを直接importせず例外MROのクラス名で判定する。
    """
    return any(
        cls.__name__ in ("TimeoutException", "TimeoutError")
        for cls in type(error).__mro__
    )
