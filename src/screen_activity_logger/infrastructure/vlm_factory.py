"""VLMバックエンドの解決とアダプタ生成（Issue #21、asr_factoryと同型）。"""

from __future__ import annotations

from screen_activity_logger.application.ports import (
    SceneDescriber,
    SpeechSummarizer,
)
from screen_activity_logger.infrastructure.chat_summarizer import (
    AnthropicChatSummarizer,
    OllamaChatSummarizer,
    OpenAIChatSummarizer,
)
from screen_activity_logger.infrastructure.anthropic_messages_describer import (
    DEFAULT_ANTHROPIC_URL,
    AnthropicMessagesSceneDescriber,
)
from screen_activity_logger.infrastructure.ollama_describer import (
    OllamaSceneDescriber,
)
from screen_activity_logger.infrastructure.openai_chat_describer import (
    DEFAULT_VLLM_MODEL,
    DEFAULT_VLLM_URL,
    OpenAIChatSceneDescriber,
)
from screen_activity_logger.infrastructure.vlm_common import (
    DEFAULT_TIMEOUT_SECONDS,
    build_auth_headers,
    validate_vlm_url,
)

VLM_BACKENDS = ("ollama", "vllm-mlx")
VLM_PROVIDERS = ("local", "gemini", "anthropic")
DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def create_describer(
    backend: str,
    model: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    base_url: str | None = None,
    api_key: str | None = None,
    provider: str = "local",
) -> SceneDescriber:
    """解決済みバックエンドからSceneDescriberを生成する。

    vllm-mlx時はmodelがOllama形式（qwen3-vl:8b）ならMLX既定モデルに読み替える
    （タグ形式はOllama固有のため）。
    provider: local / gemini / anthropic。クラウドはプロバイダ固有の形式を使う。
    """
    if provider == "gemini":
        return OpenAIChatSceneDescriber(
            model=model,
            base_url=base_url or DEFAULT_GEMINI_URL,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            warmup=False,  # クラウドでウォームアップ課金を発生させない
        )
    if provider == "anthropic":
        return AnthropicMessagesSceneDescriber(
            model=model,
            base_url=base_url or DEFAULT_ANTHROPIC_URL,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )
    if provider != "local":
        raise ValueError(f"未知のVLMプロバイダ: {provider}")
    if backend == "ollama":
        return OllamaSceneDescriber(model=model, timeout_seconds=timeout_seconds)
    if backend == "vllm-mlx":
        return OpenAIChatSceneDescriber(
            model=_resolve_vllm_model(model),
            base_url=base_url or DEFAULT_VLLM_URL,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )
    raise ValueError(f"未知のVLMバックエンド: {backend}")


def ensure_backend_available(
    backend: str,
    base_url: str = DEFAULT_VLLM_URL,
    api_key: str | None = None,
) -> None:
    """VLMサーバーの到達性チェック（重い処理の前に親切なエラーで止める）。"""
    if backend == "ollama":
        # 重いOCRの後で初めて未起動に気づかないよう起動前に確認（4AIレビューR1）
        import ollama

        try:
            # 到達性チェック自体がハングしないよう短いtimeout（4AIレビューR2）
            ollama.Client(timeout=5.0).list()
        except Exception as error:  # noqa: BLE001
            raise ValueError(
                f"Ollamaサーバーに接続できません: {type(error).__name__}。"
                " ollama serve を起動するか、--vlm-backend vllm-mlx を検討してください"
            ) from error
        return
    import httpx

    headers = build_auth_headers(api_key)
    try:
        kwargs: dict = {"timeout": 5.0}
        if headers:
            kwargs["headers"] = headers
        response = httpx.get(f"{base_url.rstrip('/')}/models", **kwargs)
        if not 200 <= response.status_code < 300:
            response.raise_for_status()
    except Exception as error:  # noqa: BLE001
        raise ValueError(
            f"vllm-mlxサーバーに接続できません（{base_url}）: {type(error).__name__}。"
            " 起動例: vllm-mlx serve"
            f" {DEFAULT_VLLM_MODEL} --port 8991"
            "（導入: pip install vllm-mlx。README参照）"
        ) from error


def ensure_provider_available(
    backend: str,
    base_url: str,
    api_key: str | None = None,
    *,
    provider: str = "local",
    model: str | None = None,
    allow_external: bool = False,
) -> None:
    """選択プロバイダの事前確認。

    GeminiはOpenAI互換のモデル取得を確認する。Anthropicはモデル一覧APIを
    持たず、無駄な課金リクエストを避けるため、キーとURLの検証だけを行う。
    """
    # CLI以外から呼ばれても、クラウド・カスタムVLMへの送信境界を同じにする。
    validate_vlm_url(base_url, allow_external=allow_external)
    if provider == "local":
        ensure_backend_available(backend, base_url, api_key=api_key)
        return
    if provider not in VLM_PROVIDERS:
        raise ValueError(f"未知のVLMプロバイダ: {provider}")
    if not api_key:
        raise ValueError(f"{provider}のAPIキーが設定されていません")
    if not model:
        raise ValueError(f"{provider}では--modelの明示指定が必要です")
    if provider == "anthropic":
        return

    import httpx
    from urllib.parse import quote

    headers = build_auth_headers(api_key)
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models/{quote(model, safe='')}",
            headers=headers,
            timeout=5.0,
        )
        # OpenAI互換層でもモデル情報エンドポイント自体は未実装のことがある
        # （例: Vertex AIのopenapiエンドポイントは404を返すがchat/completionsは動く）。
        # 許容は404のみ。認証失敗（401/403）やサーバー障害・レート制限
        # （500/429等）は事前に止める（Codexレビュー指摘）。
        if response.status_code != 404:
            response.raise_for_status()
    except Exception as error:  # noqa: BLE001
        raise ValueError(
            f"{provider}のモデルに接続できません（{base_url} / {model}）: "
            f"{type(error).__name__}"
        ) from error


def create_summarizer(
    backend: str,
    model: str,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    api_key: str | None = None,
    provider: str = "local",
) -> SpeechSummarizer:
    """VLMと同一バックエンド・モデルで発話要旨アダプタを生成する（Issue #23）。

    モデル名の読み替え規則はcreate_describerと同一（挙動差の排除）。
    timeout_seconds未指定時はアダプタ既定（60s）。
    api_key: クラウド互換エンドポイント向けAuthorizationヘッダ用（未指定ならヘッダなし）。
    """
    kwargs = {} if timeout_seconds is None else {"timeout_seconds": timeout_seconds}
    if provider == "gemini":
        return OpenAIChatSummarizer(
            model=model, base_url=base_url or DEFAULT_GEMINI_URL,
            api_key=api_key, **kwargs
        )
    if provider == "anthropic":
        return AnthropicChatSummarizer(
            model=model, base_url=base_url or DEFAULT_ANTHROPIC_URL,
            api_key=api_key, **kwargs
        )
    if provider != "local":
        raise ValueError(f"未知のVLMプロバイダ: {provider}")
    if backend == "ollama":
        return OllamaChatSummarizer(model=model, **kwargs)
    if backend == "vllm-mlx":
        return OpenAIChatSummarizer(
            model=_resolve_vllm_model(model),
            base_url=base_url or DEFAULT_VLLM_URL,
            api_key=api_key,
            **kwargs,
        )
    raise ValueError(f"未知のVLMバックエンド: {backend}")


def _resolve_vllm_model(model: str) -> str:
    """Ollamaタグ形式（qwen3-vl:8b）のみMLX既定へ読み替える。

    HF形式（org/name。リビジョン等でコロンを含み得る）は素通し（4AIレビューR2）。
    """
    return DEFAULT_VLLM_MODEL if (":" in model and "/" not in model) else model
