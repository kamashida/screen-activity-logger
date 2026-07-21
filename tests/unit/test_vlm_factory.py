"""VLMファクトリのユニットテスト（Issue #21 M3: RED）。"""

import pytest

from screen_activity_logger.infrastructure.ollama_describer import (
    OllamaSceneDescriber,
)
from screen_activity_logger.infrastructure.anthropic_messages_describer import (
    AnthropicMessagesSceneDescriber,
)
from screen_activity_logger.infrastructure.openai_chat_describer import (
    DEFAULT_VLLM_MODEL,
    OpenAIChatSceneDescriber,
)
from screen_activity_logger.infrastructure.vlm_factory import (
    create_describer,
    ensure_backend_available,
)


class TestCreateDescriber:
    def test_creates_ollama(self) -> None:
        describer = create_describer("ollama", "qwen3-vl:8b", timeout_seconds=450.0)
        assert isinstance(describer, OllamaSceneDescriber)
        assert describer._timeout_seconds == 450.0

    def test_creates_vllm_mlx(self) -> None:
        describer = create_describer(
            "vllm-mlx", "mlx-community/Custom-Model",
            base_url="http://localhost:9000/v1",
        )
        assert isinstance(describer, OpenAIChatSceneDescriber)
        assert describer._model == "mlx-community/Custom-Model"
        assert describer._base_url == "http://localhost:9000/v1"

    def test_ollama_tag_model_is_translated_for_vllm(self) -> None:
        """qwen3-vl:8b（Ollamaタグ形式）はMLX既定モデルに読み替える。"""
        describer = create_describer("vllm-mlx", "qwen3-vl:8b")
        assert describer._model == DEFAULT_VLLM_MODEL

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            create_describer("unknown", "m")

    def test_gemini_uses_openai_compat_adapter_without_warmup(self) -> None:
        describer = create_describer(
            "ollama", "gemini-test", provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key="secret",
        )
        assert isinstance(describer, OpenAIChatSceneDescriber)
        assert describer._warmup_enabled is False

    def test_anthropic_uses_messages_adapter(self) -> None:
        describer = create_describer(
            "ollama", "claude-test", provider="anthropic", api_key="secret"
        )
        assert isinstance(describer, AnthropicMessagesSceneDescriber)
        assert describer._base_url == "https://api.anthropic.com/v1"

    def test_gemini_uses_provider_default_url(self) -> None:
        describer = create_describer(
            "ollama", "gemini-test", provider="gemini", api_key="secret"
        )
        assert describer._base_url == (
            "https://generativelanguage.googleapis.com/v1beta/openai"
        )


class TestEnsureBackendAvailable:
    def test_ollama_reachable_passes(self, monkeypatch) -> None:
        import sys
        import types

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            def list(self):
                return {"models": []}

        fake = types.SimpleNamespace(Client=FakeClient)
        monkeypatch.setitem(sys.modules, "ollama", fake)

        ensure_backend_available("ollama")  # 例外なし

    def test_ollama_unreachable_raises_with_hint(self, monkeypatch) -> None:
        import sys
        import types

        class FailingClient:
            def __init__(self, timeout=None):
                pass

            def list(self):
                raise ConnectionError("refused")

        fake = types.SimpleNamespace(Client=FailingClient)
        monkeypatch.setitem(sys.modules, "ollama", fake)

        import pytest

        with pytest.raises(ValueError) as excinfo:
            ensure_backend_available("ollama")
        assert "ollama serve" in str(excinfo.value)

    def test_unreachable_vllm_raises_with_hint(self, monkeypatch) -> None:
        def fail_get(url, timeout=None):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.get", fail_get)
        with pytest.raises(ValueError) as excinfo:
            ensure_backend_available("vllm-mlx")
        message = str(excinfo.value)
        assert "vllm-mlx serve" in message  # 起動コマンドの提示
        assert "pip install vllm-mlx" in message

    def test_reachable_vllm_passes(self, monkeypatch) -> None:
        class OkResponse:
            status_code = 200

            def raise_for_status(self): ...

        monkeypatch.setattr(
            "httpx.get", lambda url, timeout=None, headers=None: OkResponse()
        )
        ensure_backend_available("vllm-mlx")

    def test_api_key_adds_authorization_header(self, monkeypatch) -> None:
        calls: list[dict] = []

        class OkResponse:
            status_code = 200

            def raise_for_status(self): ...

        def fake_get(url, timeout=None, headers=None):
            calls.append({"headers": headers})
            return OkResponse()

        monkeypatch.setattr("httpx.get", fake_get)
        ensure_backend_available("vllm-mlx", api_key="secret-key")

        (call,) = calls
        assert call["headers"] == {"Authorization": "Bearer secret-key"}

    def test_vllm_404_is_not_treated_as_reachable_with_api_key(
        self, monkeypatch
    ) -> None:
        class NotFoundResponse:
            status_code = 404

            def raise_for_status(self):
                raise RuntimeError("404 Not Found")

        monkeypatch.setattr(
            "httpx.get", lambda url, timeout=None, headers=None: NotFoundResponse()
        )
        with pytest.raises(ValueError):
            ensure_backend_available("vllm-mlx", api_key="secret-key")

    def test_cloud_endpoint_401_raises_with_api_key(self, monkeypatch) -> None:
        class UnauthorizedResponse:
            status_code = 401

            def raise_for_status(self):
                raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(
            "httpx.get", lambda url, timeout=None, headers=None: UnauthorizedResponse()
        )
        with pytest.raises(ValueError):
            ensure_backend_available("vllm-mlx", api_key="secret-key")

    def test_anthropic_preflight_does_not_make_models_request(self, monkeypatch) -> None:
        def fail_get(*args, **kwargs):
            raise AssertionError("Anthropic preflight must not call /models")

        monkeypatch.setattr("httpx.get", fail_get)
        from screen_activity_logger.infrastructure.vlm_factory import (
            ensure_provider_available,
        )

        ensure_provider_available(
            "ollama", "https://api.anthropic.com/v1", "secret-key",
            provider="anthropic", model="claude-test", allow_external=True,
        )

    def test_external_provider_requires_explicit_allowance(self) -> None:
        from screen_activity_logger.infrastructure.vlm_factory import (
            ensure_provider_available,
        )

        with pytest.raises(ValueError, match="allow-external-vlm"):
            ensure_provider_available(
                "ollama", "https://api.anthropic.com/v1", "secret-key",
                provider="anthropic", model="claude-test",
            )


class TestCreateSummarizer:
    def test_ollama_backend(self) -> None:
        from screen_activity_logger.infrastructure.chat_summarizer import (
            OllamaChatSummarizer,
        )
        from screen_activity_logger.infrastructure.vlm_factory import (
            create_summarizer,
        )

        summarizer = create_summarizer("ollama", "qwen3-vl:8b")

        assert isinstance(summarizer, OllamaChatSummarizer)

    def test_vllm_backend_translates_ollama_model_name(self) -> None:
        from screen_activity_logger.infrastructure.chat_summarizer import (
            OpenAIChatSummarizer,
        )
        from screen_activity_logger.infrastructure.openai_chat_describer import (
            DEFAULT_VLLM_MODEL,
        )
        from screen_activity_logger.infrastructure.vlm_factory import (
            create_summarizer,
        )

        summarizer = create_summarizer("vllm-mlx", "qwen3-vl:8b")

        assert isinstance(summarizer, OpenAIChatSummarizer)
        assert summarizer._model == DEFAULT_VLLM_MODEL

    def test_unknown_backend_raises(self) -> None:
        import pytest

        from screen_activity_logger.infrastructure.vlm_factory import (
            create_summarizer,
        )

        with pytest.raises(ValueError):
            create_summarizer("unknown", "m")
