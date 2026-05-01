"""Tests for ZHIPU provider implementation."""

import os
from unittest.mock import Mock, patch

import pytest

from providers.shared import ModelCapabilities, ModelResponse, ProviderType
from providers.zhipu import ZhipuModelProvider


class TestZhipuProvider:
    """Test ZHIPU provider functionality."""

    def setup_method(self):
        """Set up clean state before each test."""
        # Clear restriction service cache before each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        # Force reload registry for clean state
        ZhipuModelProvider._ensure_registry(force_reload=True)

    def teardown_method(self):
        """Clean up after each test to avoid singleton issues."""
        # Clear restriction service cache after each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(os.environ, {"ZHIPU_API_KEY": "test-zhipu-key"})
    def test_initialization(self):
        """Test provider initialization."""
        provider = ZhipuModelProvider("test-zhipu-key")
        assert provider.api_key == "test-zhipu-key"
        assert provider.get_provider_type() == ProviderType.ZHIPU
        assert provider.base_url == "https://api.z.ai/api/paas/v4"

    def test_initialization_with_custom_url(self):
        """Explicit base_url kwarg bypasses ZHIPU_BASE_URL allowlist (caller-controlled)."""
        provider = ZhipuModelProvider(
            "test-zhipu-key", base_url="https://api.z.ai/api/coding/paas/v4"
        )
        assert provider.api_key == "test-zhipu-key"
        assert provider.base_url == "https://api.z.ai/api/coding/paas/v4"

    def test_initialization_no_api_key_raises(self):
        """Empty API key must fail-fast at construction."""
        with pytest.raises(ValueError, match="ZHIPU_API_KEY is required"):
            ZhipuModelProvider("")

    @patch.dict(os.environ, {"ZHIPU_BASE_URL": "https://attacker.example.com/v1"})
    def test_zhipu_base_url_rejects_unknown_host(self):
        """ZHIPU_BASE_URL pointing at a non-allowlisted host must be rejected."""
        with pytest.raises(ValueError, match="not in the allowlist"):
            ZhipuModelProvider("test-zhipu-key")

    @patch.dict(os.environ, {"ZHIPU_BASE_URL": "http://api.z.ai/api/paas/v4"})
    def test_zhipu_base_url_rejects_http(self):
        """ZHIPU_BASE_URL must be https."""
        with pytest.raises(ValueError, match="not in the allowlist|https only"):
            ZhipuModelProvider("test-zhipu-key")

    @patch.dict(os.environ, {"ZHIPU_BASE_URL": "https://api.z.ai/api/coding/paas/v4"})
    def test_zhipu_base_url_coding_plan_accepted(self):
        """Coding-plan endpoint on api.z.ai must be accepted via env override."""
        provider = ZhipuModelProvider("test-zhipu-key")
        assert provider.base_url == "https://api.z.ai/api/coding/paas/v4"

    def test_model_validation(self):
        """Test model name validation."""
        provider = ZhipuModelProvider("test-zhipu-key")

        assert provider.validate_model_name("glm-5.1") is True
        assert provider.validate_model_name("glm-4.7") is True
        assert provider.validate_model_name("glm-4.6") is True
        assert provider.validate_model_name("glm-4.5") is True

        assert provider.validate_model_name("glm") is True
        assert provider.validate_model_name("glm5") is True
        assert provider.validate_model_name("glm51") is True
        assert provider.validate_model_name("glm47") is True
        assert provider.validate_model_name("glm46") is True
        assert provider.validate_model_name("glm45") is True
        assert provider.validate_model_name("glm4") is True

        assert provider.validate_model_name("invalid-model") is False

    def test_get_capabilities_glm45(self):
        """Test getting model capabilities for GLM-4.5."""
        provider = ZhipuModelProvider("test-zhipu-key")

        capabilities = provider.get_capabilities("glm-4.5")
        assert capabilities.model_name == "glm-4.5"
        assert capabilities.friendly_name == "GLM"
        assert capabilities.context_window == 128000
        assert capabilities.max_output_tokens == 96000
        assert capabilities.provider == ProviderType.ZHIPU
        assert capabilities.supports_extended_thinking is True
        assert not capabilities.supports_images
        assert capabilities.supports_temperature is True

        assert capabilities.temperature_constraint.min_temp == 0.0
        assert capabilities.temperature_constraint.max_temp == 2.0

    def test_get_capabilities_glm46(self):
        """Test getting model capabilities for GLM-4.6."""
        provider = ZhipuModelProvider("test-zhipu-key")

        capabilities = provider.get_capabilities("glm-4.6")
        assert capabilities.model_name == "glm-4.6"
        assert capabilities.friendly_name == "GLM"
        assert capabilities.context_window == 200000
        assert capabilities.max_output_tokens == 128000
        assert capabilities.provider == ProviderType.ZHIPU
        assert capabilities.supports_extended_thinking is True
        assert not capabilities.supports_images
        assert capabilities.supports_temperature is True

        assert capabilities.temperature_constraint.min_temp == 0.0
        assert capabilities.temperature_constraint.max_temp == 2.0

    def test_get_capabilities_glm47(self):
        """Test getting model capabilities for GLM-4.7."""
        provider = ZhipuModelProvider("test-zhipu-key")

        capabilities = provider.get_capabilities("glm-4.7")
        assert capabilities.model_name == "glm-4.7"
        assert capabilities.context_window == 200000
        assert capabilities.max_output_tokens == 128000
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_function_calling is True
        assert not capabilities.supports_images

    def test_get_capabilities_glm51(self):
        """Test getting model capabilities for GLM-5.1."""
        provider = ZhipuModelProvider("test-zhipu-key")

        capabilities = provider.get_capabilities("glm-5.1")
        assert capabilities.model_name == "glm-5.1"
        assert capabilities.context_window == 200000
        assert capabilities.max_output_tokens == 128000
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_function_calling is True
        assert not capabilities.supports_images

    def test_get_all_model_capabilities(self):
        """Test getting all model capabilities."""
        provider = ZhipuModelProvider("test-zhipu-key")
        all_caps = provider.get_all_model_capabilities()

        for name in ("glm-5.1", "glm-4.7", "glm-4.6", "glm-4.5"):
            assert name in all_caps
            assert isinstance(all_caps[name], ModelCapabilities)

    def test_get_preferred_model(self):
        """Test preferred model selection."""
        provider = ZhipuModelProvider("test-zhipu-key")

        preferred = provider.get_preferred_model("chat", ["glm-4.5", "glm-4.6", "glm-4.7", "glm-5.1"])
        assert preferred == "glm-5.1"

        preferred = provider.get_preferred_model("chat", ["glm-4.5", "glm-4.6", "glm-4.7"])
        assert preferred == "glm-4.7"

        preferred = provider.get_preferred_model("chat", ["glm-4.5", "glm-4.6"])
        assert preferred == "glm-4.6"

        preferred = provider.get_preferred_model("chat", ["glm-4.5"])
        assert preferred == "glm-4.5"

    def test_get_preferred_model_no_allowed(self):
        """Test preferred model selection with no allowed models."""
        provider = ZhipuModelProvider("test-zhipu-key")

        preferred = provider.get_preferred_model("chat", [])
        assert preferred is None

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_openai_compatible(self, mock_openai):
        """Test content generation using OpenAI-compatible interface."""
        # Mock OpenAI client response
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated content from GLM via OpenAI-compatible"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = ZhipuModelProvider("test-zhipu-key")

        response = provider.generate_content(
            prompt="Test prompt",
            model_name="glm-4.6",
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            max_output_tokens=1000,
        )

        # Verify response structure
        assert isinstance(response, ModelResponse)
        assert response.content == "Generated content from GLM via OpenAI-compatible"
        assert response.usage["input_tokens"] == 50
        assert response.usage["output_tokens"] == 100
        assert response.usage["total_tokens"] == 150
        assert response.model_name == "glm-4.6"
        assert response.friendly_name == "GLM"
        assert response.provider == ProviderType.ZHIPU

        # Verify OpenAI client was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "glm-4.6"
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 1000
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][0]["content"] == "You are a helpful assistant"
        assert call_args["messages"][1]["role"] == "user"
        assert call_args["messages"][1]["content"] == "Test prompt"

    def test_friendly_name(self):
        """Test friendly name property."""
        provider = ZhipuModelProvider("test-zhipu-key")
        assert provider.FRIENDLY_NAME == "GLM"

    def test_model_capabilities_dict(self):
        """Test model capabilities dictionary."""
        provider = ZhipuModelProvider("test-zhipu-key")
        assert hasattr(provider, "MODEL_CAPABILITIES")
        assert isinstance(provider.MODEL_CAPABILITIES, dict)
        for name in ("glm-5.1", "glm-4.7", "glm-4.6", "glm-4.5"):
            assert name in provider.MODEL_CAPABILITIES
