"""GLM (Zhipu AI) model provider implementation using OpenAI-compatible API."""

import os
from typing import TYPE_CHECKING, ClassVar, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .openai_compatible import OpenAICompatibleProvider
from .registries.zhipu import ZhipuModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

DEFAULT_ZHIPU_BASE_URL = "https://api.z.ai/api/paas/v4"
ALLOWED_ZHIPU_HOSTS: frozenset[str] = frozenset({"api.z.ai", "open.bigmodel.cn"})


def _resolve_zhipu_base_url() -> str:
    """Return the configured Z.AI base URL, restricted to known hosts.

    Honours ``ZHIPU_BASE_URL`` so users on the Coding plan can point at
    ``https://api.z.ai/api/coding/paas/v4`` without code changes, but rejects
    arbitrary hosts so a poisoned env cannot exfiltrate the bearer token.
    """
    base_url = os.getenv("ZHIPU_BASE_URL") or DEFAULT_ZHIPU_BASE_URL
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_ZHIPU_HOSTS:
        raise ValueError(
            f"ZHIPU_BASE_URL host '{parsed.hostname}' is not in the allowlist "
            f"{sorted(ALLOWED_ZHIPU_HOSTS)} (https only)."
        )
    return base_url


class ZhipuModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for GLM models exposed over OpenAI-compatible API.

    Publishes capability metadata for officially supported GLM models
    and maps tool-category preferences to appropriate GLM model.
    """

    FRIENDLY_NAME = "GLM"
    REGISTRY_CLASS = ZhipuModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    def __init__(self, api_key: str, **kwargs):
        """Initialize GLM provider with API key."""
        if not api_key:
            raise ValueError("ZHIPU_API_KEY is required to initialize the GLM provider.")
        self._ensure_registry()
        kwargs.setdefault("base_url", _resolve_zhipu_base_url())
        super().__init__(api_key=api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.ZHIPU

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Get GLM's preferred model for a given category from allowed models."""
        preferred_models = ["glm-5.1", "glm-4.7", "glm-4.6", "glm-4.5", "glm-4.5-air"]
        for model in preferred_models:
            if model in allowed_models:
                return model

        return allowed_models[0] if allowed_models else None
