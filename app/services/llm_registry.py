"""OpenAI-compatible LLM provider registry.

The registry owns model selection and public catalog shaping. Provider
credentials are resolved only when a runtime request is built; public APIs never
receive env names, base URLs, or API keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


SUPPORTED_PROVIDER_TYPE = "openai-compatible"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_PROVIDER_CATALOG: list[dict[str, Any]] = [
    {
        "id": "ninerouter",
        "label": "9Router",
        "type": SUPPORTED_PROVIDER_TYPE,
        "baseUrlEnv": "NINEROUTER_BASE_URL",
        "apiKeyEnv": "NINEROUTER_API_KEY",
        "enabled": True,
        "models": [
            {
                "id": "gpt-5.3-codex",
                "label": "GPT-5.3 Codex",
                "displayProvider": "OpenAI",
                "model": "gh/gpt-5.3-codex",
                "enabled": True,
                "capabilities": {
                    "reasoningEffort": {
                        "enabled": True,
                        "allowedValues": ["low", "medium", "high"],
                        "default": "medium",
                        "userConfigurable": True,
                        "paramStyle": "openai_chat_reasoning_effort",
                    }
                },
            },
            {
                "id": "gpt-5.4",
                "label": "GPT-5.4",
                "displayProvider": "OpenAI",
                "model": "gh/gpt-5.4",
                "enabled": True,
                "default": True,
                "capabilities": {
                    "reasoningEffort": {
                        "enabled": True,
                        "allowedValues": ["low", "medium", "high"],
                        "default": "medium",
                        "userConfigurable": True,
                        "paramStyle": "openai_chat_reasoning_effort",
                    }
                },
            },
            {
                "id": "gpt-5.4-mini",
                "label": "GPT-5.4 Mini",
                "displayProvider": "OpenAI",
                "model": "gh/gpt-5.4-mini",
                "enabled": True,
                "capabilities": {
                    "reasoningEffort": {
                        "enabled": True,
                        "allowedValues": ["low", "medium", "high"],
                        "default": "medium",
                        "userConfigurable": True,
                        "paramStyle": "openai_chat_reasoning_effort",
                    }
                },
            },
            {
                "id": "gemini-3.1-pro-preview",
                "label": "Gemini 3.1 Pro Preview",
                "displayProvider": "Google/Gemini",
                "model": "gh/gemini-3.1-pro-preview",
                "enabled": True,
                "capabilities": {"reasoningEffort": {"enabled": False, "userConfigurable": False}},
            },
            {
                "id": "claude-haiku-4.5",
                "label": "Claude Haiku 4.5",
                "displayProvider": "Anthropic",
                "model": "gh/claude-haiku-4.5",
                "enabled": True,
                "capabilities": {"reasoningEffort": {"enabled": False, "userConfigurable": False}},
            },
            {
                "id": "claude-sonnet-4.6",
                "label": "Claude Sonnet 4.6",
                "displayProvider": "Anthropic",
                "model": "gh/claude-sonnet-4.6",
                "enabled": True,
                "capabilities": {
                    "reasoningEffort": {
                        "enabled": True,
                        "allowedValues": ["low", "medium", "high"],
                        "default": "medium",
                        "userConfigurable": True,
                        "paramStyle": "openai_chat_reasoning_effort",
                    }
                },
            },
            {
                "id": "claude-opus-4.6",
                "label": "Claude Opus 4.6",
                "displayProvider": "Anthropic",
                "model": "gh/claude-opus-4.6",
                "enabled": False,
                "capabilities": {
                    "reasoningEffort": {
                        "enabled": True,
                        "allowedValues": ["low", "medium", "high", "xhigh"],
                        "default": "medium",
                        "userConfigurable": True,
                        "paramStyle": "openai_chat_reasoning_effort",
                    }
                },
            },
            {
                "id": "deepseek-v4-pro",
                "label": "DeepSeek V4 Pro",
                "displayProvider": "DeepSeek",
                "model": "kc/deepseek/deepseek-chat",
                "enabled": True,
                "capabilities": {"reasoningEffort": {"enabled": False, "userConfigurable": False}},
            },
            {
                "id": "nvidia-nemotron-3-super-120b",
                "label": "NVIDIA Nemotron 3 Super 120B",
                "displayProvider": "NVIDIA",
                "model": "kc/nvidia/nemotron-3-super-120b-a12b:free",
                "enabled": True,
                "capabilities": {"reasoningEffort": {"enabled": False, "userConfigurable": False}},
            },
            {
                "id": "moonshotai-kimi-k2.6",
                "label": "MoonshotAI Kimi K2.6",
                "displayProvider": "MoonshotAI",
                "model": "kc/moonshotai/kimi-k2.6",
                "enabled": True,
                "capabilities": {
                    "reasoningEffort": {
                        "enabled": True,
                        "allowedValues": ["instant", "thinking"],
                        "default": "instant",
                        "userConfigurable": True,
                        "paramStyle": "openrouter_reasoning",
                        "valueMap": {"instant": "none", "thinking": "minimal"},
                    }
                },
            },
            {
                "id": "inclusionai-ling-2.6-1t",
                "label": "inclusionAI Ling-2.6-1T",
                "displayProvider": "inclusionAI",
                "model": "kc/inclusionai/ling-2.6-1t:free",
                "enabled": True,
                "capabilities": {"reasoningEffort": {"enabled": False, "userConfigurable": False}},
            },
            {
                "id": "qwen3.6-plus",
                "label": "Qwen 3.6 Plus",
                "displayProvider": "Qwen",
                "model": "kc/qwen/qwen3.6-plus",
                "enabled": True,
                "capabilities": {
                    "reasoningEffort": {
                        "enabled": True,
                        "allowedValues": ["instant", "thinking"],
                        "default": "instant",
                        "userConfigurable": True,
                        "paramStyle": "openrouter_reasoning",
                        "valueMap": {"instant": "none", "thinking": "minimal"},
                    }
                },
            },
            {
                "id": "minimax-m2.7",
                "label": "MiniMax M2.7",
                "displayProvider": "MiniMax",
                "model": "kc/minimax/minimax-m2.7",
                "enabled": True,
                "capabilities": {"reasoningEffort": {"enabled": False, "userConfigurable": False}},
            },
        ],
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "type": SUPPORTED_PROVIDER_TYPE,
        "baseUrlEnv": "OPENROUTER_BASE_URL",
        "apiKeyEnv": "OPENROUTER_API_KEY",
        "enabled": False,
        "models": [
            {
                "id": "openrouter-gpt-5",
                "label": "GPT-5 via OpenRouter",
                "displayProvider": "OpenAI",
                "model": "openai/gpt-5",
                "enabled": True,
                "capabilities": {
                    "reasoningEffort": {
                        "enabled": True,
                        "allowedValues": ["low", "medium", "high"],
                        "default": "medium",
                        "userConfigurable": True,
                        "paramStyle": "openrouter_reasoning",
                    }
                },
            }
        ],
    },
    {
        "id": "custom-openai",
        "label": "Custom OpenAI Compatible",
        "type": SUPPORTED_PROVIDER_TYPE,
        "baseUrlEnv": "CUSTOM_OPENAI_BASE_URL",
        "apiKeyEnv": "CUSTOM_OPENAI_API_KEY",
        "enabled": False,
        "models": [
            {
                "id": "custom-model",
                "label": "Custom Model",
                "displayProvider": "Custom",
                "model": "custom-model-name",
                "enabled": True,
                "capabilities": {"reasoningEffort": {"enabled": False, "userConfigurable": False}},
            }
        ],
    },
]


class LLMRegistryError(ValueError):
    """Raised when provider catalog or model selection is invalid."""


def _optional_positive_int(raw: Any, field_name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise LLMRegistryError(f"{field_name} must be an integer greater than 0.")
    if raw <= 0:
        raise LLMRegistryError(f"{field_name} must be an integer greater than 0.")
    return raw


@dataclass(frozen=True)
class ReasoningEffortCapability:
    enabled: bool = False
    allowed_values: tuple[str, ...] = ()
    default: str | None = None
    user_configurable: bool = False
    param_style: str | None = None
    display_mode: str | None = None
    value_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Any) -> "ReasoningEffortCapability":
        if not isinstance(raw, dict):
            return cls()
        allowed_values = tuple(
            str(value).strip().lower()
            for value in raw.get("allowedValues", [])
            if str(value).strip()
        )
        default = str(raw.get("default") or "").strip().lower() or None
        if default and allowed_values and default not in allowed_values:
            raise LLMRegistryError("reasoningEffort.default must be in allowedValues")
        value_map = {
            str(key).strip().lower(): str(value).strip()
            for key, value in (raw.get("valueMap") or {}).items()
            if str(key).strip() and str(value).strip()
        }
        return cls(
            enabled=bool(raw.get("enabled", False)),
            allowed_values=allowed_values,
            default=default,
            user_configurable=bool(raw.get("userConfigurable", False)),
            param_style=str(raw.get("paramStyle") or "").strip() or None,
            display_mode=str(raw.get("displayMode") or "").strip() or None,
            value_map=value_map,
        )

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "userConfigurable": self.user_configurable,
        }
        if self.allowed_values:
            payload["allowedValues"] = list(self.allowed_values)
        if self.default:
            payload["default"] = self.default
        if self.display_mode:
            payload["displayMode"] = self.display_mode
        return payload

    def validate(self, value: str | None) -> str | None:
        normalized = (value or "").strip().lower()
        if not self.enabled:
            if normalized:
                raise LLMRegistryError("Selected model does not support reasoning effort.")
            return None
        if not normalized:
            return self.default
        if normalized not in self.allowed_values:
            raise LLMRegistryError(
                "Unsupported reasoning effort '%s'. Supported values: %s"
                % (normalized, ", ".join(self.allowed_values) or "none")
            )
        return normalized

    def provider_value(self, value: str | None) -> str | None:
        normalized = self.validate(value)
        if not normalized:
            return None
        return self.value_map.get(normalized, normalized)


@dataclass(frozen=True)
class ModelConfig:
    id: str
    label: str
    display_provider: str
    model: str
    enabled: bool
    default: bool
    capabilities: dict[str, Any]
    provider_id: str
    provider_label: str
    context_window: int | None = None
    max_output_tokens: int | None = None

    @property
    def reasoning_effort(self) -> ReasoningEffortCapability:
        return ReasoningEffortCapability.from_mapping(
            (self.capabilities or {}).get("reasoningEffort")
        )

    @classmethod
    def from_mapping(cls, raw: Any, provider_id: str, provider_label: str) -> "ModelConfig":
        if not isinstance(raw, dict):
            raise LLMRegistryError("Provider models must be objects.")
        model_id = str(raw.get("id") or "").strip()
        actual_model = str(raw.get("model") or "").strip()
        if not model_id or not actual_model:
            raise LLMRegistryError("Each model requires id and model.")
        return cls(
            id=model_id,
            label=str(raw.get("label") or model_id).strip(),
            display_provider=str(raw.get("displayProvider") or provider_label).strip(),
            model=actual_model,
            enabled=bool(raw.get("enabled", True)),
            default=bool(raw.get("default", False)),
            capabilities=raw.get("capabilities") if isinstance(raw.get("capabilities"), dict) else {},
            provider_id=provider_id,
            provider_label=provider_label,
            context_window=_optional_positive_int(raw.get("contextWindow"), "contextWindow"),
            max_output_tokens=_optional_positive_int(raw.get("maxOutputTokens"), "maxOutputTokens"),
        )

    def public_dict(
        self,
        *,
        fallback_context_window: int | None = None,
        fallback_max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "label": self.label,
            "displayProvider": self.display_provider,
            "enabled": self.enabled,
            "default": self.default,
            "capabilities": {
                "reasoningEffort": self.reasoning_effort.public_dict(),
            },
        }
        context_window = self.context_window or fallback_context_window
        max_output_tokens = self.max_output_tokens or fallback_max_output_tokens
        if context_window:
            payload["contextWindow"] = context_window
        if max_output_tokens:
            payload["maxOutputTokens"] = max_output_tokens
        return payload


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    label: str
    type: str
    base_url_env: str
    api_key_env: str
    enabled: bool
    models: tuple[ModelConfig, ...]

    @classmethod
    def from_mapping(cls, raw: Any) -> "ProviderConfig":
        if not isinstance(raw, dict):
            raise LLMRegistryError("Provider catalog entries must be objects.")
        provider_id = str(raw.get("id") or "").strip()
        provider_type = str(raw.get("type") or "").strip()
        if not provider_id:
            raise LLMRegistryError("Each provider requires id.")
        if provider_type != SUPPORTED_PROVIDER_TYPE:
            raise LLMRegistryError(f"Unsupported provider type '{provider_type}'.")
        label = str(raw.get("label") or provider_id).strip()
        models = tuple(
            ModelConfig.from_mapping(item, provider_id=provider_id, provider_label=label)
            for item in raw.get("models", [])
        )
        return cls(
            id=provider_id,
            label=label,
            type=provider_type,
            base_url_env=str(raw.get("baseUrlEnv") or "").strip(),
            api_key_env=str(raw.get("apiKeyEnv") or "").strip(),
            enabled=bool(raw.get("enabled", True)),
            models=models,
        )


@dataclass(frozen=True)
class RuntimeModelSelection:
    provider: ProviderConfig
    model: ModelConfig
    base_url: str
    api_key: str
    litellm_model: str
    reasoning_effort: str | None
    reasoning_param_style: str | None
    context_window: int
    max_output_tokens: int


def _resolve_catalog_file(catalog_file: str, project_root: Path = PROJECT_ROOT) -> Path:
    catalog_path = Path(catalog_file).expanduser()
    if not catalog_path.is_absolute():
        catalog_path = project_root / catalog_path
    return catalog_path


def _load_catalog_from_file(catalog_file: str, project_root: Path = PROJECT_ROOT) -> Any:
    catalog_path = _resolve_catalog_file(catalog_file, project_root)
    try:
        raw_text = catalog_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise LLMRegistryError(
            f"LLM_PROVIDER_CATALOG_FILE does not exist: {catalog_path}"
        ) from exc
    except OSError as exc:
        raise LLMRegistryError(
            f"LLM_PROVIDER_CATALOG_FILE could not be read: {catalog_path}"
        ) from exc

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMRegistryError(
            f"LLM_PROVIDER_CATALOG_FILE contains invalid JSON: {catalog_path}: {exc.msg}"
        ) from exc


def load_provider_catalog(
    catalog_json: str | None,
    catalog_file: str | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[ProviderConfig, ...]:
    raw_text = (catalog_json or "").strip()
    if raw_text:
        try:
            raw_catalog = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMRegistryError(f"LLM_PROVIDER_CATALOG_JSON is invalid JSON: {exc.msg}") from exc
        source_name = "LLM_PROVIDER_CATALOG_JSON"
    elif (catalog_file or "").strip():
        raw_catalog = _load_catalog_from_file((catalog_file or "").strip(), project_root)
        source_name = "LLM_PROVIDER_CATALOG_FILE"
    else:
        raw_catalog = DEFAULT_PROVIDER_CATALOG
        source_name = "LLM_PROVIDER_CATALOG_JSON"

    if not isinstance(raw_catalog, list):
        raise LLMRegistryError(f"{source_name} must be a JSON array.")
    return tuple(ProviderConfig.from_mapping(item) for item in raw_catalog)


class OpenAICompatibleProviderRegistry:
    def __init__(
        self,
        providers: tuple[ProviderConfig, ...],
        *,
        default_provider_id: str | None,
        default_model_id: str | None,
        env_resolver,
        fallback_context_window: int = 200000,
        fallback_max_output_tokens: int = 5000,
    ):
        self.providers = providers
        self.default_provider_id = (default_provider_id or "").strip()
        self.default_model_id = (default_model_id or "").strip()
        self._env_resolver = env_resolver
        self.fallback_context_window = _optional_positive_int(
            fallback_context_window,
            "fallback_context_window",
        ) or 200000
        self.fallback_max_output_tokens = _optional_positive_int(
            fallback_max_output_tokens,
            "fallback_max_output_tokens",
        ) or 5000

    @classmethod
    def from_settings(cls, settings) -> "OpenAICompatibleProviderRegistry":
        return cls(
            load_provider_catalog(
                getattr(settings, "llm_provider_catalog_json", ""),
                getattr(settings, "llm_provider_catalog_file", ""),
            ),
            default_provider_id=getattr(settings, "llm_default_provider", ""),
            default_model_id=getattr(settings, "llm_default_model", ""),
            env_resolver=settings.get_llm_runtime_env,
            fallback_context_window=getattr(settings, "llm_context_window", 200000),
            fallback_max_output_tokens=getattr(settings, "max_output_tokens", 5000),
        )

    def public_catalog(self) -> dict[str, Any]:
        models = [
            model.public_dict(
                fallback_context_window=self.fallback_context_window,
                fallback_max_output_tokens=self.fallback_max_output_tokens,
            )
            for provider in self.providers
            if self._provider_selectable(provider)
            for model in provider.models
            if model.enabled
        ]
        public_model_ids = {model["id"] for model in models}
        default_model_id = self.default_model_id if self.default_model_id in public_model_ids else None
        default_model_id = default_model_id or self._first_default_model_id(models)
        return {"defaultModel": default_model_id, "models": models}

    def select_model(
        self,
        model_id: str | None = None,
        *,
        reasoning_effort: str | None = None,
    ) -> RuntimeModelSelection:
        requested_model_id = (model_id or "").strip() or self.default_model_id
        candidate = self._find_enabled_model(requested_model_id)
        if candidate is None and not requested_model_id:
            candidate = self._first_enabled_model()
        if candidate is None:
            raise LLMRegistryError(f"Unknown or disabled chat model '{requested_model_id}'.")

        provider, model = candidate
        if not self._provider_selectable(provider):
            raise LLMRegistryError(f"Provider '{provider.id}' is disabled or missing runtime configuration.")

        base_url = self._resolve_required_env(provider.base_url_env, f"{provider.id} base URL")
        api_key = self._resolve_required_env(provider.api_key_env, f"{provider.id} API key")
        reasoning = model.reasoning_effort
        provider_reasoning_effort = reasoning.provider_value(reasoning_effort)

        return RuntimeModelSelection(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            litellm_model=f"openai/{model.model}",
            reasoning_effort=provider_reasoning_effort,
            reasoning_param_style=reasoning.param_style if provider_reasoning_effort else None,
            context_window=model.context_window or self.fallback_context_window,
            max_output_tokens=model.max_output_tokens or self.fallback_max_output_tokens,
        )

    def validate_model_request(self, model_id: str | None, reasoning_effort: str | None = None) -> str | None:
        selection = self.select_model(model_id, reasoning_effort=reasoning_effort)
        return selection.model.id

    def validate_catalog_model_request(
        self,
        model_id: str | None,
        reasoning_effort: str | None = None,
    ) -> str | None:
        requested_model_id = (model_id or "").strip()
        if not requested_model_id:
            return None
        candidate = self._find_enabled_model(requested_model_id)
        if candidate is None:
            raise LLMRegistryError(f"Unknown or disabled chat model '{requested_model_id}'.")
        _, model = candidate
        model.reasoning_effort.validate(reasoning_effort)
        return model.id

    def resolve_catalog_litellm_model(self, model_id: str | None = None) -> str:
        candidate = self._find_enabled_model((model_id or "").strip())
        if candidate is None and not (model_id or "").strip():
            candidate = self._first_enabled_model()
        if candidate is None:
            raise LLMRegistryError(f"Unknown or disabled chat model '{model_id or ''}'.")
        _, model = candidate
        return f"openai/{model.model}"

    def _find_enabled_model(self, model_id: str | None) -> tuple[ProviderConfig, ModelConfig] | None:
        normalized = (model_id or "").strip()
        for provider in self.providers:
            for model in provider.models:
                if model.id == normalized and provider.enabled and model.enabled:
                    return provider, model
        return None

    def _first_enabled_model(self) -> tuple[ProviderConfig, ModelConfig] | None:
        for provider in self.providers:
            if self.default_provider_id and provider.id != self.default_provider_id:
                continue
            if not provider.enabled:
                continue
            for model in provider.models:
                if model.enabled and model.default:
                    return provider, model
            for model in provider.models:
                if model.enabled:
                    return provider, model
        for provider in self.providers:
            if not provider.enabled:
                continue
            for model in provider.models:
                if model.enabled:
                    return provider, model
        return None

    def _provider_selectable(self, provider: ProviderConfig) -> bool:
        if not provider.enabled:
            return False
        return bool(self._env_resolver(provider.base_url_env)) and bool(self._env_resolver(provider.api_key_env))

    def _resolve_required_env(self, env_name: str, label: str) -> str:
        value = self._env_resolver(env_name)
        if not value:
            raise LLMRegistryError(f"Missing required {label}. Configure {env_name}.")
        return value

    @staticmethod
    def _first_default_model_id(models: list[dict[str, Any]]) -> str | None:
        for model in models:
            if model.get("default"):
                return model.get("id")
        return models[0].get("id") if models else None
