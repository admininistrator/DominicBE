import json

import pytest

from app.services.llm_registry import (
    LLMRegistryError,
    OpenAICompatibleProviderRegistry,
    load_provider_catalog,
)


CATALOG = [
    {
        "id": "ninerouter",
        "label": "9Router",
        "type": "openai-compatible",
        "baseUrlEnv": "NINEROUTER_BASE_URL",
        "apiKeyEnv": "NINEROUTER_API_KEY",
        "enabled": True,
        "models": [
            {
                "id": "gpt-default",
                "label": "GPT Default",
                "displayProvider": "OpenAI",
                "model": "openai/gpt-default",
                "enabled": True,
                "default": True,
                "contextWindow": 123456,
                "maxOutputTokens": 4321,
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
                "id": "disabled-model",
                "label": "Disabled",
                "displayProvider": "OpenAI",
                "model": "openai/disabled",
                "enabled": False,
            },
        ],
    },
    {
        "id": "disabled-provider",
        "label": "Disabled Provider",
        "type": "openai-compatible",
        "baseUrlEnv": "DISABLED_BASE_URL",
        "apiKeyEnv": "DISABLED_API_KEY",
        "enabled": False,
        "models": [
            {
                "id": "disabled-provider-model",
                "label": "Hidden",
                "displayProvider": "Hidden",
                "model": "hidden/model",
                "enabled": True,
            }
        ],
    },
]


def make_registry(env=None, *, fallback_context_window=200000, fallback_max_output_tokens=5000):
    values = {
        "NINEROUTER_BASE_URL": "http://localhost:20128/v1",
        "NINEROUTER_API_KEY": "test-key",
    }
    if env is not None:
        values = env
    return OpenAICompatibleProviderRegistry(
        load_provider_catalog(json.dumps(CATALOG)),
        default_provider_id="ninerouter",
        default_model_id="gpt-default",
        env_resolver=lambda name: values.get(name, ""),
        fallback_context_window=fallback_context_window,
        fallback_max_output_tokens=fallback_max_output_tokens,
    )


def catalog_with_model(model_id: str) -> list[dict]:
    catalog = [dict(CATALOG[0])]
    catalog[0]["models"] = [
        {
            "id": model_id,
            "label": model_id,
            "displayProvider": "Custom",
            "model": f"custom/{model_id}",
            "enabled": True,
            "default": True,
            "capabilities": {"reasoningEffort": {"enabled": False, "userConfigurable": False}},
        }
    ]
    return catalog


def test_catalog_json_env_takes_precedence_over_file(tmp_path):
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(catalog_with_model("file-model")), encoding="utf-8")

    providers = load_provider_catalog(
        json.dumps(catalog_with_model("env-model")),
        str(catalog_file),
        project_root=tmp_path,
    )

    assert providers[0].models[0].id == "env-model"


def test_catalog_file_loads_when_json_env_is_empty(tmp_path):
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(catalog_with_model("file-model")), encoding="utf-8")

    providers = load_provider_catalog("", str(catalog_file), project_root=tmp_path)

    assert providers[0].models[0].id == "file-model"


def test_relative_catalog_file_resolves_from_project_root(tmp_path):
    catalog_dir = tmp_path / "config"
    catalog_dir.mkdir()
    catalog_file = catalog_dir / "llm_provider_catalog.json"
    catalog_file.write_text(json.dumps(catalog_with_model("relative-model")), encoding="utf-8")

    providers = load_provider_catalog(
        "",
        "config/llm_provider_catalog.json",
        project_root=tmp_path,
    )

    assert providers[0].models[0].id == "relative-model"


def test_missing_catalog_file_raises_registry_error(tmp_path):
    missing_file = tmp_path / "missing.json"

    with pytest.raises(LLMRegistryError, match="does not exist"):
        load_provider_catalog("", str(missing_file), project_root=tmp_path)


def test_invalid_catalog_file_json_raises_registry_error_without_raw_content(tmp_path):
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text('{"secret":"do-not-leak"', encoding="utf-8")

    with pytest.raises(LLMRegistryError) as exc_info:
        load_provider_catalog("", str(catalog_file), project_root=tmp_path)

    message = str(exc_info.value)
    assert "invalid JSON" in message
    assert "do-not-leak" not in message


def test_from_settings_passes_catalog_file_when_json_env_is_empty(tmp_path):
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(catalog_with_model("settings-file-model")), encoding="utf-8")

    class FakeSettings:
        llm_provider_catalog_json = ""
        llm_provider_catalog_file = str(catalog_file)
        llm_default_provider = "ninerouter"
        llm_default_model = "settings-file-model"
        llm_context_window = 200000
        max_output_tokens = 5000

        def get_llm_runtime_env(self, name):
            return "value"

    registry = OpenAICompatibleProviderRegistry.from_settings(FakeSettings())

    assert registry.public_catalog()["defaultModel"] == "settings-file-model"


def test_selects_default_model_and_runtime_provider_config():
    selection = make_registry().select_model(None)

    assert selection.provider.id == "ninerouter"
    assert selection.model.id == "gpt-default"
    assert selection.base_url == "http://localhost:20128/v1"
    assert selection.api_key == "test-key"
    assert selection.litellm_model == "openai/openai/gpt-default"
    assert selection.reasoning_effort == "medium"
    assert selection.context_window == 123456
    assert selection.max_output_tokens == 4321


def test_missing_model_token_limits_fall_back_to_global_settings():
    catalog = [dict(CATALOG[0])]
    catalog[0]["models"] = [
        {
            "id": "fallback-model",
            "label": "Fallback Model",
            "displayProvider": "OpenAI",
            "model": "openai/fallback-model",
            "enabled": True,
            "default": True,
        }
    ]
    registry = OpenAICompatibleProviderRegistry(
        load_provider_catalog(json.dumps(catalog)),
        default_provider_id="ninerouter",
        default_model_id="fallback-model",
        env_resolver=lambda name: "value",
        fallback_context_window=98765,
        fallback_max_output_tokens=6789,
    )

    selection = registry.select_model("fallback-model")

    assert selection.context_window == 98765
    assert selection.max_output_tokens == 6789
    public_model = registry.public_catalog()["models"][0]
    assert public_model["contextWindow"] == 98765
    assert public_model["maxOutputTokens"] == 6789


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("contextWindow", 0),
        ("contextWindow", -1),
        ("contextWindow", "200000"),
        ("contextWindow", True),
        ("maxOutputTokens", 0),
        ("maxOutputTokens", -1),
        ("maxOutputTokens", "5000"),
        ("maxOutputTokens", False),
    ],
)
def test_invalid_model_token_limits_are_rejected(field_name, value):
    catalog = [dict(CATALOG[0])]
    catalog[0]["models"] = [dict(CATALOG[0]["models"][0], **{field_name: value})]

    with pytest.raises(LLMRegistryError, match=field_name):
        load_provider_catalog(json.dumps(catalog))


def test_rejects_unknown_and_disabled_models():
    registry = make_registry()

    with pytest.raises(LLMRegistryError, match="Unknown or disabled"):
        registry.select_model("missing-model")

    with pytest.raises(LLMRegistryError, match="Unknown or disabled"):
        registry.select_model("disabled-model")


def test_disabled_provider_models_are_not_public():
    catalog = make_registry().public_catalog()
    serialized_catalog = json.dumps(catalog)

    model_ids = {model["id"] for model in catalog["models"]}
    assert "gpt-default" in model_ids
    assert "disabled-provider-model" not in model_ids
    assert all("apiKeyEnv" not in model and "baseUrlEnv" not in model for model in catalog["models"])
    assert all("providerLabel" not in model for model in catalog["models"])
    assert "NINEROUTER_API_KEY" not in serialized_catalog
    assert "NINEROUTER_BASE_URL" not in serialized_catalog
    assert "http://localhost:20128/v1" not in serialized_catalog
    assert "test-key" not in serialized_catalog
    public_default = next(model for model in catalog["models"] if model["id"] == "gpt-default")
    assert public_default["contextWindow"] == 123456
    assert public_default["maxOutputTokens"] == 4321


def test_missing_provider_config_removes_public_models_and_fails_runtime():
    registry = make_registry(env={"NINEROUTER_BASE_URL": "http://localhost:20128/v1"})

    assert registry.public_catalog()["models"] == []
    with pytest.raises(LLMRegistryError, match="missing runtime configuration"):
        registry.select_model("gpt-default")


def test_reasoning_effort_is_validated_and_mapped():
    registry = make_registry()

    assert registry.select_model("gpt-default", reasoning_effort="high").reasoning_effort == "high"

    with pytest.raises(LLMRegistryError, match="Unsupported reasoning effort"):
        registry.select_model("gpt-default", reasoning_effort="xhigh")


def test_rejects_reasoning_for_non_reasoning_model():
    catalog = [dict(CATALOG[0])]
    catalog[0]["models"] = [
        {
            "id": "plain-model",
            "label": "Plain",
            "displayProvider": "Plain",
            "model": "plain-model",
            "enabled": True,
            "capabilities": {
                "reasoningEffort": {
                    "enabled": False,
                    "userConfigurable": False,
                }
            },
        }
    ]
    registry = OpenAICompatibleProviderRegistry(
        load_provider_catalog(json.dumps(catalog)),
        default_provider_id="ninerouter",
        default_model_id="plain-model",
        env_resolver=lambda name: "value",
    )

    assert registry.select_model("plain-model").reasoning_effort is None
    with pytest.raises(LLMRegistryError, match="does not support reasoning"):
        registry.select_model("plain-model", reasoning_effort="low")
