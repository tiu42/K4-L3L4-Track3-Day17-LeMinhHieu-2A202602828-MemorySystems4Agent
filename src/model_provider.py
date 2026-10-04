from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

# Common spellings / typos people put in `.env`.
_PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "azure_openai_compatible": "custom",
    "custom": "custom",
    "openai_compatible": "custom",
    "openai-compatible": "custom",
    "compatible": "custom",
    "vllm": "custom",
    "lmstudio": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google_genai": "gemini",
    "google-genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "open_router": "openrouter",
    "open-router": "openrouter",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-5-5",
    "ollama": "qwen2.5:7b",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    def is_usable(self) -> bool:
        """True when enough credentials exist to call the live model."""

        if self.provider == "ollama":
            return True
        if self.provider == "custom":
            return bool(self.base_url)
        return bool(self.api_key)


try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # offline mode works without LangChain
    BaseCallbackHandler = object


class UsageTracker(BaseCallbackHandler):
    """Collect real provider token usage for every LLM call made during one turn.

    Covers the agent's main calls, tool-call round trips, and the hidden
    summarization call of `SummarizationMiddleware` (tagged `lc_source=summarization`).
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.summary_calls = 0
        self._summary_runs: set = set()

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        if (metadata or {}).get("lc_source") == "summarization":
            self._summary_runs.add(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        self.llm_calls += 1
        if run_id in self._summary_runs:
            self.summary_calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.input_tokens += int(usage.get("input_tokens", 0))
                self.output_tokens += int(usage.get("output_tokens", 0))


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; reject unknown providers."""

    key = (value or "").strip().lower().replace(" ", "_")
    if key not in _PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider {value!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
    return _PROVIDER_ALIASES[key]


def build_chat_model(config: ProviderConfig):
    """Instantiate the real LangChain chat model for the selected provider.

    Imports are lazy so offline mode works without every provider SDK installed.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("Provider 'custom' requires a base_url (CUSTOM_BASE_URL).")
            kwargs["base_url"] = config.base_url
            kwargs.setdefault("api_key", "not-needed")
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name, temperature=config.temperature, google_api_key=config.api_key
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model_name, temperature=config.temperature, api_key=config.api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        kwargs = {"model": config.model_name, "temperature": config.temperature, "api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenRouter(**kwargs)

    raise ValueError(f"Unsupported provider: {provider}")
