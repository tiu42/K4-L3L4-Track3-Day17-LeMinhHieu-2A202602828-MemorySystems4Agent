from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import DEFAULT_MODELS, ProviderConfig, normalize_provider

# Provider -> env vars holding its API key (first non-empty wins).
_API_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "custom": ("CUSTOM_API_KEY", "OPENAI_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
    "openrouter": ("OPENROUTER_API_KEY",),
}

_BASE_URL_ENV = {
    "custom": "CUSTOM_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - paths: repo root, dataset directory, state directory
    - compact memory: token threshold + number of recent messages kept verbatim
    - persistent memory guardrail: minimum confidence before writing to `User.md`
    - provider settings for the main model and the judge model
    - `live_mode`: only True when explicitly enabled AND credentials exist
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = 0.6
    summary_max_items: int = 8
    max_superseded_history: int = 5
    live_mode: bool = False
    # Live only: expose the `save_user_fact` tool to the LLM (extra calls + schema tokens).
    live_memory_tool: bool = False


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    raw_provider = _env(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai")
    provider = normalize_provider(raw_provider)
    same_as_fallback = fallback is not None and provider == fallback.provider

    model_name = _env(f"{prefix}_MODEL") or (fallback.model_name if same_as_fallback else DEFAULT_MODELS[provider])
    temperature = float(_env(f"{prefix}_TEMPERATURE", "0") or 0)

    api_key = _env(f"{prefix}_API_KEY")
    if api_key is None:
        for env_name in _API_KEY_ENV[provider]:
            api_key = _env(env_name)
            if api_key:
                break

    base_url = _env(f"{prefix}_BASE_URL")
    if base_url is None and provider in _BASE_URL_ENV:
        base_url = _env(_BASE_URL_ENV[provider])

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` + environment variables and return a populated LabConfig.

    Env knobs:
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE / LLM_API_KEY / LLM_BASE_URL
    - JUDGE_PROVIDER / JUDGE_MODEL / ... (defaults to the main model)
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
    - OLLAMA_BASE_URL, CUSTOM_BASE_URL / CUSTOM_API_KEY
    - LAB_MODE=live to call the real model (default: deterministic offline mode)
    - COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES, PROFILE_CONFIDENCE_THRESHOLD
    - LAB_STATE_DIR (default `<root>/state`), e.g. to run several benchmarks in parallel
    - LIVE_MEMORY_TOOL=1 to give the LLM the `save_user_fact` tool (off by default, see REPORT.md)
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    state_dir = Path(_env("LAB_STATE_DIR") or root / "state")
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)
    live_requested = (_env("LAB_MODE", "offline") or "offline").lower() == "live"

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(_env("COMPACT_THRESHOLD_TOKENS", "1000")),
        compact_keep_messages=int(_env("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        profile_confidence_threshold=float(_env("PROFILE_CONFIDENCE_THRESHOLD", "0.6")),
        live_mode=live_requested and model.is_usable(),
        live_memory_tool=(_env("LIVE_MEMORY_TOOL", "0") or "0").lower() not in ("0", "false", "no"),
    )
