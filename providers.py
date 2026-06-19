"""Shared LLM provider configuration and a LangChain chat-model factory.

The web app uses the raw provider SDKs for one-shot summaries; the agent layer
(``agent.py``) needs LangChain ``BaseChatModel`` instances to drive the
deepagents harness. Both read the same ``PROVIDERS`` table so a single API key
configures every feature.
"""

from __future__ import annotations

import os

# provider -> config. OpenAI/Qwen/DeepSeek share the OpenAI-compatible API.
PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5-20251001",
        "models": [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-opus-4-7",
        ],
        "openai_compatible": False,
    },
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "openai_compatible": True,
    },
    "qwen": {
        "label": "Qwen (DashScope)",
        "env_key": "DASHSCOPE_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen2.5-72b-instruct"],
        "openai_compatible": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "openai_compatible": True,
    },
}


def provider_configured(provider: str) -> bool:
    cfg = PROVIDERS.get(provider)
    return bool(cfg and os.environ.get(cfg["env_key"]))


def first_available_provider() -> str | None:
    for name in PROVIDERS:
        if provider_configured(name):
            return name
    return None


def resolve_provider_model(provider: str | None, model: str | None) -> tuple[str | None, str]:
    """Resolve a (provider, model) pair, honoring 'auto' and provider defaults."""
    provider = (provider or "auto").strip().lower()
    if provider in ("auto", "", "extractive"):
        provider = first_available_provider() or ""
    if provider not in PROVIDERS:
        return None, ""
    chosen_model = (model or "").strip() or PROVIDERS[provider]["default_model"]
    return provider, chosen_model


# Each provider here is a distinct model family; an LLM judge should not share the
# generator's family (self-preference bias). Preference order for picking a judge.
_JUDGE_PREFERENCE = ("openai", "anthropic", "qwen", "deepseek")


def default_judge_model(gen_provider: str | None) -> tuple[str | None, str]:
    """Pick a configured provider from a DIFFERENT family than the generator."""
    gen_provider = (gen_provider or "").lower()
    for p in _JUDGE_PREFERENCE:
        if p != gen_provider and provider_configured(p):
            return p, PROVIDERS[p]["default_model"]
    return None, ""


def resolve_judge(gen_provider: str, gen_model: str, judge_provider: str | None = None,
                  judge_model: str | None = None) -> tuple[str, str, bool]:
    """Resolve the (provider, model, cross_family) to use as an LLM judge.

    Prefers an explicit judge, else a different-family configured provider, else
    falls back to the generator (cross_family=False) when nothing else is set.
    """
    if judge_provider:
        jp, jm = resolve_provider_model(judge_provider, judge_model)
        if jp:
            return jp, jm, jp != gen_provider
    jp, jm = default_judge_model(gen_provider)
    if jp:
        return jp, jm, True
    return gen_provider, gen_model, False


# LLM-as-judge PANEL: diverse, capable models spanning families. (gpt-5.3 is not
# released; gpt-5.1 is the latest GPT-5.x. qwen3-max is the strong Qwen3 judge.)
JUDGE_PANEL_CANDIDATES = [
    ("openai", "gpt-5-mini"),
    ("openai", "gpt-5.1"),
    ("qwen", "qwen-plus-latest"),
    ("qwen", "qwen3-max"),
    ("deepseek", "deepseek-chat"),
]


def judge_panel(gen_provider: str | None, *, max_judges: int = 6) -> list[tuple[str, str]]:
    """Build a panel of configured judges from DIFFERENT families than the generator.

    A multi-judge panel (diverse families/models) averages out any single model's
    idiosyncratic strictness/bias. Excludes the generator's own family.
    """
    gen_provider = (gen_provider or "").lower()
    panel = [(p, m) for p, m in JUDGE_PANEL_CANDIDATES
             if p != gen_provider and provider_configured(p)]
    return panel[:max_judges]


class ProviderError(RuntimeError):
    """Raised when a chat model cannot be constructed for the agent layer."""


def build_chat_model(provider: str, model: str, *, temperature: float = 0.0,
                     max_tokens: int = 4096, timeout: int = 120):
    """Build a LangChain ``BaseChatModel`` for the given provider/model.

    Raises ``ProviderError`` with an actionable message when the provider is
    unknown, the SDK adapter is missing, or the API key is unset.
    """
    cfg = PROVIDERS.get(provider)
    if cfg is None:
        raise ProviderError(f"Unknown provider: {provider!r}")
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        raise ProviderError(
            f"{cfg['env_key']} is not set. Export it to use the {cfg['label']} agent."
        )

    if cfg.get("openai_compatible"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ProviderError(
                "langchain-openai is not installed. Run `pip install langchain-openai` "
                "to use OpenAI-compatible providers (OpenAI/Qwen/DeepSeek) with the agent."
            ) from exc
        base_url = os.environ.get(cfg.get("base_url_env", ""), cfg.get("base_url"))
        kwargs = {"model": model, "api_key": api_key, "base_url": base_url,
                  "timeout": timeout, "max_retries": 2}
        # Reasoning models (gpt-5*, o1/o3) use max_completion_tokens and only the
        # default temperature; classic chat models use max_tokens + temperature.
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    # Anthropic
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ProviderError(
            "langchain-anthropic is not installed. Run `pip install langchain-anthropic` "
            "to use Claude with the agent."
        ) from exc
    return ChatAnthropic(
        model=model, api_key=api_key,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout, max_retries=2,
    )


def agent_dependencies_available() -> bool:
    """True when the deepagents harness can be imported."""
    try:
        import deepagents  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True
