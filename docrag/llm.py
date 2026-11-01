"""Language-model provider construction."""

import os
from urllib.parse import urlparse


def make_llm(provider: str = "anthropic", model: str | None = None, base_url: str | None = None):
    """Create an Anthropic or OpenAI-compatible chat model."""
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic provider")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or "claude-haiku-5-5", max_tokens=600)
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI-compatible support is optional; install with `pip install "docrag[openai]"`'
            ) from exc
        local_endpoint = bool(base_url) and urlparse(base_url).hostname in {"localhost", "127.0.0.1", "::1"}
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not local_endpoint:
            raise RuntimeError("OPENAI_API_KEY is required for this OpenAI-compatible provider")
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            base_url=base_url,
            max_tokens=600,
            api_key=api_key or "ollama",
        )
    raise ValueError(f"unknown provider: {provider}")
