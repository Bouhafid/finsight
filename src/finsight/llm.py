"""OpenAI-compatible chat client: Azure OpenAI in production, Ollama locally."""

import os
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI


@dataclass(frozen=True)
class LLM:
    client: OpenAI
    model: str

    def complete_json(self, messages: list[dict], max_tokens: int) -> tuple[str, str]:
        """Return the raw JSON text and the finish reason."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "azure":
        client = OpenAI(
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )
        return LLM(client, os.environ["AZURE_OPENAI_DEPLOYMENT"])

    if provider == "ollama":
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
            api_key="ollama",
            # A small model on CPU is slow; callers handle retries themselves.
            timeout=float(os.getenv("LLM_TIMEOUT", "600")),
            max_retries=0,
        )
        return LLM(client, os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))

    raise ValueError(f"Unknown LLM provider: {provider}")
