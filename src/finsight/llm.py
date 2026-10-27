"""LLM clients behind one interface: Ollama locally, Azure OpenAI or Amazon Bedrock hosted."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import openai
from openai import OpenAI


class LLMError(Exception):
    """A transient provider failure (network, rate limit, server error): retrying may help."""


class LLMTimeout(LLMError):
    """The provider did not answer in time; the same request would take just as long."""


def is_transient(status_code: int) -> bool:
    """Rate limits and server errors pass; a bad key or an unknown model does not."""
    return status_code == 429 or status_code >= 500


class LLM(Protocol):
    model: str

    def complete_json(
        self, messages: list[dict], max_tokens: int, schema: dict | None = None
    ) -> tuple[str, str]:
        """Return the raw JSON text and the finish reason ("stop" or "length").

        `schema` is the expected JSON shape; providers that can enforce it do.
        Configuration errors (bad key, unknown model) are raised as they are.
        """
        ...


@dataclass(frozen=True)
class OpenAICompatibleLLM:
    """Ollama and Azure OpenAI both speak the OpenAI chat completions API."""

    client: OpenAI
    model: str

    def complete_json(
        self, messages: list[dict], max_tokens: int, schema: dict | None = None
    ) -> tuple[str, str]:
        # JSON mode only: small local models handle a full JSON schema poorly.
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
            )
        except openai.APITimeoutError as error:
            raise LLMTimeout(str(error)) from error
        except openai.APIConnectionError as error:
            raise LLMError(str(error)) from error
        except openai.APIStatusError as error:
            if is_transient(error.status_code):
                raise LLMError(str(error)) from error
            raise

        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "bedrock":
        from finsight.bedrock import BedrockClaude

        return BedrockClaude.from_env()

    if provider == "azure":
        client = OpenAI(
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )
        return OpenAICompatibleLLM(client, os.environ["AZURE_OPENAI_DEPLOYMENT"])

    if provider == "ollama":
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
            api_key="ollama",
            # A small model on CPU is slow; callers handle retries themselves.
            timeout=float(os.getenv("LLM_TIMEOUT", "600")),
            max_retries=0,
        )
        return OpenAICompatibleLLM(client, os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))

    raise ValueError(f"Unknown LLM provider: {provider}")
