"""Claude on Amazon Bedrock, through Anthropic's SDK (Bedrock Mantle endpoint)."""

import os
from dataclasses import dataclass

import anthropic
from anthropic import AnthropicBedrockMantle

from finsight.llm import LLMError, LLMTimeout, is_transient

DEFAULT_MODEL = "anthropic.claude-haiku-5-5"

# Adaptive thinking counts against max_tokens: leave it room so the JSON is never cut off.
THINKING_HEADROOM = 8000


@dataclass(frozen=True)
class BedrockClaude:
    client: AnthropicBedrockMantle
    model: str
    effort: str = "medium"

    @classmethod
    def from_env(cls) -> "BedrockClaude":
        # Credentials: a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK, or the usual AWS chain.
        client = AnthropicBedrockMantle(
            aws_region=os.environ["AWS_REGION"],
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )
        return cls(
            client,
            os.getenv("BEDROCK_MODEL", DEFAULT_MODEL),
            os.getenv("LLM_EFFORT", "medium"),
        )

    def complete_json(
        self, messages: list[dict], max_tokens: int, schema: dict | None = None
    ) -> tuple[str, str]:
        # The Messages API takes the system prompt apart from the conversation.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        output_config: dict = {"effort": self.effort}
        if schema:
            # Structured outputs: the answer is guaranteed to be JSON matching `schema`.
            output_config["format"] = {"type": "json_schema", "schema": schema}

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens + THINKING_HEADROOM,
                system=system,
                messages=[m for m in messages if m["role"] != "system"],
                output_config=output_config,
            )
        except anthropic.APITimeoutError as error:
            raise LLMTimeout(str(error)) from error
        except anthropic.APIConnectionError as error:
            raise LLMError(str(error)) from error
        except anthropic.APIStatusError as error:
            if is_transient(error.status_code):
                raise LLMError(str(error)) from error
            raise

        if response.stop_reason == "refusal":
            raise LLMError("the model declined the request")

        text = "".join(block.text for block in response.content if block.type == "text")
        return text, "length" if response.stop_reason == "max_tokens" else "stop"
