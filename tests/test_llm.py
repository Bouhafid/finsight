from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from finsight.bedrock import THINKING_HEADROOM, BedrockClaude
from finsight.llm import LLMError, LLMTimeout

MESSAGES = [
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "Verify this claim."},
]
SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class FakeMessages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def bedrock(outcome) -> BedrockClaude:
    client = SimpleNamespace(messages=FakeMessages(outcome))
    return BedrockClaude(client, "anthropic.claude-haiku-5-5", "low")


def response(stop_reason: str, text: str = '{"ok": true}'):
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=text),
    ]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def status_error(cls, code: int):
    request = httpx2.Request("POST", "https://bedrock.example")
    return cls("error", response=httpx2.Response(code, request=request), body=None)


def test_bedrock_sends_system_apart_and_enforces_the_schema():
    llm = bedrock(response("end_turn"))

    assert llm.complete_json(MESSAGES, max_tokens=900, schema=SCHEMA) == ('{"ok": true}', "stop")

    sent = llm.client.messages.kwargs
    assert sent["system"] == "Return JSON."
    assert sent["messages"] == [MESSAGES[1]]
    assert sent["max_tokens"] == 900 + THINKING_HEADROOM
    assert sent["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }


def test_bedrock_reports_truncated_output_like_openai():
    llm = bedrock(response("max_tokens", '{"atomic_'))

    assert llm.complete_json(MESSAGES, max_tokens=900) == ('{"atomic_', "length")


def test_bedrock_refusal_is_a_failed_call():
    with pytest.raises(LLMError):
        bedrock(response("refusal", "")).complete_json(MESSAGES, max_tokens=900)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic.APITimeoutError(httpx2.Request("POST", "https://x")), LLMTimeout),
        (status_error(anthropic.RateLimitError, 429), LLMError),
        (status_error(anthropic.InternalServerError, 500), LLMError),
    ],
)
def test_bedrock_transient_errors_become_llm_errors(error, expected):
    with pytest.raises(expected):
        bedrock(error).complete_json(MESSAGES, max_tokens=900)


def test_bedrock_configuration_errors_are_not_hidden():
    # A bad key must stop the run, not be counted as a rejected claim.
    with pytest.raises(anthropic.AuthenticationError):
        bedrock(status_error(anthropic.AuthenticationError, 401)).complete_json(
            MESSAGES, max_tokens=900
        )
