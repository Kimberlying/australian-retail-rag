"""Scripted stand-ins for ``anthropic.Anthropic``: offline, deterministic, inspectable."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(query: str, block_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name="run_sql", input={"query": query})


def message(
    *blocks: SimpleNamespace,
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_read: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        model="claude-opus-5",
        stop_reason=stop_reason,
        content=list(blocks),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=0,
        ),
    )


class _Stream:
    def __init__(self, final: SimpleNamespace, deltas: list[str]):
        self._final = final
        self.text_stream: Iterator[str] = iter(deltas)

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def get_final_message(self) -> SimpleNamespace:
        return self._final


class ScriptedAnthropic:
    """Returns the scripted messages in order and records every request."""

    def __init__(self, responses: list[SimpleNamespace], *, deltas: list[str] | None = None):
        self.responses = list(responses)
        self.deltas = deltas
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(create=self._create, stream=self._stream)
        )

    def _next(self, kwargs: dict[str, Any]) -> SimpleNamespace:
        # Snapshot the message list: the caller keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs.get("messages", []))})
        return self.responses.pop(0)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        return self._next(kwargs)

    def _stream(self, **kwargs: Any) -> _Stream:
        final = self._next(kwargs)
        text = "".join(block.text for block in final.content if block.type == "text")
        return _Stream(final, self.deltas if self.deltas is not None else [text])


def install(monkeypatch: pytest.MonkeyPatch, fake: ScriptedAnthropic) -> ScriptedAnthropic:
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: fake)
    return fake
