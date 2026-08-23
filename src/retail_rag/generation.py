from __future__ import annotations

import os

from .config import anthropic_model
from .models import RetrievedChunk


SYSTEM_PROMPT = """You are a careful Australian retail research assistant.
Answer only from the supplied evidence. If the evidence is insufficient, say so.
Keep public facts separate from synthetic data. Include a short Sources section
using the source names supplied in the context."""


def build_context(retrievals: list[RetrievedChunk]) -> str:
    if not retrievals:
        return "No evidence was retrieved."
    blocks = []
    for item in retrievals:
        blocks.append(
            f"[source={item.chunk.source}; chunk={item.chunk.chunk_id}; score={item.score:.3f}]\n"
            f"{item.chunk.text}"
        )
    return "\n\n".join(blocks)


def generate_with_claude(question: str, retrievals: list[RetrievedChunk]) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=anthropic_model(),
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Evidence:\n{build_context(retrievals)}"
                ),
            }
        ],
    )
    text_blocks = [block.text for block in message.content if getattr(block, "type", "") == "text"]
    return "\n".join(text_blocks).strip() or None


def local_preview(question: str, retrievals: list[RetrievedChunk]) -> str:
    if not retrievals:
        return "No matching evidence was retrieved. Try a more specific question."
    lines = [
        "Claude is not configured, so this is a local retrieval preview.",
        f"Question: {question}",
        "",
        "Retrieved evidence:",
    ]
    for item in retrievals:
        lines.append(f"- {item.chunk.source} ({item.score:.3f}): {item.chunk.text}")
    return "\n".join(lines)
