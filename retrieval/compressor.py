"""
Contextual compression: extract only the sentences from each chunk
that are relevant to the query. Reduces ~300 tokens → ~40 tokens of signal.
"""

import asyncio
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a text extraction assistant.
Given a query and a passage from a support ticket, extract ONLY the sentences
that are directly relevant to answering the query.

Return just the extracted sentences as plain text. If nothing is relevant, return an empty string.
Do not add explanations or commentary."""


async def _compress_one(
    client: AsyncOpenAI,
    query: str,
    chunk: dict,
    model: str,
) -> dict:
    """Compress a single chunk to relevant sentences."""
    text = chunk.get("text", "")
    if not text.strip():
        return {**chunk, "compressed_text": ""}

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nPassage:\n{text}"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        compressed = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Compression failed for chunk {chunk.get('chunk_id')}: {e}")
        compressed = text[:300]  # Fallback: truncate

    return {**chunk, "compressed_text": compressed}


async def compress_chunks(
    client: AsyncOpenAI,
    query: str,
    chunks: list[dict],
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """Compress all chunks in parallel."""
    tasks = [_compress_one(client, query, c, model) for c in chunks]
    results = await asyncio.gather(*tasks)
    logger.info(f"Compressed {len(chunks)} chunks")
    return list(results)
