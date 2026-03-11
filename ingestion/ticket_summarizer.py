"""
Use GPT-4o to extract structured fields from each ticket's conversation thread.
Results are cached to summaries_cache.jsonl to avoid re-processing on incremental runs.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

CACHE_FILE = Path("summaries_cache.jsonl")
BATCH_SIZE = 50  # Concurrent GPT-4o-mini calls

SYSTEM_PROMPT = """You are a technical support analyst. Given a Zendesk ticket conversation,
extract structured information and return ONLY a JSON object with these exact fields:

{
  "problem_description": "One sentence: what the customer reported",
  "root_cause": "One sentence: the actual underlying cause identified",
  "resolution_summary": "One sentence: what fixed the issue",
  "suggested_action": "Actionable next step if someone sees this issue again",
  "diagnostic_steps": ["step 1 the agent took", "step 2", "..."],
  "issue_type": "one of: dtmf, sms, voice_call, sip, api, authentication, latency, other",
  "product": "one of: voice_api, messaging_api, sip_trunking, plivo_cx, other",
  "region": "one of: US, EMEA, APAC, LATAM, unknown",
  "resolution_type": "one of: carrier_escalation, config_fix, code_fix, user_error, known_issue, other"
}

Rules:
- Be concise and technical
- If a field cannot be determined, use null
- Return ONLY the JSON object, no markdown, no explanation"""


def _build_ticket_text(ticket: dict) -> str:
    """Build the full conversation text to send to GPT-4o."""
    parts = [f"Subject: {ticket.get('subject', 'N/A')}"]

    for i, comment in enumerate(ticket.get("comments", [])):
        author_type = "Customer" if i == 0 else "Support"
        body = (comment.get("plain_body") or comment.get("body") or "").strip()
        if body:
            parts.append(f"\n[{author_type}]: {body}")

    return "\n".join(parts)


def _load_cache() -> dict[str, dict]:
    """Load existing summaries cache keyed by ticket_id."""
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cache[str(entry["ticket_id"])] = entry
    logger.info(f"Loaded {len(cache)} cached summaries")
    return cache


def _append_cache(entry: dict):
    """Append a single summary to cache file."""
    with open(CACHE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class TicketSummarizer:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.cache = _load_cache()

    async def _summarize_one(self, ticket: dict) -> dict:
        """Summarize a single ticket using GPT-4o."""
        ticket_id = str(ticket["id"])

        if ticket_id in self.cache:
            return self.cache[ticket_id]

        text = _build_ticket_text(ticket)

        summary = None
        for attempt in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Analyze this ticket:\n\n{text}"},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                summary = json.loads(response.choices[0].message.content)
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Summarization failed for ticket {ticket_id}: {e}")
                    break

        if summary is None:
            summary = {
                "problem_description": ticket.get("subject"),
                "root_cause": None,
                "resolution_summary": None,
                "suggested_action": None,
                "diagnostic_steps": [],
                "issue_type": "other",
                "product": "other",
                "region": "unknown",
                "resolution_type": "other",
            }

        entry = {"ticket_id": ticket_id, **summary}
        _append_cache(entry)
        self.cache[ticket_id] = entry
        return entry

    async def summarize_batch(self, tickets: list[dict]) -> list[dict]:
        """Summarize a list of tickets with concurrency control."""
        results = []
        semaphore = asyncio.Semaphore(BATCH_SIZE)

        async def bounded(ticket):
            async with semaphore:
                return await self._summarize_one(ticket)

        tasks = [bounded(t) for t in tickets]
        results = await asyncio.gather(*tasks)
        return list(results)

    def summarize_all(self, tickets: list[dict]) -> list[dict]:
        """Synchronous entry point."""
        return asyncio.run(self.summarize_batch(tickets))
