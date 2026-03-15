"""
LangGraph node functions. Each function handles one reasoning step.
All nodes receive the full AgentState and return a partial state update.
"""

import asyncio
import json
import logging
import math
import re
import time
from typing import Any

from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from api.cost_tracker import estimate_tokens, make_entry
from graph.state import AgentState
from retrieval.hyde import generate_hyde
from retrieval.hybrid_retriever import retrieve
from retrieval.reranker import rerank
from retrieval.compressor import compress_chunks
from retrieval.zendesk_fetcher import fetch_ticket_by_id

_ZENDESK_URL_RE = re.compile(r'https?://[a-zA-Z0-9-]+\.zendesk\.com/agent/tickets/(\d+)')

# Signals that indicate the query contains exact technical identifiers.
# For these queries HyDE is bypassed — the raw question is embedded directly.
# Reason: HyDE hallucinates a generic resolution for exact error strings/UUIDs,
# pulling in wrong tickets. BM25 sparse vectors do exact string matching better.
_TECHNICAL_SIGNAL_RE = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'  # UUID / call UUID
    r'|\bSIP\s*[3-5]\d{2}\b'                                             # SIP response codes
    r'|\bHTTP\s*[3-5]\d{2}\b'                                            # HTTP status codes
    r'|Traceback \(most recent|Exception in thread|at com\.|at org\.'    # Stack traces
    r'|EHOSTUNREACH|ECONNREFUSED|ECONNRESET|ETIMEDOUT|ENOTFOUND'         # POSIX network errors
    r'|HangupCauseCode|NORMAL_CLEARING|NO_ROUTE_DESTINATION'             # SIP/telephony hangup causes
    r'|USER_BUSY|NO_ANSWER|CALL_REJECTED|UNALLOCATED_NUMBER'             # More SIP causes
    r'|websocket\.send|websocket\.receive|ASGI message'                  # ASGI/WebSocket errors
    r'|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',                        # IP addresses
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

CLASSIFY_HYDE_PROMPT = """You are a classifier and retrieval assistant for SupportIQ — an internal tool for Plivo support agents answering questions about the Plivo Voice API.

Return ONLY a JSON object:
{
  "is_voice_related": true|false,
  "query_type": "ticket_search|product_question",
  "intent": "troubleshoot|explain|find-similar|summarize|other",
  "is_ambiguous": true|false,
  "needs_clarification": true|false,
  "clarification_question": "question to ask if is_ambiguous OR needs_clarification, else null",
  "hyde_text": "hypothetical document text for embedding (see instructions below)"
}

STEP 1 — is_voice_related:
- true if the query is about voice calls, SIP, WebRTC, PSTN, call quality, audio, DTMF, call routing, IVR, caller ID, call recordings, voice SDK, outbound/inbound calls, call failures, Zentrunk, CPS limits, voice capacity, or any other Plivo Voice API topic.
- false ONLY when the query has absolutely no conceivable connection to voice: pure SMS campaigns, pure invoice/payment questions, greetings, general small talk.
- WHEN IN DOUBT: set is_voice_related=true, needs_clarification=true, and ask a short question to understand how this relates to voice. Never reject something that might be voice-related — ask first.
If false: set query_type="ticket_search", intent="other", is_ambiguous=false, needs_clarification=false, hyde_text=<the original question>.

STEP 2 — query_type (if voice_related):
- "product_question": user wants to understand a feature/concept.
  Examples: "What is Play API?", "How does DTMF work?", "What are Zentrunk rate limits?"
- "ticket_search": user has a problem or wants past cases.
  Examples: "Play API returning 404", "SIP 403 Forbidden", "WebRTC drops after 30s"
If unsure, mark is_ambiguous=true and ask one clarifying question.

STEP 3 — intent (if voice_related):
troubleshoot | explain | find-similar | summarize

STEP 4 — is_ambiguous:
Set true ONLY if the query has two or more completely different possible interpretations.
Example: "issue with calls" — could mean setup, quality, billing, routing.
Ask the single most useful clarifying question.

STEP 5 — needs_clarification (voice_related=true, NOT ambiguous):
Set true ONLY when the query describes a vague symptom with ZERO diagnostic details — no error codes,
no SDK/platform, no call direction, no reproduction pattern.
- "my calls are dropping" → needs_clarification=true (no error code, no SDK, no pattern)
- "audio is bad on calls" → needs_clarification=true (no platform, no direction, no codec info)
- "SIP 403 on outbound calls" → needs_clarification=false (has error code + direction — proceed)
- "WebRTC drops after 30s" → needs_clarification=false (has SDK + specific symptom — proceed)
- "How does DTMF work?" → needs_clarification=false (product question — no details needed)
If needs_clarification=true: ask for the single most valuable missing piece.
  e.g. "What error code or behaviour are you seeing?" / "Which SDK — Browser, Android, or iOS?"
Do NOT set needs_clarification=true if the query contains ANY specific technical signal.

STEP 6 — hyde_text:
Write a short hypothetical document (2-4 sentences) that would perfectly answer this query.
- For ticket_search: write as if you are the resolution note in a Zendesk ticket. Include likely root cause and fix.
- For product_question: write as if you are a Plivo docs page explaining the feature.
- Keep it dense with technical terms relevant to the query — this text will be embedded for vector search.
If is_ambiguous=true or needs_clarification=true or is_voice_related=false, set hyde_text to the original question verbatim."""

DOCS_ANSWER_PROMPT = """You are SupportIQ — a senior Plivo Voice API engineer. When an agent asks how something works, give a direct, complete explanation.

Rules:
1. If a LIVE TICKET is present at the top of the context, treat it as the primary context for your answer.
2. Explain directly — the agent wants to understand how something works, not a doc summary.
3. Structure your answer: what it is → how it works → key constraints or gotchas.
4. Cite docs by title: "Per the Confluence doc 'Voice API Overview'..." or "Per the Plivo docs..."
5. Do NOT reference support tickets for product explanation questions.
6. If the docs don't cover something, say so — do not hallucinate.
7. End with one actionable next step or relevant doc link.

Return ONLY a JSON object:
{
  "reasoning": "1 sentence: which docs were relevant and why",
  "answer": "clear, structured explanation with doc citations",
  "suggested_action": "one actionable next step or doc reference",
  "citations": [
    {
      "ticket_id": null,
      "subject": "doc title",
      "excerpt": "most relevant sentence from the doc",
      "resolution_type": null
    }
  ]
}"""

ANSWER_PROMPT = """You are SupportIQ — a senior Plivo Voice API engineer who has investigated hundreds of support cases. When an agent brings you a problem, diagnose the root cause first, then cite the evidence.

Rules:
1. If a LIVE TICKET is present at the top of the context, that is the primary problem — your full attention goes there.
2. Start from symptoms: identify exactly what the customer is experiencing before naming a cause.
3. Lead with your diagnosis — the 2-3 most likely root causes in order of probability. Bring in ticket evidence to support your diagnosis, not the other way around.
4. Only cite a source if its root cause genuinely matches the described symptoms.
5. Do NOT introduce products, features, or tools (e.g. PHLO) unless the customer mentioned them or multiple sources strongly point to them.
6. Cite ticket sources as "Ticket #36286 showed the same pattern..." — evidence supporting your diagnosis.
7. Cite doc sources as "Per the Confluence SOP on X..." or "Jira VT-123 confirms..."
8. Provide one concrete suggested_action the agent can take right now.
9. If context is insufficient, state what information would help — do not hallucinate.
10. Temporal awareness: each source shows a "Solved:" date. If an older source contradicts a newer source on the same behaviour, explicitly note that platform behaviour has likely changed and rely on the newer source.
11. Ryuk navigation: Ryuk is Plivo's internal admin dashboard — agents look up customers by email, Auth ID, or username. When suggesting investigation steps, use exact Ryuk paths:
    - Account state: Account Management → Accounts → [search email] → Account Settings tab.
      Key fields: Enabled (Yes/No), Voice Enabled, Zentrunk Enabled, CPS Allowed, Zentrunk CPS Allowed, Voice Carrier Type, Plivo Carrier Fallback, Risk Status, Account Notes.
    - Call records (Voice API): Voice → Fetch CDRs (search by call UUID or date range)
    - Call visual trace: Voice → Call Timeline (search by call UUID — shows full SIP/media event sequence)
    - PCAP analysis: Voice → PCAP Analyser v2 (load PCAP for a call UUID — check SIP responses, RTP, carrier leg failures)
    - CPS limits: Voice → Voice CPS utilisation (check if customer is hitting their CPS ceiling)
    - Zentrunk trunks: Zentrunk → Trunks (credentials, IP whitelist, registration status)
    - Zentrunk call records: Zentrunk → Fetch CDRs
    - Zentrunk routing: Zentrunk → Routing (check routing profile assigned to account)
    - Number issues: Numbers → Number Management or Number History
    - Finance / limit charges: Finance → Recurring Charge (add charges for CPS/limit increases)
    Always prefer "In Ryuk → [section] → [subsection], look up [customer email], check [specific field]" over vague "check the console" instructions.

EXAMPLE — what good looks like:
---
Q: "Customer getting SIP 403 Forbidden on outbound calls. Started yesterday."

BAD answer: "Based on Ticket #36286, this may be a SIP authentication issue. See also Ticket #44123."

GOOD answer: "SIP 403 on outbound calls means the carrier rejected the auth request — this is a carrier-side failure, not Plivo. Root causes in order of likelihood: (1) Plivo media server IP not whitelisted on the carrier portal — most common trigger for failures that 'just started' since IPs can rotate; (2) SIP trunk credentials changed on the carrier side but not updated in Plivo config; (3) From header format mismatch. Tickets #36286 and #44123 both matched this exact pattern and were resolved by re-adding Plivo's current media server IPs to the carrier whitelist. Start there: pull the current Plivo media IPs from the console and verify they're listed on the carrier portal."
---

Return ONLY a JSON object:
{
  "reasoning": "1-2 sentences: what are the symptoms, what is the most likely diagnosis and why",
  "answer": "expert diagnosis with root causes first, then cited evidence",
  "suggested_action": "one concrete next step the agent can take right now",
  "citations": [
    {
      "ticket_id": "... or null for non-ticket sources",
      "subject": "...",
      "excerpt": "most relevant sentence",
      "resolution_type": "..."
    }
  ]
}"""


# ── Node: fetch_ticket_context ───────────────────────────────────────────────

async def fetch_ticket_context_node(state: AgentState, settings) -> dict:
    """
    If the user's message contains a Zendesk ticket URL, fetch that ticket live
    and inject its full thread as context. Cleans the URL from the question.
    """
    question = state["question"]
    match = _ZENDESK_URL_RE.search(question)
    if not match:
        return {"injected_context": None}

    ticket_id = match.group(1)
    # Remove the URL from the question; fall back to a sensible default
    cleaned = _ZENDESK_URL_RE.sub("", question).strip()
    if not cleaned:
        cleaned = f"What is the solution for this ticket? (#{ticket_id})"

    try:
        context = await fetch_ticket_by_id(
            ticket_id=ticket_id,
            subdomain=settings.zendesk_subdomain,
            email=settings.zendesk_email,
            api_key=settings.zendesk_api_key,
        )
        logger.info(f"Fetched live ticket #{ticket_id} from Zendesk ({len(context)} chars)")
        return {
            "injected_context": context,
            "question": cleaned,
        }
    except Exception as e:
        logger.error(f"Failed to fetch ticket #{ticket_id}: {e}")
        return {"injected_context": None}


# ── Node: classify_intent ────────────────────────────────────────────────────

async def classify_intent(state: AgentState, client: AsyncOpenAI, model: str) -> dict:
    """Classify intent, query_type, and generate HyDE text — all in one LLM call."""
    t0 = time.monotonic()
    history_text = ""
    if state.get("chat_history"):
        last = state["chat_history"][-3:]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in last)

    user_msg = state["question"]
    if history_text:
        user_msg = f"Conversation so far:\n{history_text}\n\nNew question: {state['question']}"

    # If a live ticket was injected, prepend a summary so HyDE is grounded in the actual problem
    injected = state.get("injected_context")
    if injected:
        # Give the LLM the first ~800 chars of the ticket thread for classification + HyDE
        snippet = injected[:800]
        user_msg = f"[Live ticket submitted for analysis]\n{snippet}\n\n---\nAgent's question: {user_msg}"

    classify_cost_entry = None
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFY_HYDE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if response.usage:
            classify_cost_entry = make_entry(
                "classify+hyde", model,
                response.usage.prompt_tokens, response.usage.completion_tokens,
            )
    except Exception as e:
        logger.error(f"Classify+HyDE failed: {e}")
        result = {
            "is_voice_related": True,
            "query_type": "ticket_search",
            "intent": "troubleshoot",
            "is_ambiguous": False,
            "clarification_question": None,
            "hyde_text": state["question"],
        }

    is_voice = result.get("is_voice_related", True)
    query_type = result.get("query_type", "ticket_search")
    hyde_text = result.get("hyde_text") or state["question"]

    # HyDE bypass: if query contains exact technical signals (UUIDs, SIP/HTTP error codes,
    # stack traces, IP addresses), embed the raw question instead of the hallucinated
    # hypothetical. BM25 sparse vectors handle exact string matching better than a
    # generic LLM-generated resolution document.
    hyde_bypassed = False
    if _TECHNICAL_SIGNAL_RE.search(state["question"]):
        hyde_text = state["question"]
        hyde_bypassed = True

    costs = list(state.get("cost_entries") or [])
    if classify_cost_entry:
        costs.append(classify_cost_entry)

    is_ambiguous = result.get("is_ambiguous", False)
    needs_clarification = result.get("needs_clarification", False)

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(
        f"Classify+HyDE: voice={is_voice} | type={query_type} | intent={result.get('intent')} "
        f"| ambiguous={is_ambiguous} | needs_clarification={needs_clarification} "
        f"| hyde_bypass={hyde_bypassed} | {elapsed}ms"
    )
    return {
        "is_voice_related": is_voice,
        "query_type": query_type,
        "intent": result.get("intent", "troubleshoot"),
        "is_ambiguous": is_ambiguous,
        "needs_clarification": needs_clarification,
        "clarification_question": result.get("clarification_question"),
        "awaiting_clarification": is_ambiguous or needs_clarification,
        "hyde_text": hyde_text,
        "cost_entries": costs,
        "timings": {**(state.get("timings") or {}), "classify_ms": elapsed},
    }


# ── Node: generate_hyde ──────────────────────────────────────────────────────

async def generate_hyde_node(
    state: AgentState,
    client: AsyncOpenAI,
    embedder,
    model: str,
) -> dict:
    """Embed the hyde_text produced by classify_intent. Pure embed — no LLM call."""
    t0 = time.monotonic()
    # hyde_text already set by classify_intent (merged call)
    hyde_text = state.get("hyde_text") or state["question"]
    dense, sparse = await embedder.embed_query_async(hyde_text)

    embed_entry = make_entry("embed_hyde", embedder.dense_model, estimate_tokens(hyde_text))
    costs = list(state.get("cost_entries") or [])
    costs.append(embed_entry)

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Embed: {elapsed}ms")
    return {
        "hyde_dense": dense,
        "hyde_sparse": sparse,
        "cost_entries": costs,
        "timings": {**(state.get("timings") or {}), "hyde_ms": elapsed},
    }


# ── Node: retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: AgentState, qdrant: QdrantClient, top_k: int) -> dict:
    """Guaranteed-lane hybrid retrieval using embedded HyDE text."""
    t0 = time.monotonic()
    chunks = retrieve(
        client=qdrant,
        dense_vector=state["hyde_dense"],
        sparse_vector=state["hyde_sparse"],
        top_k=top_k,
        filters=state.get("metadata_filters") or None,
        query_type=state.get("query_type", "ticket_search"),
    )
    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Retrieve: {len(chunks)} chunks in {elapsed}ms")
    return {
        "retrieved_chunks": chunks,
        "timings": {**(state.get("timings") or {}), "retrieve_ms": elapsed},
    }


# ── Node: rerank ─────────────────────────────────────────────────────────────

def rerank_node(state: AgentState, top_n: int) -> dict:
    """Cross-encoder reranking of retrieved chunks."""
    t0 = time.monotonic()
    reranked = rerank(
        query=state["question"],
        chunks=state["retrieved_chunks"],
        top_n=top_n,
    )
    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Rerank: {len(reranked)} chunks in {elapsed}ms")
    return {
        "reranked_chunks": reranked,
        "timings": {**(state.get("timings") or {}), "rerank_ms": elapsed},
    }


# ── Node: compress ───────────────────────────────────────────────────────────

async def compress_node(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
    qdrant: QdrantClient = None,
) -> dict:
    """
    Contextual compression — only for product_question queries (verbose Confluence/Slack docs).

    For ticket_search: compression is SKIPPED. Raw chunks go directly to GPT-4o.
    Reason: GPT-4o-mini compression strips technical "noise" — stack traces, SIP headers,
    call UUIDs, log timestamps — that GPT-4o needs for accurate diagnosis. GPT-4o has a
    128k context window; 4 raw ticket chunks (~3k tokens) is trivial. Removing compression
    saves ~2.2s and preserves the exact technical signals the model needs.

    For product_question: compression is kept. Confluence/Slack/Jira docs can be verbose
    and repetitive; trimming to relevant sentences genuinely helps here.
    """
    t0 = time.monotonic()

    if state.get("query_type") == "ticket_search":
        # Parent-child expansion: for each winning ticket, fetch ALL its sibling chunks
        # (problem + investigation + resolution) and combine into one context block.
        # This ensures GPT-4o sees the full diagnostic thread, not just the one chunk
        # that won the reranker lottery.
        _CHUNK_ORDER = {"problem": 0, "investigation": 1, "resolution": 2}
        winning_chunks = state.get("reranked_chunks", [])[:4]
        expanded = []
        for c in winning_chunks:
            tid = c.get("ticket_id")
            if not tid or not qdrant:
                c = dict(c)
                c["compressed_text"] = c.get("text", "")
                expanded.append(c)
                continue

            siblings = _fetch_all_ticket_chunks(qdrant, str(tid))
            siblings.sort(key=lambda x: _CHUNK_ORDER.get(x.get("chunk_type", ""), 99))

            parts = []
            for s in siblings:
                label = s.get("chunk_type", "context").upper()
                parts.append(f"[{label}]\n{s.get('text', '')}")

            c = dict(c)
            c["compressed_text"] = "\n\n".join(parts) if parts else c.get("text", "")
            expanded.append(c)

        elapsed = int((time.monotonic() - t0) * 1000)
        logger.info(f"Compress: parent-child expansion {len(winning_chunks)} tickets → {len(expanded)} expanded in {elapsed}ms")
        return {
            "compressed_chunks": expanded,
            "timings": {**(state.get("timings") or {}), "compress_ms": elapsed},
        }

    # product_question: compress Confluence/Slack/Jira docs
    input_chunks = state["reranked_chunks"][:4]
    compressed = await compress_chunks(
        client=client,
        query=state["question"],
        chunks=input_chunks,
        model=model,
    )

    # Estimate cost: no response.usage exposed by compress_chunks, so use text length heuristic
    compress_input = sum(estimate_tokens(c.get("text", "")) for c in input_chunks)
    compress_output = min(200 * len(input_chunks), 800)  # max_tokens=200 per chunk
    compress_entry = make_entry("compress", model, compress_input, compress_output)
    costs = list(state.get("cost_entries") or [])
    costs.append(compress_entry)

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Compress: {elapsed}ms")
    return {
        "compressed_chunks": compressed,
        "cost_entries": costs,
        "timings": {**(state.get("timings") or {}), "compress_ms": elapsed},
    }


# ── Node: check_confidence ───────────────────────────────────────────────────

_DOC_SOURCE_QUALITY = {"confluence": 1.0, "docs": 0.9, "slack": 0.7, "jira": 0.6}


def check_confidence_node(state: AgentState, threshold: float) -> dict:
    """
    Compute confidence score from multiple signals.
    Uses different formula for product_question vs ticket_search because
    ticket-specific signals (resolution_type, csat, recency) are meaningless for docs.
    """
    chunks = state.get("reranked_chunks", [])
    if not chunks:
        return {
            "confidence_score": 0.0,
            "confidence_factors": {"error": "No chunks retrieved"},
        }

    # Factor 1: Top reranker score normalised 0-1 via sigmoid.
    # bge-reranker-v2-m3 returns raw logits (any range); sigmoid maps them cleanly to 0-1.
    top_score = chunks[0].get("rerank_score", 0)
    retrieval_sim = 1 / (1 + math.exp(-top_score))

    if state.get("query_type") == "product_question":
        # ── Docs path ──────────────────────────────────────────────────────────
        # Ticket signals (resolution_type, csat, recency) don't exist on doc chunks.
        # Use retrieval relevance (60%) + source quality (40%).
        src_scores = [_DOC_SOURCE_QUALITY.get(c.get("source", ""), 0.5) for c in chunks]
        avg_src = sum(src_scores) / len(src_scores)

        score = round(min(max(retrieval_sim * 0.60 + avg_src * 0.40, 0.0), 1.0), 3)
        factors = {
            "retrieval_similarity": round(retrieval_sim, 3),
            "source_quality": f"avg {round(avg_src, 2)} "
                              f"({'Confluence/Docs' if avg_src > 0.8 else 'Jira/Slack'})",
        }

    else:
        # ── Ticket path ────────────────────────────────────────────────────────
        # Factor 2: Resolution consistency (how many top chunks share same resolution_type)
        res_types = [c.get("resolution_type") for c in chunks if c.get("resolution_type")]
        if res_types:
            most_common = max(set(res_types), key=res_types.count)
            consistency = res_types.count(most_common) / len(res_types)
        else:
            consistency = 0.5

        # Factor 3: Recency (average age, penalise old tickets)
        from datetime import datetime, timezone
        ages = []
        for c in chunks:
            created = c.get("created_at")
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    ages.append((datetime.now(timezone.utc) - dt).days)
                except Exception:
                    pass
        recency = 1.0 - min(sum(ages) / max(len(ages), 1) / 365, 1.0) if ages else 0.7

        # Factor 4: CSAT
        _CSAT_MAP = {"good": 5, "bad": 1, "offered": 3, "unoffered": 3}
        csats_raw = [c.get("csat_score") for c in chunks if c.get("csat_score") is not None]
        csats = [_CSAT_MAP.get(str(v).lower(), 3) for v in csats_raw]
        csat_norm = (sum(csats) / len(csats)) / 5.0 if csats else 0.7

        score = round(min(max(
            retrieval_sim * 0.40
            + consistency * 0.25
            + recency * 0.15
            + csat_norm * 0.20,
            0.0), 1.0), 3)

        factors = {
            "retrieval_similarity": round(retrieval_sim, 3),
            "resolution_consistency": f"{'High' if consistency > 0.7 else 'Medium' if consistency > 0.4 else 'Low'}",
            "recency": f"{'< 90 days' if recency > 0.75 else '< 1 year' if recency > 0.5 else '> 1 year'}",
            "source_quality": f"avg CSAT {round(sum(csats)/len(csats), 1) if csats else 'N/A'}",
        }

    logger.info(f"Confidence [{state.get('query_type','ticket_search')}]: {score} | {factors}")
    return {
        "confidence_score": score,
        "confidence_factors": factors,
        "timings": {**(state.get("timings") or {}), "confidence_ms": 0},
    }


# ── Node: generate_answer ────────────────────────────────────────────────────

async def generate_answer_node(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
    qdrant: QdrantClient,
) -> dict:
    """CoT answer generation using compressed chunks as context."""
    t0 = time.monotonic()
    chunks = state.get("compressed_chunks") or state.get("reranked_chunks", [])

    DOC_SOURCES = {"docs", "confluence", "jira", "slack", "ryuk"}

    # Build context — live ticket first (if present), then RAG chunks
    context_parts = []
    injected = state.get("injected_context")
    if injected:
        context_parts.append(
            f"=== LIVE TICKET (submitted by agent for analysis) ===\n{injected}\n==="
        )

    for i, c in enumerate(chunks, 1):
        text = c.get("compressed_text") or c.get("text", "")
        source = c.get("source", "")
        if source in DOC_SOURCES:
            if source == "confluence":
                label = f"Confluence — {c.get('title', 'N/A')}"
            elif source == "jira":
                label = f"Jira {c.get('project', '')} — {c.get('summary', c.get('title', 'N/A'))}"
            elif source == "slack":
                label = f"Slack #{c.get('channel', 'N/A')}"
            elif source == "ryuk":
                label = f"Ryuk — {c.get('nav_path', c.get('section', 'N/A'))}"
            else:
                label = f"Plivo Docs — {c.get('page_title', 'N/A')} > {c.get('section_title', '')}"
            context_parts.append(f"[{label}]\n{text}")
        else:
            context_parts.append(
                f"[Ticket #{c.get('ticket_id', 'N/A')} — {c.get('subject', 'N/A')}]\n"
                f"Type: {c.get('chunk_type')} | Product: {c.get('product')} | "
                f"Region: {c.get('region')} | CSAT: {c.get('csat_score', 'N/A')} | "
                f"Solved: {c.get('solved_at') or c.get('created_at', 'N/A')}\n"
                f"{text}"
            )
    context = "\n\n---\n\n".join(context_parts)

    # Pick prompt based on query type
    prompt = DOCS_ANSWER_PROMPT if state.get("query_type") == "product_question" else ANSWER_PROMPT

    # Include recent chat history for multi-turn awareness
    messages = [{"role": "system", "content": prompt}]
    for turn in (state.get("chat_history") or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Question: {state['question']}\n\nContext (tickets, docs, Confluence, Jira, Slack):\n{context}",
    })

    answer_cost_entry = None
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if response.usage:
            answer_cost_entry = make_entry(
                "answer", model,
                response.usage.prompt_tokens, response.usage.completion_tokens,
            )
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        result = {
            "reasoning": "Could not generate answer",
            "answer": "I encountered an error generating an answer. Please try rephrasing your question.",
            "suggested_action": None,
            "citations": [],
        }

    # Fetch related tickets from the same cluster
    related = _fetch_related(qdrant, chunks)

    # Append this turn to chat_history for multi-turn context
    answer_text = result.get("answer", "")
    history = list(state.get("chat_history") or [])
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": answer_text})

    # Enrich LLM citations with metadata from the actual chunks
    llm_citations = result.get("citations", [])
    chunk_map = {str(c.get("ticket_id")): c for c in chunks if c.get("ticket_id")}

    enriched_citations = []
    for cit in llm_citations:
        tid = cit.get("ticket_id")
        if not tid or str(tid).lower() in ("none", "null", ""):
            continue  # Doc citations — handled by the supplementary loop below
        chunk = chunk_map.get(str(tid), {})
        enriched_citations.append({
            "source": "ticket",
            "ticket_id": tid,
            "subject": cit.get("subject"),
            "excerpt": cit.get("excerpt", ""),
            "resolution_type": cit.get("resolution_type") or chunk.get("resolution_type"),
            "csat_score": chunk.get("csat_score"),
            "zendesk_url": chunk.get("zendesk_url"),
        })

    # Append non-ticket chunks that appeared in context as supplementary citations
    DOC_SOURCES = {"docs", "confluence", "jira", "slack", "ryuk"}
    for c in chunks:
        if c.get("source") in DOC_SOURCES:
            enriched_citations.append({
                "source": c.get("source"),
                "page_title": c.get("title") or c.get("page_title") or c.get("nav_path") or c.get("summary", ""),
                "section_title": c.get("section_title") or c.get("channel") or c.get("project", ""),
                "url": c.get("url", ""),
                "excerpt": (c.get("compressed_text") or c.get("text", ""))[:200],
            })

    costs = list(state.get("cost_entries") or [])
    if answer_cost_entry:
        costs.append(answer_cost_entry)

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Answer generation: {elapsed}ms")
    return {
        "answer": answer_text,
        "reasoning": result.get("reasoning", ""),
        "citations": enriched_citations,
        "suggested_action": result.get("suggested_action", ""),
        "related_tickets": related,
        "chat_history": history,
        "cost_entries": costs,
        "timings": {**(state.get("timings") or {}), "answer_ms": elapsed},
    }


def _fetch_all_ticket_chunks(qdrant: QdrantClient, ticket_id: str) -> list[dict]:
    """
    Fetch all sibling chunks (problem + investigation + resolution) for a ticket.
    Searches both Qdrant collections so parent-child context is always complete.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    all_chunks = []
    for collection in ("ticket_problems", "ticket_resolutions"):
        try:
            results = qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="ticket_id", match=MatchValue(value=ticket_id))]
                ),
                limit=5,
                with_vectors=False,
            )
            for point in results[0]:
                all_chunks.append(point.payload)
        except Exception as e:
            logger.warning(f"Failed to fetch {collection} chunks for ticket {ticket_id}: {e}")
    return all_chunks


def _fetch_related(qdrant: QdrantClient, chunks: list[dict]) -> list[dict]:
    """Fetch other tickets in the same cluster as the top result."""
    if not chunks:
        return []

    cluster_id = chunks[0].get("cluster_id")
    if not cluster_id:
        return []

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = qdrant.scroll(
            collection_name="ticket_resolutions",
            scroll_filter=Filter(
                must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))]
            ),
            limit=5,
        )
        related = []
        top_ticket_id = chunks[0].get("ticket_id")
        for point in results[0]:
            if point.payload.get("ticket_id") != top_ticket_id:
                related.append({
                    "ticket_id": point.payload.get("ticket_id"),
                    "subject": point.payload.get("subject"),
                    "resolution_type": point.payload.get("resolution_type"),
                })
        return related[:4]
    except Exception as e:
        logger.warning(f"Related ticket fetch failed: {e}")
        return []
