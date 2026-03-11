"""
LangGraph node functions. Each function handles one reasoning step.
All nodes receive the full AgentState and return a partial state update.
"""

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from graph.state import AgentState
from retrieval.hyde import generate_hyde
from retrieval.hybrid_retriever import retrieve
from retrieval.reranker import rerank
from retrieval.compressor import compress_chunks

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

INTENT_PROMPT = """You are a query classifier for a technical support knowledge base.

Classify the user's question and return ONLY a JSON object:
{
  "intent": "troubleshoot|explain|find-similar|summarize",
  "is_ambiguous": true|false,
  "clarification_question": "question to ask if ambiguous, else null"
}

Intent definitions:
- troubleshoot: user has a problem and wants a fix
- explain: user wants to understand how/why something works
- find-similar: user wants to see past tickets like this one
- summarize: user wants an overview of a topic or trend

Ambiguous = the query is too vague to retrieve relevant tickets without more context.
Only mark ambiguous if a single clarifying question would significantly narrow the scope.
Do NOT mark ambiguous if the query is reasonable to answer as-is."""

ANSWER_PROMPT = """You are SupportIQ, an expert technical support analyst.
Answer the question using ONLY the provided ticket context.

Rules:
1. Think step-by-step before answering (show your reasoning briefly)
2. Cite specific ticket IDs for every claim: e.g. "As seen in Ticket #36286..."
3. If multiple tickets show the same root cause, note the pattern
4. Provide a concrete suggested_action the agent can take right now
5. If context is insufficient, say so clearly — do not hallucinate

Return ONLY a JSON object:
{
  "reasoning": "brief step-by-step thinking",
  "answer": "your full answer with ticket citations",
  "suggested_action": "one concrete actionable next step",
  "citations": [
    {
      "ticket_id": "...",
      "subject": "...",
      "excerpt": "most relevant sentence from this ticket",
      "resolution_type": "..."
    }
  ]
}"""


# ── Node: classify_intent ────────────────────────────────────────────────────

async def classify_intent(state: AgentState, client: AsyncOpenAI, model: str) -> dict:
    """Classify intent and detect ambiguity."""
    history_text = ""
    if state.get("chat_history"):
        last = state["chat_history"][-3:]  # Last 3 turns for context
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in last)

    user_msg = state["question"]
    if history_text:
        user_msg = f"Conversation so far:\n{history_text}\n\nNew question: {state['question']}"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        result = {"intent": "troubleshoot", "is_ambiguous": False, "clarification_question": None}

    logger.info(f"Intent: {result.get('intent')} | Ambiguous: {result.get('is_ambiguous')}")
    return {
        "intent": result.get("intent", "troubleshoot"),
        "is_ambiguous": result.get("is_ambiguous", False),
        "clarification_question": result.get("clarification_question"),
        "awaiting_clarification": result.get("is_ambiguous", False),
    }


# ── Node: generate_hyde ──────────────────────────────────────────────────────

async def generate_hyde_node(
    state: AgentState,
    client: AsyncOpenAI,
    embedder,
    model: str,
) -> dict:
    """Generate HyDE text and embed it. On retry, skip HyDE and embed raw query."""
    if state.get("retry_count", 0) > 0:
        # Retry: embed raw question to get different retrieval signal
        hyde_text = state["question"]
    else:
        hyde_text = await generate_hyde(client, state["question"], model=model)

    dense, sparse = await embedder.embed_query_async(hyde_text)

    return {
        "hyde_text": hyde_text,
        "hyde_dense": dense,
        "hyde_sparse": sparse,
    }


# ── Node: retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: AgentState, qdrant: QdrantClient, top_k: int) -> dict:
    """Hybrid retrieval using HyDE embedding."""
    chunks = retrieve(
        client=qdrant,
        dense_vector=state["hyde_dense"],
        sparse_vector=state["hyde_sparse"],
        top_k=top_k,
        filters=state.get("metadata_filters") or None,
    )
    return {"retrieved_chunks": chunks}


# ── Node: rerank ─────────────────────────────────────────────────────────────

def rerank_node(state: AgentState, top_n: int) -> dict:
    """Cross-encoder reranking of retrieved chunks."""
    reranked = rerank(
        query=state["question"],
        chunks=state["retrieved_chunks"],
        top_n=top_n,
    )
    return {"reranked_chunks": reranked}


# ── Node: compress ───────────────────────────────────────────────────────────

async def compress_node(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
) -> dict:
    """Contextual compression of reranked chunks."""
    compressed = await compress_chunks(
        client=client,
        query=state["question"],
        chunks=state["reranked_chunks"],
        model=model,
    )
    return {"compressed_chunks": compressed}


# ── Node: check_confidence ───────────────────────────────────────────────────

def check_confidence_node(state: AgentState, threshold: float) -> dict:
    """
    Compute confidence score from multiple signals.
    Returns score + factors dict for display.
    """
    chunks = state.get("reranked_chunks", [])
    if not chunks:
        return {
            "confidence_score": 0.0,
            "confidence_factors": {"error": "No chunks retrieved"},
        }

    # Factor 1: Top reranker score (normalised 0-1)
    top_score = chunks[0].get("rerank_score", 0)
    retrieval_sim = min(max((top_score + 10) / 20, 0), 1)  # Normalise ms-marco scores

    # Factor 2: Resolution consistency (how many of top-5 share same resolution_type)
    res_types = [c.get("resolution_type") for c in chunks if c.get("resolution_type")]
    if res_types:
        most_common = max(set(res_types), key=res_types.count)
        consistency = res_types.count(most_common) / len(res_types)
    else:
        consistency = 0.5

    # Factor 3: Recency (average days since creation, penalise old tickets)
    from datetime import datetime, timezone
    ages = []
    for c in chunks:
        created = c.get("created_at")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).days
                ages.append(age_days)
            except Exception:
                pass
    recency = 1.0 - min(sum(ages) / max(len(ages), 1) / 365, 1.0) if ages else 0.7

    # Factor 4: CSAT quality
    csats = [c.get("csat_score") for c in chunks if c.get("csat_score") is not None]
    csat_norm = (sum(csats) / len(csats)) / 5.0 if csats else 0.7

    # Weighted composite
    score = (
        retrieval_sim * 0.40
        + consistency * 0.25
        + recency * 0.15
        + csat_norm * 0.20
    )
    score = round(min(max(score, 0.0), 1.0), 3)

    factors = {
        "retrieval_similarity": round(retrieval_sim, 3),
        "resolution_consistency": f"{'High' if consistency > 0.7 else 'Medium' if consistency > 0.4 else 'Low'}",
        "recency": f"{'< 90 days' if recency > 0.75 else '< 1 year' if recency > 0.5 else '> 1 year'}",
        "source_quality": f"avg CSAT {round(sum(csats)/len(csats), 1) if csats else 'N/A'}",
    }

    logger.info(f"Confidence score: {score} | Factors: {factors}")
    return {"confidence_score": score, "confidence_factors": factors}


# ── Node: generate_answer ────────────────────────────────────────────────────

async def generate_answer_node(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
    qdrant: QdrantClient,
) -> dict:
    """CoT answer generation using compressed chunks as context."""
    chunks = state.get("compressed_chunks") or state.get("reranked_chunks", [])

    # Build context — handle both ticket chunks and docs chunks
    context_parts = []
    for i, c in enumerate(chunks, 1):
        text = c.get("compressed_text") or c.get("text", "")
        if c.get("source") == "docs":
            context_parts.append(
                f"[Plivo Docs — {c.get('page_title', 'N/A')} > {c.get('section_title', '')}]\n"
                f"URL: {c.get('url', '')}\n"
                f"{text}"
            )
        else:
            context_parts.append(
                f"[Ticket #{c.get('ticket_id', 'N/A')} — {c.get('subject', 'N/A')}]\n"
                f"Type: {c.get('chunk_type')} | Product: {c.get('product')} | "
                f"Region: {c.get('region')} | CSAT: {c.get('csat_score', 'N/A')}\n"
                f"{text}"
            )
    context = "\n\n---\n\n".join(context_parts)

    # Include recent chat history for multi-turn awareness
    messages = [{"role": "system", "content": ANSWER_PROMPT}]
    for turn in (state.get("chat_history") or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Question: {state['question']}\n\nTicket Context:\n{context}",
    })

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
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
        tid = str(cit.get("ticket_id", ""))
        chunk = chunk_map.get(tid, {})
        enriched_citations.append({
            "source": "ticket",
            "ticket_id": cit.get("ticket_id"),
            "subject": cit.get("subject"),
            "excerpt": cit.get("excerpt", ""),
            "resolution_type": cit.get("resolution_type") or chunk.get("resolution_type"),
            "csat_score": chunk.get("csat_score"),
            "zendesk_url": chunk.get("zendesk_url"),
        })

    # Append any docs chunks that appeared in context (as supplementary citations)
    for c in chunks:
        if c.get("source") == "docs":
            enriched_citations.append({
                "source": "docs",
                "page_title": c.get("page_title"),
                "section_title": c.get("section_title"),
                "url": c.get("url"),
                "excerpt": (c.get("compressed_text") or c.get("text", ""))[:200],
            })

    return {
        "answer": answer_text,
        "citations": enriched_citations,
        "suggested_action": result.get("suggested_action", ""),
        "related_tickets": related,
        "chat_history": history,
    }


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
