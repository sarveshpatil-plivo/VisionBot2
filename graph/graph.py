"""
LangGraph agent graph definition.
Wires all nodes together with conditional edges.
"""

import logging
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from graph.state import AgentState
from graph.nodes import (
    classify_intent,
    generate_hyde_node,
    retrieve_node,
    rerank_node,
    compress_node,
    check_confidence_node,
    generate_answer_node,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def build_graph(
    openai_client: AsyncOpenAI,
    qdrant_client: QdrantClient,
    embedder,
    settings,
) -> StateGraph:
    """Build and compile the LangGraph agent."""

    # ── Wrap nodes with dependencies ─────────────────────────────────────────

    async def node_classify(state):
        return await classify_intent(
            state,
            client=openai_client,
            model=settings.llm_mini_model,
        )

    async def node_hyde(state):
        return await generate_hyde_node(
            state,
            client=openai_client,
            embedder=embedder,
            model=settings.llm_mini_model,
        )

    def node_retrieve(state):
        return retrieve_node(state, qdrant=qdrant_client, top_k=settings.top_k_retrieve)

    def node_rerank(state):
        return rerank_node(state, top_n=settings.top_k_rerank)

    async def node_compress(state):
        return await compress_node(
            state,
            client=openai_client,
            model=settings.llm_mini_model,
        )

    def node_confidence(state):
        return check_confidence_node(state, threshold=settings.confidence_threshold)

    async def node_answer(state):
        return await generate_answer_node(
            state,
            client=openai_client,
            model=settings.llm_model,
            qdrant=qdrant_client,
        )

    def node_reject_off_topic(state):
        """Short-circuit for non-voice queries — no retrieval, no LLM cost."""
        return {
            "answer": "I can only help with Plivo Voice API questions — things like call failures, SIP issues, WebRTC, audio quality, DTMF, call routing, and similar topics. What voice issue can I help you with?",
            "citations": [],
            "suggested_action": "",
            "related_tickets": [],
            "confidence_score": 0.0,
            "confidence_factors": {},
        }

    def node_ask_clarification(state):
        """Return the clarification question — graph pauses here for user input."""
        return {
            "answer": state.get("clarification_question", "Could you provide more details?"),
            "awaiting_clarification": True,
            "citations": [],
            "suggested_action": "",
            "related_tickets": [],
        }

    def node_increment_retry(state):
        return {"retry_count": (state.get("retry_count") or 0) + 1}

    # ── Routing functions ─────────────────────────────────────────────────────

    def route_after_classify(state) -> str:
        if not state.get("is_voice_related", True):
            return "reject_off_topic"
        if state.get("is_ambiguous") and not state.get("chat_history"):
            return "ask_clarification"
        return "generate_hyde"

    def route_after_confidence(state) -> str:
        retry = state.get("retry_count") or 0
        if state.get("confidence_score", 1.0) < settings.confidence_threshold and retry < MAX_RETRIES:
            return "increment_retry"
        return "generate_answer"

    # ── Build graph ───────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", node_classify)
    graph.add_node("reject_off_topic", node_reject_off_topic)
    graph.add_node("ask_clarification", node_ask_clarification)
    graph.add_node("generate_hyde", node_hyde)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("rerank", node_rerank)
    graph.add_node("compress", node_compress)
    graph.add_node("check_confidence", node_confidence)
    graph.add_node("increment_retry", node_increment_retry)
    graph.add_node("generate_answer", node_answer)

    # Entry
    graph.set_entry_point("classify_intent")

    # Edges
    graph.add_conditional_edges("classify_intent", route_after_classify, {
        "reject_off_topic": "reject_off_topic",
        "ask_clarification": "ask_clarification",
        "generate_hyde": "generate_hyde",
    })
    graph.add_edge("reject_off_topic", END)
    graph.add_edge("ask_clarification", END)  # Pause — frontend sends follow-up
    graph.add_edge("generate_hyde", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "check_confidence")
    graph.add_conditional_edges("check_confidence", route_after_confidence, {
        "increment_retry": "increment_retry",
        "generate_answer": "generate_answer",
    })
    graph.add_edge("increment_retry", "generate_hyde")  # Retry with incremented count
    graph.add_edge("generate_answer", END)

    # Compile with in-memory checkpointer (sufficient for 20 users)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
