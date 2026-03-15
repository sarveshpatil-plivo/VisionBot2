"""FastAPI server — streaming query endpoint, feedback, health."""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from api.config import settings
from api.cost_tracker import log_query_cost
from api.models import FeedbackRequest, HealthResponse, QueryRequest, QueryResponse
from graph.graph import build_graph
from ingestion.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

LOG_FILE = Path("query_logs.jsonl")

# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(title="SupportIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for internal ngrok access; protected by API token
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────

_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
_qdrant_client = QdrantClient(path=settings.qdrant_path)
_embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
_graph = build_graph(_openai_client, _qdrant_client, _embedder, settings)

# ── Auth ──────────────────────────────────────────────────────────────────────

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.api_secret_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials


# ── Logging ───────────────────────────────────────────────────────────────────

def log_query(session_id: str, question: str, final_state: dict, latency_ms: int):
    timings = final_state.get("timings") or {}
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question": question,
        "intent": final_state.get("intent", ""),
        "voice_related": final_state.get("is_voice_related", True),
        "confidence": round(final_state.get("confidence_score", 0.0), 3),
        "citations": len(final_state.get("citations") or []),
        "retry_count": final_state.get("retry_count", 0),
        "steps_ms": {k: v for k, v in timings.items() if v},
        "total_ms": latency_ms,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    # Also print a compact summary to stdout for live monitoring
    steps = " → ".join(f"{k.replace('_ms','')}:{v}" for k, v in timings.items() if v)
    logger.info(f"[{session_id[:8]}] {question[:60]!r} | {steps} | total:{latency_ms}ms | conf:{entry['confidence']}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
# Routes registered under both / and /api so dev proxy and prod build both work

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse, include_in_schema=False)
async def health():
    try:
        _qdrant_client.get_collections()
        qdrant_status = "ok"
    except Exception:
        qdrant_status = "error"
    return HealthResponse(status="ok", qdrant=qdrant_status)


@app.post("/query")
@app.post("/api/query", include_in_schema=False)
async def query(
    request: QueryRequest,
    _token: str = Depends(verify_token),
):
    """
    Streaming SSE endpoint.
    Streams answer tokens, then sends a final JSON metadata chunk.
    """
    start = time.monotonic()

    initial_state = {
        "question": request.question,
        "session_id": request.session_id,
        # chat_history intentionally omitted — LangGraph MemorySaver preserves it across turns
        "metadata_filters": request.metadata_filters,
        "retry_count": 0,
        "injected_context": None,
        "is_voice_related": True,
        "is_ambiguous": False,
        "awaiting_clarification": False,
        "confidence_score": 0.0,
        "confidence_factors": {},
        "citations": [],
        "suggested_action": "",
        "related_tickets": [],
        "error": None,
    }

    config = {"configurable": {"thread_id": request.session_id}}

    async def event_stream():
        final_state = {}
        try:
            async for event in _graph.astream_events(initial_state, config=config, version="v2"):
                kind = event.get("event")
                if kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output", {})

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        # Stream the extracted answer text word-by-word for a smooth UX
        # (generate_answer uses JSON mode so raw tokens would expose JSON structure)
        answer = final_state.get("answer", "")
        if answer:
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.015)  # ~65 words/sec

        latency_ms = int((time.monotonic() - start) * 1000)

        # Send metadata as final chunk
        metadata = {
            "type": "done",
            "citations": final_state.get("citations", []),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "confidence_factors": final_state.get("confidence_factors", {}),
            "suggested_action": final_state.get("suggested_action", ""),
            "related_tickets": final_state.get("related_tickets", []),
            "reasoning": final_state.get("reasoning", ""),
            "intent": final_state.get("intent", ""),
            "awaiting_clarification": final_state.get("awaiting_clarification", False),
            "latency_ms": latency_ms,
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        log_query(request.session_id, request.question, final_state, latency_ms)
        log_query_cost(
            session_id=request.session_id,
            question=request.question,
            query_type=final_state.get("query_type", ""),
            cost_entries=final_state.get("cost_entries") or [],
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/feedback")
@app.post("/api/feedback", include_in_schema=False)
async def feedback(
    request: FeedbackRequest,
    _token: str = Depends(verify_token),
):
    """
    Record agent feedback (helpful / not helpful).
    In V2 this will update ticket retrieval weights in Qdrant.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": request.session_id,
        "question": request.question,
        "rating": request.rating,
        "comment": request.comment,
        "ticket_ids": request.ticket_ids,
    }
    with open("feedback_logs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"Feedback logged: rating={request.rating} for session {request.session_id}")
    return {"status": "ok"}


@app.get("/tickets/{ticket_id}")
@app.get("/api/tickets/{ticket_id}", include_in_schema=False)
async def get_ticket(
    ticket_id: str,
    _token: str = Depends(verify_token),
):
    """Fetch full ticket payload from Qdrant by ticket_id."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = _qdrant_client.scroll(
            collection_name="ticket_resolutions",
            scroll_filter=Filter(
                must=[FieldCondition(key="ticket_id", match=MatchValue(value=ticket_id))]
            ),
            limit=1,
        )
        if results[0]:
            return results[0][0].payload
    except Exception as e:
        logger.error(f"Ticket fetch error: {e}")

    raise HTTPException(status_code=404, detail="Ticket not found")


# ── Serve React frontend (must be last) ───────────────────────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
