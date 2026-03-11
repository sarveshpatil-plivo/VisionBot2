"""Pydantic request/response schemas."""

from typing import Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    session_id: str                          # For multi-turn routing
    metadata_filters: dict[str, str] = {}   # Optional: {"product": "voice_api"}


class Citation(BaseModel):
    source: str = "ticket"              # "ticket" or "docs"
    # Ticket fields
    ticket_id: Optional[str] = None
    subject: Optional[str] = None
    resolution_type: Optional[str] = None
    csat_score: Optional[float] = None
    zendesk_url: Optional[str] = None
    # Docs fields
    page_title: Optional[str] = None
    section_title: Optional[str] = None
    url: Optional[str] = None
    # Common
    excerpt: str = ""


class ConfidenceFactors(BaseModel):
    retrieval_similarity: float
    resolution_consistency: str
    recency: str
    source_quality: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    confidence_score: float = 0.0
    confidence_factors: dict[str, Any] = {}
    suggested_action: Optional[str] = None
    related_tickets: list[dict] = []
    intent: Optional[str] = None
    awaiting_clarification: bool = False
    session_id: str


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    helpful: bool                   # True = helpful, False = not helpful
    ticket_ids: list[str] = []      # Ticket IDs cited in the answer


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    version: str = "1.0.0"
