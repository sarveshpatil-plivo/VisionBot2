"""
Fetch a single Zendesk ticket + all comments by ticket ID.
Used when an agent pastes a Zendesk URL into the chat.
"""

import logging
import httpx

logger = logging.getLogger(__name__)


async def fetch_ticket_by_id(ticket_id: str, subdomain: str, email: str, api_key: str) -> str:
    """
    Fetch ticket details + all comments from Zendesk API.
    Returns formatted plain text ready for injection into the LLM context.
    """
    base_url = f"https://{subdomain}.zendesk.com/api/v2"
    auth = (f"{email}/token", api_key)

    async with httpx.AsyncClient(auth=auth, timeout=20) as client:
        ticket_resp = await client.get(f"{base_url}/tickets/{ticket_id}.json")
        ticket_resp.raise_for_status()
        ticket = ticket_resp.json()["ticket"]

        comments_resp = await client.get(f"{base_url}/tickets/{ticket_id}/comments.json")
        comments_resp.raise_for_status()
        comments = comments_resp.json().get("comments", [])

    subject = ticket.get("subject", "N/A")
    status = ticket.get("status", "N/A")
    created = ticket.get("created_at", "N/A")
    tags = ", ".join(ticket.get("tags", []))

    lines = [
        f"=== LIVE TICKET #{ticket_id} ===",
        f"Subject: {subject}",
        f"Status:  {status}",
        f"Created: {created}",
        f"Tags:    {tags}",
        "",
        "--- Thread ---",
    ]

    for c in comments:
        visibility = "Public" if c.get("public") else "Internal"
        body = (c.get("plain_body") or c.get("body") or "").strip()
        # Trim very long comments to avoid flooding context
        if len(body) > 1500:
            body = body[:1500] + " … [truncated]"
        lines.append(f"[{visibility} | {c.get('created_at', '')}]\n{body}")
        lines.append("")

    return "\n".join(lines)
