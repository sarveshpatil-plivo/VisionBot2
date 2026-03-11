"""PII scrubbing using Microsoft Presidio."""

import logging
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

ENTITIES_TO_REDACT = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "CRYPTO",
    "IBAN_CODE",
    "CREDIT_CARD",
    "URL",
]


class PIIScrubber:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def scrub(self, text: str) -> str:
        """Remove PII from a text string."""
        if not text or not text.strip():
            return text
        try:
            results = self.analyzer.analyze(
                text=text,
                entities=ENTITIES_TO_REDACT,
                language="en",
            )
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text
        except Exception as e:
            logger.warning(f"PII scrub failed: {e} — returning original text")
            return text

    def scrub_ticket(self, ticket: dict) -> dict:
        """Scrub PII from subject and all comment bodies."""
        ticket = dict(ticket)  # shallow copy

        if ticket.get("subject"):
            ticket["subject"] = self.scrub(ticket["subject"])

        if ticket.get("description"):
            ticket["description"] = self.scrub(ticket["description"])

        scrubbed_comments = []
        for comment in ticket.get("comments", []):
            c = dict(comment)
            if c.get("body"):
                c["body"] = self.scrub(c["body"])
            if c.get("plain_body"):
                c["plain_body"] = self.scrub(c["plain_body"])
            scrubbed_comments.append(c)

        ticket["comments"] = scrubbed_comments
        return ticket
