# guardrails.py - rules for refusal / escalation
import re
from typing import Optional

# A small list of cues that often mean escalation is appropriate
ESCALATE_CUES = [
    "human", "agent", "representative", "manager", "complaint",
    "angry", "furious", "lawsuit", "report", "scam", "fraud"
]

# Requests we should refuse (scope) even before RAG exists
OUT_OF_SCOPE_PATTERNS = [
    r"\bmedical\b", r"\blegal\b", r"\bdiagnose\b", r"\buse\b",
    r"\bcredit card\b", r"\bpassword\b", r"\bsocial security\b"
]
# r"\b...\b" means match the text inside as a whole word

def should_escalate(user_text: str) -> bool:
    """Resurn True if the user likely wants a human or this is high-risk."""
    t = user_text.lower()
    return any(cue in t for cue in ESCALATE_CUES)

def out_of_scope_reason(user_text: str) -> Optional[str]:
    """Return a reason string if the request is out-of-scope; otherwise None."""
    t = user_text.lower()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, t):
            return "This request is outside what I can safely help with here."
    return None

