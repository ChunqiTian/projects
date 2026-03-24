# router.py - simple intent routing (heuristics first)

import re 
from typing import Literal

Intent = Literal["faq", "policy", "order", "escalate", "refuse"]
# faq - frequently asked questions (general queries)
ORDER_ID_PATTERN = r"\b\d{5,}\b" # Simple pattern for order IDs (5+ digits) 

def route_intent(user_text: str) -> Intent:
    """
    Heuristic routing for step 1
    In step 3, you can replace/augment with LLM-based intent classification
    """
    t = user_text.lower()

    if any(x in t for x in ["return policy", "refund policy", "shipping policy", "cancel subscription"]):
        return "policy"
    
    if re.search(ORDER_ID_PATTERN, t) or any(x in t for x in ["my order", "order status", "track", "tracking"]):
        return "order"
    
    if any(x in t for x in ["refund", "return", "shipping", "delivery", "exchange", "account", "login"]):
        return "faq"
    
    return "faq" # default route (safe fallback)

