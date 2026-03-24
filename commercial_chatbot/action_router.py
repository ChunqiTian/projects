# This file decides whether to answer from docs, call a tool, or escalate
"""
Functions
1. extract_order_id(text: str) # this tries to find a 5 digit order ID
2. extract_email(text:str) # this looks for an email pattern
3. rounte_action(user_text: str) # this is the most important step 8 decision function - decides the route
    - route: tool_call | docs_answer | escalate
"""

# Step 8: Tool calling / action rounting
"""
- answer from knowledge base
- call a tool
- escalate to a human
RAG + decision layer + workflow/action layer
Add:
- a tool schema / tool registry
- mock tools for now
- an action router that decides what kind of request this is
- an orchestrator that picks one of: answer from docs, call a tool, escalate
Design thinking process:
1. Decide the possible outcomes - eg. policy_answer; tool_call; escalate; refuse
2. Separate decision from execution 
    - decide what should happen - eg order lookup; password reset; FAQ
    - actually do things - eg. call lookup_order("123") or start_password_reset("a@b.com")
    - route_action() - decides the action
    - execute_tool_action() - actually calls the tool
3. Start with mock tools, eg
    - lookup_order(order_id)
    - create_ticket(issue_type, message)
    - start_password_reset(email)
4. Keep tool outputs structured - return {"tool_name": "loopup_order", "order_id": order_id...}
    - for debug, log, test
5. Keep final esponse formatting separate 
Files: main.py | schemas| handler| actin_router| tools| tool_executor| response_builder| retrieve| answer| guardrails| router
"""
# Step 11 - logging
"""
Log to detect intent, chosen route, required arguments, and escalation reasons. 
"""

import re
from typing import Dict, Any, Optional

from logger_config import setup_logger
logger = setup_logger("router") # Create a logger for this module

def extract_order_id(text: str) -> Optional[str]:
    """
    Try to extract an order ID from the user's message
    Why this function exists:
    - order lookup needs a specific identifier
    - rounting alone is not enough; tools also need arguments
    """
    match = re.search(r"]b\d{5}\b", text) 
        #- \b = word boundary; \d{5} = exactly 5 digits; \b=another word boundary
    if match: return match.group(0) # group(0) returns the entire match; if 1: 1st captured group; 2 for 2nd captured g
    return None

def extract_email(text: str) -> Optional[str]:
    """
    Try to extract an email address from the user's message
    Why this function exists:
    - password reset flow needs an email
    - regex is a simple practical way o parse it here
    """
    match = re.search(r"[\w.-]+@[\w\.-]+\.\w+", text) 
        #r for raw string match [...]@[...].(one or more)words; eg abc@email.com
        #\w.- means word or . or -
    if match: return match.group(0)
    return None

def route_action(user_text: str) -> Dict[str, Any]:
    """
    Decide the next system action
    Possinle outputs: docs_answer | tool_call| escalate
    Why this func exists:
    - centralize workflow deciion logic
    - easier to modify than scattering if/else across many files
    Returns: dict describing the route decision
    """    
    # Log raw incoming text at info level
    logger.info(f"Routing started for user_text: {user_text}")
    text = user_text.lower()
    # Order status workflow
    if "order" in text and ("where" in text or "status" in text or "track" in text):
        logger.info("Detected intent: order_status") # Log the detected high-level intent

        order_id = extract_order_id(user_text) # extract order ID
        if order_id: # If order ID is found, choose tool call
            return {
                "route": "tool_call",
                "tool_name": "lookup_order",
                "arguments": {"order_id": order_id},
                "confidence": 0.95
            }
        else:
            # If order intent exists but ID is missing, escalate
            logger.warning("Escalation triggered: order help requested but no valid order ID found.")
            return {
                "route": "escalate",
                "reason": "User wants order help but no valid order ID was found.",
                "confidence": 0.70
            }
        
    # Password reset workflow
    if "password" in text and ("reset" in text or "forgot" in text):
        logger.info("Detected intent: password_reset") # Log the detected high-level intent
        email = extract_email(user_text)
        if email: # If email exists, start password reset tool
            logger.info(f"Tool route selected: start_password_reset | email={email}")
            return {
                "route": "tool_call",
                "tool_name": "start_password_reset",
                "arguments": {"email": email},
                "confidence": 0.95
            }
        else: # Missing email may need escalation or follow-up depending on your design
            logger.warning("Escalation triggered: password reset requested but no email found.")
            return {
                "route": "escalate",
                "reason": "Password reset requested but no email was found.",
                "confidence": 0.72
            }

    # Ticket creation workflow
    if any (word in text for word in ["broken", "damaged", "charged twice", "billng issue", "complaint"]):
        logger.info("Detected intent: support_issue")
        logger.info("Tool route selected: create_ticket")
        return {
            "route": "tool_call",
            "tool_name": "create_ticket",
            "arguments": {
                "issue_type": "support_issue",
                "message": user_text
            },
            "confidence": 0.88
        }

    # Human escalation workflow
    if any(word in text for word in ["agent", "representative", "human", "speak to someone"]):
        logger.warning("Escalation triggered: user explicitly asked for a human.")
        return {
            "route": "escalate",
            "reason": "User explicitly asked for a human.",
            "confidence": 0.98
        }
    
    # Default: try knowledge-base answer
    logger.info("Defaul route selected: docs_answer")
    return {
        "route": "docs_answer",
        "confidence": 0.80
    }



