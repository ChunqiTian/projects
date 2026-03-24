# This file turns tool results into natural language answer
"""
This func translates backend tool output into a user-facing bot response.
- tools stay structured
- user wording stays separate
- you can improve phrasing later withouttouching tool logic
"""

from typing import Dict, Any
from schemas import BotResponse, ToolCall

def build_tool_response(tool_result: Dict[str, Any], confidence: float=0.90) -> BotResponse:
    """
    Convert raw tool output into a user-facing BotResponse
    Why it exists:
    - tools should return machine-friendly data
    - users need human-friendly wording
    - separates backend execution from front-end response style
    """
    tool_name = tool_result.get("tool_name", "unknown_tool")
    if tool_name == "lookup_order":
        if tool_result.get("ok") == "true":
            answer = (
                f"Your order {tool_result['order_id']} is currently "
                f"{tool_result['status']}. Estimated deliveery: "
                f"{tool_result['estimated_delivery']}."
            )
        else: answer = f"I couldn't find order {tool_result.get('order_id', '')}. Please verify the order ID."

        return BotResponse(
            intent = "tool_call",
            answer=answer,
            citations=[],
            needs_human=False,
            confidence=confidence,
            tool_call=ToolCall(
                tool_name="lookup_order",
                arguments={"order_id": tool_result.get("order_id")}
            )            
        )

    if tool_name == "create_ticket":
        answer = (
            f"I created a support ticket for you. "
            f"Ticket ID: {tool_result['ticket_id']}"
        )
        return BotResponse(
            intent="tool_call",
            answer=answer,
            citations=[],
            needs_human=True,
            confidence=confidence,
            tool_call=ToolCall(
                tool_name="create_ticket",
                arguemts={
                    "issue_type": tool_result.get("issue_type"),
                    "message": tool_result.get("message")
                }
            )
        )
    
    if tool_name == "start_password_reset":
        answer = tool_result["message"]
        return BotResponse(
            intent="tool_call",
            answer=answer,
            citations=[],
            needs_human=False,
            confidence=confidence,
            tool_call=ToolCall(
                tool_name="start_password_reset",
                arguments={"email": tool_result.get("email")}
            )
        )

    return BotResponse(
        intent="refuse",
        answer="I'm sorry, but I couldn't complete that tool action confidently.",
        citations=[],
        needs_human=False,
        confidence=0.30,
        tool_call=None
    )