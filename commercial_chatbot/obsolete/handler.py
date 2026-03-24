# This is the main Step 8 orchestration layer
"""
Function: handle_message(user_text: str)
This is your main controller. 
This is where everything gets tied together
Flow:
1. Guardrails - check for: unsafe| unsupported| out of scope | obviously escalatory cases
2. Action route - Decide wheher user needs: docs answer| tool call | escalaton
3. Execute route 
    - If tool route -> call tool
    - If escalate route -> return escalation
    - If docs route -> use retrieval and answer synthesis
"""
from schemas import BotResponse
from guardrails import should_escalate, out_of_scope_reason
#from action_router import route_action
#from tool_executor import execute_tool_action
#from response_builder import build_tool_response
#from rag.retrieve_hybrid import retriever
#from rag.answer import answer_from_evidence
from router import route_intent
from memory import ConversationState, update_state_from_user_message, build_context_summary


def handle_message(user_text: str, state: ConversationState) -> BotResponse:
    """
    step 8
    Main orchestration funcion for Step 8
    Flow: guardrails -> action routing -> docs answer| tool execution | escalation
    Why it exists:
    - gives the proj one clean entry point
    - combines all earlier components into one workflow
    Step 9
    Main multi-turn message handler
    Diff from earlier version: This func now uses conversation state.
    So each new user message is interpreted in the conext of earlier turns
    Flow:
    1. update memory from mew message
    2. apply guardrails
    3. route using current turn + remembered context
    4. generate response
    5. store assistant response in history
    """
    # First update the state using the latest user message
    update_state_from_user_message(state, user_text)

    # Build a short summary of remebered state
    context_summary = build_context_summary(state)

    # check whether the message is out of scope 
    reason = out_of_scope_reason(user_text) # (step 8)
    if reason:
        state.last_intent = "refuse" # s9 -  Remember what happend last 
        response = BotResponse(
            intent="refuse",
            answer=reason,
            citations=[],
            needs_human=False,
            confidence=0.95,
            # tool_call=None
        )
        state.history.append({"role": "assistant", "text": response.answer}) # s9 - Save assistant answer in history
        return response

    # Then check if the existing guardrails say a human should take over
    if should_escalate(user_text): # (step 8)
        state.last_intent="escalate"
        #return BotResponse(
        response = BotResponse(
            intent="escalate",
            answer="I'm sorry you're dealing with that. I'm handing this over to a human support agent.",
            citations=[],
            needs_human=True,
            confidence=0.98,
            # tool_call=None
        )
        state.history.append({"role": "assisant", "text": response.answer})
        return response
    
    """
    # Step 8: decide what route to take
    action = route_action(user_text)
    route = action["route"]

    # Route 1: tool call
    if route == "tool_call":
        tool_result = execute_tool_action(tool_name=action["tool_name"], arguments=action["arguments"])
        return build_tool_response(tool_result, confidence=action["confidence"])

    # Route 2: escalate
    if route == "escalate":
        return BotResponse(
            intent="escalate",
            answer="I'm handing this over to a human support agent for further help.",
            citations=[],
            needs_human=True,
            confidence=action["confidence"],
            tool_call=None
        )

    # Route 3: docs answer
    results = retriever(user_text, top_k=5)
    return answer_from_evidence(user_text, results)
    """

    # Route based on current message
    current_intent = route_intent(user_text)
    # If current message is vague, fall back to remebered issue.
    # Eg. user says "What if it was final sale?" - route_intent may be weak alone, so memory helps
    if current_intent == "unknown" and state.issue_type: current_intent = state.issue_type
    state.last_intent = current_intent # Store the chosen intent in memory
    
    # Handle a late package case
    if current_intent == "late_package":
        if not state.order_id:
            answer=(
                "I can help check a late package."
                "Please provide your order ID so I can continue."
            )
            response = BotResponse(
                intent="late_package",
                answer=answer,
                citations=[],
                needs_human=False,
                confidence=0.85
            )
        else:
            # We can mention remebered context
            answer = (
                f"I understant your package for order {state.order_id} is late."
                "If you'd like, I can help check status or explain refund options."
            )
            response = BotResponse(
                intent="late_package",
                answer=answer,
                citations=[],
                needs_human=False,
                confidence=0.90
            )
        state.history.append({"role": "assistant", "text": response.answer})
        return response
        
    # Handle refund-related questions
    if current_intent == "refund_request" or state.wants_refund:
        # Eg of using memory to answer a follow-up
        if state.final_sale is True:
            answer = (
                "If the item was final sale, it may not be eligible for a refund"
                "under the refund policy. I can still help review the policy details."
            )
        elif state.order_id:
            answer = (
                f"I can help with a refund request for order {state.order_id}."
                "I can also check whether any policy conditions, such as final-sale status, apply."
            )
        else:
            answer = (
                "I can help with your refund request."
                "Please share your order ID so I can continue"
            )
        response = BotResponse(
            intent="refund_request",
            answer=answer,
            citations=[],
            needs_human=False,
            confidence=0.88
        )
        state.history.append({"role": "assistant", "text": response.answer})
        return response
    
    # Handle password reset
    if current_intent == "password_reset":
        if not state.email: answer="I can help start a password reset. Please provide the email address on the account."
        else: answer = f"I can help start a password reset for {state.email}."
        response = BotResponse(
            intent="password_reset",
            answer=answer,
            citations=[],
            needs_human=False,
            confidence=0.90
        )
        state.history.append({"role": "assistant", "text": response.answer})
        return response
    
    # Default fallback response
    response = BotResponse(
        intent="unknown",
        answer=(
            "I'm not fully sure yet, but here is the context I have so far: "
            f"{context_summary}"
        ),
        citations=[],
        needs_human=False,
        confidence=0.55
    )
    state.history.append({"role": "assistant", "text": response.answer})
    return response









