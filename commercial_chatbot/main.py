# step 8
"""
from handler import handle_message

def main():
    
    Simple CLI loop for testing the support bot.
    Why this funciton exists:
    - lets you manually test routing and tool behavior
    - useful before buildng a web UI
    

    print("Commercial Support Chatbot (step 8) - type 'q' to quit.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"q", "quit"}: break
        response = handle_message(user_text)
        print("\nBot:")
        print(response.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
"""

# Step 11 - logging
# Import tranceback so we can log full error details if sth crashes
import traceback
from logger_config import setup_logger # import logger setup
from debug_utils import log_retrieval_results, log_final_decision # import helper log funcs

# Import your existing proj funcs
from action_router import route_action
from rag.retrieve_hybrid import HybridRetriever
from rag.answer import answer_from_evidence
from schemas import BotResponse

logger = setup_logger("pipeline") # create a logger for the main pipeline

def handle_message(user_text: str) -> BotResponse:
    """
    Main chatbot pipeline. 
    Why this func exists:
    - orchestrate the full request flow
    - connect routing, retrieval, answering, and escalation
    - provide a single place to observe system behavior
    Params: user_text: raw user message from the user
    Returns: BotResponse
    """
    try:
        # Log start of a new request
        logger.info("=" * 80)
        logger.info(f"New request received: {user_text}")
        # Step 1: route the request
        route_info = route_action(user_text)
        # Extract the selected route name
        route = route_info["route"]
        # Log route decision details
        logger.info(f"Route decision: {route_info}")

        # If route says tool_call, return placeholder / mock tool behavior
        if route == "tool_call":
            logger.info(f"Executing tool path: {route_info['tool_name']}") # log tool execution intent
                        
            # Mock response eg
            response = BotResponse(
                intent="tool_result",
                answer=f"Mock tool executed: {route_info['tool_name']}",
                citations=[],
                needs_human=False,
                confidence=route_info["confidence"]
            )

            # Log final decisions summary
            log_final_decision(
                logger=logger, 
                route=response.intent,
                confidence=response.confidence,
                needs_human=response.needs_human
            )
            # Return the tool response
            return response
        
        # If route says escalate, return escalation response
        if route == "escalate": 
            logger.warning(f"Escalation path chosen: {route_info['reason']}")

            response = BotResponse(
                intent="escalate",
                answer="I'm escalating this to a human support agent.",
                citations=[],
                needs_human=True, 
                confidence=route_info["confidence"]
            )

            # Log final decision summary
            log_final_decision(
                logger=logger,
                route=response.intent, 
                confidence=response.confidence,
                needs_human=response.needs_human
            )

            # Return escalation response
            return response
        
        # Otherwise default to docs retrieval + grounded answer
        logger.info("Starting retrieval path.")

        # Run hybrid retrieval
        results = HybridRetriever.retreiver(user_text, top_k=5)

        # Log retrieved chunk summaries
        log_retrieval_results(logger, results, max_items=3)

        # Generate grounded answer from evidence
        response = answer_from_evidence(user_text, results)

        # Log final decision summary
        log_final_decision(
            logger=logger, 
            route=response.intent, 
            confidence=response.confidence, 
            needs_human=response.needs_human
        )
        # Return final response
        return response
    
    except Exception as e:
        # Log high-level error message
        logger.error(f"Unhandled exception: {str(e)}")
        # Log full traceback for deep debugging
        logger.error(traceback.format_exc())
        #Return safe fallback response
        return BotResponse(
            intent="error",
            answer="Sorry, something went wrong while processing your request.",
            citations=[],
            needs_human=True, 
            confidence=0.0
        )

# Note: step 9 - see the note in memory.py
from memory import ConversationState # import your state class
from obsolete.handler import handle_message

# Step 10 - UI Layer with Streamlit - see note in file app.py

# Step 11 - Logging & debugging visibility - see note in file logger_config.py



def main():
    """
    CLI loop fot the multi-turn support bot.
    What changed from earlier versions: We now create one ConcersationState object at the start,
        and resue it for every use turn in the same sesson.
    """
    state = ConversationState() # Create one persistent state object for this conversation session
    print("commercial chatbot - type 'q' to quit.")

    # Start chat loop
    while True:
        user_text = input("\nYouL:").strip() # Read user input
        if user_text.lower() in {"q", "quit"}: break # Exit if user wants to quit
        response = handle_message(user_text, state) # Pass both into handler
        print("\nBot:") # Print bot response
        print(response.model_dump_json(indent=2))
        # Optional: print current memory for debugging
        print("\n(DEBUG STATE)")
        print(state)

if __name__=="__main__": # Run main only when this file is executed directly.
    main()

    





