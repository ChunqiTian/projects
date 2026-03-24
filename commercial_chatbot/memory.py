# Step 9 - Conversation memory / multi-turn behavior
"""
Step 9 is where your bot starts feeling like a real support assistant instead of a oneshot Q&A system. 
A real support chat flow may like:
1. User gives the prob
2. Gives missing details later
3. Changes the goal
4. Asks a follow-up vased on earlier context
The bot needs to remember things like: the current issue | order ID| email|
    whether the user wants refund/tracking/password reset | whether the item was final sale
We want to add: conversation state; slot memory; multi-turn understanding; simple context mgmt
Design:
- before: user_text -> route -> retrieve/tool/escalate -> response
- after: user_text + conversation_state -> interpret current turn -> update memory -> choose action -> response
Thinking process:
1. create a conversation state object - stores memory
2. Add helper functions to extract slots - eg. detect order ID|email|refund intent|final sale mention
3. Update state after each user turn 
4. Resolve missing info from memory - eg. "Can I get refund?" -> bot checks memory & sees order_id="123" (knows the Q is about that order)
5. Keep only a small recent history  
Files
- memory.py (hold state + update logic)
- updated main.py (keeps conversation alive across turns)
- updated handler.py (uses memory when deciding what to do)
Function:
1. add_turn() - stores recent messages in history and trims old ones
2. extract_order_id() - finds order #s (imported)
3. extract_email() - finds email from user messages (imported)
4. detect_issue_type() - maps free text into a simpler issue label 
    - eg. "My package is late" => "late_package"
5. detet_refund_request() - checks whether the user asking for a refund
6. detect_final_sale() - may no refund in such case
7. update_state_from_user_messsage() - update memory in one place
8. build_context_summary() - creates a short readable summary of current memory
"""

from dataclasses import dataclass, field # use dataclass to make  clean state container
from typing import Optional, List, Dict # These typing tools help doc what kinds of values are stored
import re # re is for regular expressions, useful for finding order IDs and emails
from helper import extract_email, extract_order_id

@dataclass
class ConversationState: 
    """
    Stores conversation memory across multiple turns
    Why this class exists:
    Earlier versions of the bot treated every message independent.
    This class lets the bot carry important details forward, eg:
    - order ID | email | issue type | whether user wants a refund | whether final sale was mentioned | recent turn history
    Think of this as the bot's short-term working memory.
    """
    # The most recent detected high-level intent.
    last_intent: Optional[str] = None # last_intent - variable name | tyoe: str or None
    # Store order ID if user provided one, like 12345
    order_id: Optional[str] = None
    # Store email if user provided one
    email: Optional[str] = None
    # Store current issue category, eg. "last_package" or "password_reset"
    issue_type: Optional[str] = None
    # Whether the user is currently asking about refunds
    wants_refund: bool=False
    # Whether the user mentioned final sale 
    final_sale: Optional[bool] = None
    # Small recent conversation histor for context
    history: List[Dict[str, str]] = field(default_factory=list)
    # Store a short summary of the last tool/action result if useful
    last_action: Optional[str] = None

def add_turn(state: ConversationState, role: str, text: str, max_turns: int=6) -> None:
    """
    Add a message to conversation history and keep only the most recent turns
    Why this function exists: Real conversations can get long
    We usually don't want infinite history bz:
    - it becomes noisy | harder to debug | increases prompt/context size later
    Params:
    - state: the current conversation memory object
    - role: 'user' or 'assistant'
    - text: the actual message content
    - max_turns: how many recent message to keep
    """
    state.history.append({"role": role, "text": text}) # Append the new turn as a small dictionary
        # go into the history list inside state, add one more dict item to the list
    state.history = state.history[-max_turns:] # keep only the latest max_truns items

def detect_issue_type(text: str) -> Optional[str]:
    """
    Infer the issue category from keywords.
    Why this function exists: Multi-turn bots need a compact representation of the user's problem
    This lets future turns reuse the current issue. 
    Eg My package is late -> late_package
    """
    lowered = text.lower()
    if any(word in lowered for word in ["late" "delayed", "package is late", "where is my order", "not arrived"]):
        return "late_package"
    if "refund" in lowered or "return" in lowered: 
        return "refund_request"
    if "password" in lowered or "broken" in lowered: 
        return "damaged_item"
    return None

def detect_refund_interest(text: str) -> bool:
    """
    If the user request a refund?
    Why this func is separated from detect_issue_type?
    - The user may have late_package issue but ask for refund
    So it's an important state flag.
    """
    lowered = text.lower()
    return any(word in lowered for word in ["refund", "money back", "return it", "return this"])

def detect_final_sale(text: str) -> Optional[bool]:
    """
    Detect whether final sale was mentioned. 
    Return policy may be diff for final sale items
    """
    lowered = text.lower()
    if "final sale" in lowered: # user eplicitly says final sale
        # Handle phrases like "not final sale first"
        if "not final sale" in lowered or "wasn't final sale" in lowered or "was not final sale" in lowered:
            return False
        return True
    return None # Nothing useful found

def update_state_from_user_message(state: ConversationState, user_text: str) -> ConversationState:
    """
    Update conversation memory using the latest user message
    This is the main state-update function
    What is does:
    1. store the user turn in history
    2. extract structured slots like order_id and email
    3. detect issue type
    4. detect refund intent
    5. detect final-sale condition
    Why it matters: It centralizes memory updates in one place. That makes the bot easier to maintain and debug.
    """
    add_turn(state, "user", user_text) # Save the user message in history first
    
    order_id = extract_order_id(user_text) # Try to extract order ID
    if order_id: state.order_id = order_id
    
    email = extract_email(user_text) # Try to extract email
    if email: state.email = email
    
    issue_type = detect_issue_type(user_text) # Try to detect issue type
    if issue_type: state.issue_type = issue_type
    
    if detect_refund_interest(user_text): state.wants_refund=True # Update refund interest if needed
    
    final_sale = detect_final_sale(user_text) # Update final-sale state if user mentioned it
    if final_sale is not None: state.final_sale = final_sale
    
    return state # Return the updated state

def build_context_summary(state: ConversationState) -> str:
    """
    Create a compact text summary of remebered conversation state
    Why this function exists: 
    Sometimes your handler, retriever, or LLM prompt needs a short summary of the current known facts
    Example output:
    'Known context: issue=late_package; order_id=12345; wants_refund=True'
    """
    parts = [] # empty list of summary parts
    # Add whichever fields are currently known. 
    if state.issue_type: parts.append(f"issue={state.issue_type}") 
    if state.order_id: parts.append(f"order_id={state.order_id}")
    if state.email: parts.append(f"email={state.email}")
    parts.append(f"wants_refund={state.wants_refund}")
    if state.final_sale is not None: parts.append(f"final_sale={state.final_sale}")
    if state.last_intent: parts.append(f"last_intent={state.last_intent}")
    if not parts: return "known context: none" # If nohing is known et, return a simple fallback
    return "Know context: " + "; ".join(parts) # Join pieces into one readable summary













