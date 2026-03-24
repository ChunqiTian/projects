from typing import Dict, Any, Optional
import re
def extract_order_id(text: str) -> Optional[str]:
    """
    Try to extract an order ID from the user's message
    Why this function exists:
    - order lookup needs a specific identifier
    - rounting alone is not enough; tools also need arguments
    """
    match = re.search(r"\b\d{5}\b", text) 
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