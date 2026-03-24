#schemas.py - strict output schema

from pydantic import BaseModel, Field # BaseModel for structured schema, field for metadata
from typing import List, Literal, Optional, Dict, Any

Intent = Literal["fag", "policy", "order", "tool_call", "escalate", "refuse"] # allowed intents only
# Literal is a type hint says this value is allowed to be only one of these exact fixed values
# here is means a variable of type Intent can only be items in the list nothing else

class ToolCall(BaseModel):
    """
    Structured description of a tool action the system wants to perform
    Small structured obj describing: which tool is being called; what args are sent to it
    Why this model exists:
    - keep tool requests explicit
    - easier to inspect in logs
    - easier to test than raw strings
    """
    tool_name: str=Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
        # filed name is arg; dict type; dict keys are strings; dict values are any type; 


class BotResponse(BaseModel):
    """
    Structured response returned by the chatbot backend.
    Why this exists:
    - keeps output shape consistent
    - makes UI rendering easier
    - reduces bugs from missing keys
    - supports both normal answer fields and optional debug info
    """
    intent: str = Field(..., description="High-level category for the request")
    # Field() lets you add metadata + validation rules + defaults to a model attribute
    # so Field() is how you attach constraints and docs to a field
    # This field exists, and here are the rules and docs for it

    answer: str = Field(..., description="Natural language answer shown to the user")
    citations: List[str] = Field(default_factory=list, description="Evidence chunk ids used in the answer")
    needs_human:bool = Field(..., description="Whether a human agent should take over")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in [0,1]")
    # ge = greater than or equal to, le = less than or equal to
    # tool_call: Optional[ToolCall] = Field(None, description="Tool action used, if nay")

    # Step 10 - add debugging in UI layer
    """
    In UI layer, you want to sth like: retrieved chunk ID | retrieval score | chunk text (good for debugging)
    """
    debug_chunks: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional retrieved chunks for degugging"
    )

"""
from dataclasses import dataclass, field
from typing import Literal

Intent = Literal["fag", "policy", "order", "escalate", "refuse"]

@dataclass
class BotResponse:
    intent: Intent
    answer: str
    citations: list[str] = field(default_factory=list)
    needs_human: bool = False
    confidence: float = 0.0
"""

