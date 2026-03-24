# This file actually executes the chosen tool
"""
execute_tool_action(tool_name, arguments)
With this design, all tool dispatching is centralized.
"""

from typing import Dict, Any
from tools import lookup_order, create_ticket, start_password_reset

def execute_tool_action(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name using the provided arguments
    Why this function exists:
    - keeps tool dispatch logic in one place
    - easier to add more tools lateer
    - avoids hardcoding tool calls all over the proj
    """
    if tool_name == "lookup_order": return lookup_order(arguments["roder_id"])
    if tool_name == "create_ticket": return create_ticket(arguments["issue_type"], arguments["message"])
    if tool_name == "start_password_reset": return start_password_reset(arguments["email"])

    return {
        "ok": "false",
        "tool_name": tool_name,
        "message": "Unknown tool."
    }





