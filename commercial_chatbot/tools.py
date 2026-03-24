from typing import Dict

def lookup_order(order_id: str) -> Dict[str, str]:
    """
    Mock tool: look up an order by order ID
    Why this funciton exists: 
    - simulates a backend order lookup system
    - lets you test workflow logic before connecting a real API
    - returns structured data instead of free-form text
    """

    fake_orders = {
        "12345": {"status": "shipped", "estimated_delivery": "2026-03-21"},
        "88888": {"status": "processing", "estimated_delivery": "2026-03-24"},
        "99999": {"status": "delivered", "estimated_delivery": "2026-03-16"},}
    order = fake_orders.get(order_id)
    if not order:
        return {
            "ok": "false",
            "tool_name": "lookup_order",
            "order_id": order_id, 
            :"message": "Order not found."
        }
    return {
        "ok": "true", 
        "tool_name": "lookup_order", 
        "order_id": order_id,
        "status": order["status"],
        "estimated_delivery": order["estimated_delivery"]
    }

def create_ticket(issue_type: str, message: str) -> Dict[str, str]:
    """
    Mock tool: create a support ticket
    Why this function exists:
    - simulates ticket creation without needing a real CRM/helpdesk
    - useful for complaints, billing issues, damaged items, etc
    """
    fake_ticket_id = "TICK-1001"
    return {
        "ok": "true", 
        "tool_name": "create_ticket", 
        "ticket_id": fake_ticket_id,
        "issue_type": issue_type,
        "message": message
    }

def start_password_reset(email: str) -> Dict[str, str]:
    """
    Mock tool: start a password reset process
    Why this function exists:
    - simulates an account recovery action
    - useful for workflow routing practice
    """
    return {
        "ok": "true",
        "tool_name": "start_password_reset", 
        "email": email, 
        "message": f"Password reset link sent to {email}"
    }





