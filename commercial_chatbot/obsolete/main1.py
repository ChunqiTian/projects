# Chatbot



'''
Repo skeleton

1. app.py     - simple CLI (commercial line interface) or Stremlit later
2. main.py    - entrypoint for running locally (CLI)
3. config.py  - env config (API key, model name)
4. prompts.py - system prompt + templates
5. schemas.py - Pydantic schemas (strict output contract)
6. router.py  - intent routing (heuristics now, LLM later)
7. guardrails.py - refusal/escalation rules (simple now)
8. knowledge_base.py - (empty for now, add in step 2)
README.md     - project overview and setup instructions
'''

# Step 1: Implementation
'''
- Classify intent
- refuse unsafe/out-of-scope requests
- escalate to human when needed
- return a structured response (BotResponse) with intent, answer, citations, needs_human

File structure:
1. schemas.py - strict output schema
2. guardrails.py - rules for refusal / escalation
3. router.py - simple intent routing (heuristics first)
4. prompts.py - a system prompt for later LLM use
5. main.py - run a simple CLI loop today
'''
  
# for step 1
from schemas import BotResponse # structured output schema from step 1
from router import route_intent # intent router from step 1
from guardrails import should_escalate, out_of_scope_reason # safety / scope checks

# for step 2
from chunk import build_chunks # build searchable chunk records
from retrieve import retrieve_top_k, evidence_is_strong # retrieval logic
from answer import build_answer_from_chunks, format_citations # answer building

# build chunks one time when the app starts
# this avoids re-reading files on every user message
KNOWLEDGE_CHUNKS = build_chunks("knowledge_base", chunk_size=2)

"""
Handle_message function order
1. Guardrails first
- Is it out of scope?
- Is it unsafe?
- Should it be escalated?
If yes, stop early. 

2. Intent routing 
- Is this a policy question?
- Is it account help?
- Is it sth else?
This prevents unnecessary retrieval for unrelated messages. 

3. Retrieval 
Only if the message is the kind of thing your knowledge base can answer. 
then
- Retrieve top chunks
- Inspect evidence strength
- Answer only if strong enough

4. Refusal if weak evidence
This is the key anti-hallucination behavior
- You are training yourself to think: the system is allowed to say "I don't know"
That is a professional LLM engineering habit. 

"""

def handle_message(user_text: str) -> BotResponse:
    """
    Core Step-1 handler: guardrails -> routing -> canned structured respose.
    Main bot handler

    Full decision flow:
    1. Check out-of-scope refusal
    2. Check escalation
    3. Route intent
    4. If this looks like a policy question, use retrieval
    5. Answer from evidence if strong enough
    6. Refuse if evidence is weak
    """
    reason = out_of_scope_reason(user_text) # ask quardrails if the request is out of scope

    if reason: # if we got a refusal reason back
        return BotResponse(
            intent="refuse",    # mark this as a refusal type response
            answer=reason,      # explain the reason to the user
            citations=[],       # no citation because we refused
            needs_human=False,  # not a human escalation case
            confidence=0.95,    # high confidence because scope rules are deterministic
        )
    
    if should_escalate(user_text): # check if the message should go to a human
        return BotResponse(
            intent="escalate",     # escalation intent
            answer="I can connect you with a support agent. Please share your order number (if any) and a short description of the issue.",
            citations=[],          # no policy citation here
            needs_human=True,      # human needed
            confidence=0.9,        # fairly confident
        )
    
    intent = route_intent(user_text)  # classify the general type of request
    """
    # Step 1: canned responses (RAG comes Step2, tools come step 3)
    if intent == "order":
        return BotResponse(
            intent="order",
            answer="I can help with otder status. Please provide your order ID (eg., 12345).",
            citations=[],
            needs_human=False,
            confidence=0.7
        )
    
    if intent == "policy":
        return BotResponse(
            intent="policy",
            answer="I can help with our policies (shipping/returns/refunds). Which policy are you asking about?",
            citations=[],
            needs_human=False,
            confidence=0.65
        )
    
    return BotResponse(
        intent="faq",
        answer="I can help with shipping, returns, refunds, orders, and account questions. What can I help you with?",
        citations=[],
        needs_human=False,
        confidence=0.6
    )
    """



    # Step 2 - add a tiny RAG-style policy lookup layer so the bot can:
    """

    1. read policy documents
    2. split them into chunks
    3. search the most relevant chunks
    4. answer from evidence
    5. attach citations like [refund_policy:chunk_1]
    6. refuse when evidence is weak

    Pipeline
    User question -> Load policy docs -> Split docs into chunks -> Search chunks -> Take top-k relevant chunks 
    -> if evidence strong enough: answer from chunks + citations; else: refuse

    docs:
    - Knowledge_base
        - refund_policy.md
        - shipping_policy.md
        - account_policy.md
    - rag (retrieval augmented generation)
        - __init__.py
        - chunk.py
        - retrieve.py
        - answer.py
    """

    # for step 2, we use retrieval mainly for policy/faq-like questions
    if intent in ["policy_question", "faq", "shipping", "refund", "account_help"]:
        top_results = retrieve_top_k(user_text,KNOWLEDGE_CHUNKS, k=2) # fetch the best matching chunks

        if evidence_is_strong(top_results, threshold=0.2): #only answer if evidence is good enough
            answer = build_answer_from_chunks(user_text, top_results) # create answer from evidence
            citations = format_citations(top_results)

            return BotResponse(
                intent=intent,      # keep detected intent
                answer=answer,      # grounded answer
                citations=citations,# structured citation list
                needs_human=False,  # no escalation needed
                confidence=0.80,    # moderate confidence because retrieval matched
            ) 

        return BotResponse(
            intent="refuse",        # weak evidence means do not guess
            answer="I'm not confident enough to answer that from the current policy documents.",
            citations=[],           # no strong evidence to cite
            needs_human=False,
            confidence=0.85,
        )

    #fallback for other allowed but non-policy messages
    return BotResponse(
        intent=intent,
        answer="I can help with policy-related questions such as refunds, shipping, and account access.",
        citations=[],
        needs_human=False,
        confidence=0.60,
    )

# Step 3 - Better answer generation from retrieved evidence
"""
User question → Retrieve evidence → Synthesize answer from evidence (step 3)
 → Return answer + citations

You want the bot to produce a cleaner support-style answer

Goal - Turn retrieved chunks into a response that is
- concise / nature / grounded in evidence / still cited

You add
- answer synthesis from top chunks
- answer only from evidence rule
- better formatting
- clearer refusal wording

Summary: retrieval not = good final answer, you need a separate answering layer
"""











def main():
    print("commercial Chatbot (Step 1) - type 'q' to quit.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"q", "quit"}:
            break
        
        resp = handle_message(user_text)
        print("\nBot (structured JSON)")
        print(resp.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
""" # step 2 test
    tests = [
        "Can I get a refund after 20 days?",
        "How long does express shipping take?",
        "Can support agents tell me my password?",
        "Do you allow refunds on final-sale items?",
        "What is your company mission statement?",
    ]

    for question in tests:
        response = handle_message(question)
        print("=" * 80)
        print("QUESTION:", question)
        print("INTENT:", response.intent)
        print("ANSWER:", response.answer)
        print("CITATIONS:", response.citations)
        print("NEEDS HUMAN:", response.needs_human)
        print("CONFIDENCE:", response.confidence)
"""


