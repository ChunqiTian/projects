# Step 1, 2
# This file formats final answer from retrieved evidence.
"""
This file is the response formatting layer

At this stage, we are trying to be:
- grounded
- inspectable
- safe
This is the mindset for an early RAG project. 
"""

from typing import List, Tuple, Dict

def format_citations(results: List[Tuple[Dict[str, str], float]]) -> List[str]: #keep
    """
    Extract citation IDs from retrieved results.

    Example output:
    ['[refund_policy:chunk_0]', '[refund_policy:chunk_1]']

    Why this matters:
    The bot should show the user where the answer came from. 
    This increases trust and makes debugging easier.
    """
    citations = [] # list of formatted citation strings
    seen = set() # track citations already added

    for chunk, _score in results: # loop through retrieved chunk-score pairs
        # leading _ -- chunk and score are a pair, but we only care chunk for now
        citation = f'[{chunk["doc_id"]}:{chunk["chunk_id"]}]' # turn chunk_id into citation format

        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    return citations


def build_answer_from_chunks(results: List[Tuple[Dict[str,str], float]], max_chunks: int=2) -> str: # remove or replace by synthesize_answer()
    """
    # In Step 6 - remove bez same as synthesize_answer()
    Build a simple extractive answer from the retrieved chunks. 

    For this first version, we do not generate fancy summaries. 
    We simply stitch together the top chunk texts. 

    Why this is a good first step:
    - very transparent
    - easy to debug
    - reduces hallucination risk

    Later you can replace this with an LLM-based synthesis step.
    """
    """
    if not results: # if no results exist, we can't build an answer
        return "I could not find relevant policy evidence for that question."
    
    top_texts = [] # collect text from top chunks

    for chunk, score in results[:max_chunks]: # loop through retrieved results
        text = chunk.get("text", "").strip() # read and remove extra spaces
        if text: # only keep non-empty text
            top_texts.append(text)  # store the chunk text

    if not top_texts: # safety check in case retrieved chunks had no usable text
        return "I couldn't find enough reliable policy language to answer clearly."

    combined = " ".join(top_texts) # combine chunks into one readable answer body
    # join each item with an empty line in between

    citations = " ".join(format_citations(results)) # combine citations into one string

    return f"{combined}\n\nCitations: {citations}" # final answer with citations
    """

# Step 3 - Better answer generation from retrieved evidence
"""
User question → Retrieve evidence → Synthesize answer from evidence (step 3)
 → Return answer + citations

Flow:
1. check evidence quality: -- refuse or not
2. If evidence is good, synthesize answer text
3. Collect citations
4. Build final structured response


You want the bot to produce a cleaner support-style answer

Goal - Turn retrieved chunks into a response that is
- concise / nature / grounded in evidence / still cited

You add
- answer synthesis from top chunks
- answer only from evidence rule
- better formatting
- clearer refusal wording

Summary: retrieval not = good final answer, you need a separate answering layer

What Step 3 should teach you:
- Why retrieved chunks should not be dumped directly
- how to write an answer syntheizer
- how to keep answers grounded
- how to create a clean refusal when evidence is missing
- how to preserve citations

Functions
1. is_evidence_strong(...). - already in guardrails.py
2. format_citation(...) - already in answer.py
3. collect_citations(...) - merged in format_citation
4. synthesize_answer(...)
5. build_refusal_answer(...) - already in guardrails.py
6. answer_from_evidence(...)
"""

from schemas import BotResponse

def synthesize_answer(results: List[Tuple[Dict[str, str], float]], max_chunks: int=2) -> str: # same as build_answer_from_chunk()

    """
    This function turns retrieved evidence into a cleaner support-style answer. 

    Important:
    This is a simple rule-based synthesizer for learning
    Later, you could replace this with an LLM-based synthesis step

    For now, we will:
    1. take the top relevant chunks
    2. clean their text
    3. combine them into one answer

    This teaches the architecture without needing a model first.
    """

    # Create an empty list to hold cleaned text snippets.
    parts = []

    # Only use the top few retreieved results
    for chunk, _score in results[:max_chunks]:
        text = chunk.get("text", "").replace("\n", " ").strip()

        if text: parts.append(text)

    if not parts: return "I couldn't find enough policy evidence to answer that clearly."
    return " ".join(parts)


# See step 6 for the same function at the end
""" 
def answer_from_evidence(question: str, results: List[Tuple[Dict[str, str], float]]) -> BotResponse: # replace
    # Refuse if nothing was retrieved
    if not results:
        return BotResponse(
            intent="refuse", 
            answer="I'm sorry, but I couldn't find enough reliable policy info to answer that confidently.",
            citations=[],
            needs_human=False, 
            confidence=0.20
        )
    
    # Get the top score from the first result tuple
    top_score = results[0][1]

    # Refuse if evidence is weak
    if top_score < 0.30:
        return BotResponse(
            intent="refuse",
            answer="I am sorry, but I could not find enough reliable policy information to answer that confidently.",
            citations=[],
            needs_human=False,
            confidence=0.20
        )

    # Build answer and citations from retrieved results. 
    answer_text = synthesize_answer(results)
    citations = format_citations(results)

    return BotResponse(
        intent="policy_answer",
        answer=answer_text,
        citations=citations,
        needs_human=False,
        confidence=round(top_score, 2)
    )
"""

# Step 6 - Stronger refusal logic and confidence gating
"""
Main idea: instead of thinking: top score high => answer
        You want the bot to think: Answer only if evidence is relevant, consistent, and specific enough.
Thinking process:
1. Start from product behavior, not code: 
    A. strong evidence; B. related but weak evid; C. conflicting evid; D. ambiguous user q
2. Break trustworthiness into small checks
    - Is there any evidence at all?; Is the top score strong enough?...
3. Keep confidence tied to evid
    - Stronger top score -> higher confidence
    - Weak or mixed evidence -> low confidence
    - Refusal cases -> low confidence
4. Prefer safe refusal over weak answering 
    - if you are choosing between: a possibly wrong answer vs a safe refusal - pick refusal

Note: retrieved results look like: (chunk_dict, score)

Functions:
1. format_citations() - Turn retrieved chunks into a clean citation list
2. synthesize_answer() - Build the actual answer text from top retrieved chunks
    - Retrieval gives raw chunk text, users need a readable answer
    - take top 1-2 chunk texts, join them together
    - Later, if you andd an LLM synthesis layer, this func can be upgraded
3. is_query_ambiguous() - catch vague questions
    - In this case, bot should not provide full answer. 
4. has_minimum_relevant_results() - Check whether there is enough supporting evidence. 
    - It asks: How many chunks have score > 0.3? - if <2, evidence is too thin.
5. has_conflicting_signals() - Detect mixed evidence
    - eg. retrieves from return policy, final sale policy, shipping policy. All with similar score
        - Similar scores means they all close to each other, can not tell which one is the best answer
6. classify_evidence_strength() - Turn raw retrieval scores into a simple label: "none", "weak", "strong"
    - below 0.3: no reliable evidence
    - between 0.3 and 0.45: related but weak
    - above 0.45: possibly strong, but still needs support count check
7. build_refusal_response() - Provide refusing reasons
    - Weak evid: no evid, weak evid, ambiguous query, conflicting evid
8. score_confidence() - Set confidence based on evid
    - confidence = top score + small support bonus
9. answer_from_evidence() - This is the main orchestration func
    - Flow: 1. ambiguity chech; 2 evid classify 3. refusal is none; 4 ref if weak
        5. ref if conflicing; 6 answer only if strong

"""

def is_query_ambiguous(question: str) -> bool:
    """
    Detect whether the user's question is too vague to answer confidently. 
    It do not try to fully understand language, but simply catches common vague support questions. 
    Eg. Can I return it? / Does this apply? / Can you do that?
    Why ambiguity matters? The user's question may be missing product, order region...
    Return: True - if the q looks vague; False - otherwise
    """
    q = question.lower().strip() # normalize the q for easier checking
    ambiguous_patterns = { "it", "this", "that", "they", "them", "these", "those"}  
        # vague pronoun-like words that may hide missing context
    short_vague_questions = {
        "can i return it",
        "does this apply",
        "what about international",
        "can you do that",
        "is that allowed",
        "does it work",
    }  # common vague support-style questions
    if q in short_vague_questions: return True # exact match
    words = q.split() # split q into words for simple analysis

    # If the q is very short and contains vague references, it may not provide enough context for a reliable answer
    if len(words) <= 5 and any(word in ambiguous_patterns for word in words): return True
    return False # otherwise, treat the q as specific enough

def has_minimum_relevant_results(
        results: List[Tuple[Dict[str, str], float]],
        min_score: float=0.3,
        min_count: int=2
) -> bool:
    """
    Check whether we have enough reasonably relevant chunks
    Why this matters:
    - one decent chunk may not be enough to answer a policy q
    - multiple relevant chunks usually increase trustworthiness
    - this helps distinguish "some signal" from "enough signal"
    Params: - min_score: a chunk must score at least this high to count as relevant
        - min_count: minimum # of such chunks required
    Returns:
        - True: if enough chunks meet the threshold
        - False otherwise
    """
    strong_count = 0 # count how many retrieved chunks are relevant enough
    for chunk, score in results: # check every retrieved result
        if score >= min_score: # count only chunks that meet the relevant floor
            strong_count += 1
    return strong_count >= min_count # True only if enough relevant chunks exist

def has_conflicting_signals(
        results: List[Tuple[Dict[str, str], float]],
        strong_band: float=0.3
) -> bool:
    """
    Detect whether retrieval seems mixed or conflicting
    Since your bot is not yet doing full contradiction detection, we use a practical proxy:
    Conflict suspicion increases when:
    - there are several moderately relevant chunks
    - they come from different docs
    - no single doc clearly dominates
    Why this matters:
    - if diff policy docs are surfacing with similar strength, the bot should be more cautious
    - this avoids confident answers from mixed evidence
    Returns: T if signals look mixed/conflicting; F otherwise
    """
    doc_ids=[] # collect the doc ids for reasonably relevant chunks
    for chunk, score in results: # inspect each retrieved result
        if score >= strong_band: # only consider chunks that are somewhat relevant
            doc_id = chunk.get("doc_id", "") # read the source doc id
            if doc_id: #keep only non-empth doc ids
                doc_ids.append(doc_id)
    unique_docs = set(doc_ids) # distinct docs among relevant results

    # If several relevant chunks come from multiple docs, but none is clearly dominant, treat as potentially mixed. 
    if len(doc_ids) >= 3 and len(unique_docs) >= 2: return True
    return False

def classify_evidence_strength(
        results: List[Tuple[Dict[str, str], float]],
        answer_threshold: float=0.45,
        weak_threshold: float=0.3
) -> str:
    """
    Classify retrieved evidence into one of three categories:
    1. strong -> safe enough to answer
    2. weak -> related but not enough
    3. none -> almost no useful evidence
    Why this function exists:
    - it creates a clean decision layer before final answer generation
    - it keeps threshold logic in one place
    - it makes the product behavior easier to tune later
    Logic:
    - no res and top score below weak_threshold -> none
    - top score below answer_threshold -> weak
    - otherwise check whether enough results support answering
    Return: one of [strong, weak, none]
    """
    if not results: return "none" # if retrieval returned nothing
    top_score = results[0][1] # best retrieval score from the first result
    if top_score < weak_threshold: return "none"
    if top_score < answer_threshold: return "weak"
    
    # Even if top score is good, we still prefer multiple supporting chunks
    if not has_minimum_relevant_results(results, min_score=weak_threshold, min_count=2):
        return "weak"
    return "strong"

def build_refusal_response(reason: str, results:List[Tuple[Dict[str, str], float]]) -> BotResponse:
    """
    Build a consistent refusal response based on the reason
    Why this exists:
    - refusal wording should be standardized
    - keeps answer_from_evidence() cleaner
    - makes it easy to tune refusal messages later
    Supported reasons: "none", "weak", "ambiguous", "conflict"
    Returns: a BotResponse with refusal-style answer and low confidence
    """
    related_citations = format_citations(results[:2]) #optional citations from the top few results
    if reason == "none":
        message = (
            "I couldn’t find enough reliable policy evidence to answer that confidently. "
            "Please contact support for confirmation."
        )
        confidence = 0.20

    elif reason == "weak":
        message = (
            "I found related policy information, but not enough to answer confidently. "
            "Please contact support for confirmation."
        )
        confidence = 0.30

    elif reason == "ambiguous":
        message = (
            "Your question may need more specific details before I can answer it reliably. "
            "I’m not confident the current policy documents answer that clearly."
        )
        confidence = 0.25

    elif reason == "conflict":
        message = (
            "I found policy information that may be mixed or not fully consistent, "
            "so I’m not confident answering this clearly. Please contact support for confirmation."
        )
        confidence = 0.25

    else:
        message = (
            "I’m not confident the current policy documents answer that clearly. "
            "Please contact support for confirmation."
        )
        confidence = 0.20

    return BotResponse(
        intent="refuse", # this is a refusal case
        answer = message, # safe refusal message
        citations=related_citations, # optional related citations if you wnat to show partial evidence
        needs_human=False, # you can switch this to True if your product wants escalation flags
        confidence=confidence # keep conf low in refusal cases
    )

def score_confidence(results: List[Tuple[Dict[str, str], float]], evidence_label: str) -> float:
    """
    Convert retrieval evidence into a final confidence score. 
    Why this function exists:
    - confidence should not be arbitrary
    - it should reflect evidence strength
    - this keeps the scoring logic explicit and easy to tune
    Simple strategy:
    - start from the top retrieval score
    - give a small bonus if several good chunks support the answer
    - cap the value between 0 and 0.95
    Returns: conficence as a float
    """
    if not results: return 0.2 # no retrieval evidence
    top_score = results[0][1] # strongest retrieval score
    relevant_count = sum(1 for chunk, score in results if score >= 0.3) # num of moderately relevant chunks
    if evidence_label == "strong":
        confidence = top_score + min(0.1, 0.02 * relevant_count) # reward multi-chunk support
        return round(min(confidence, 0.95), 2) # cap at 0.95
    if evidence_label == "weak": return 0.3 # weak evid should never look confident
    return 0.2 # default low confidence for refusal-like cases

from logger_config import setup_logger # import the shared logger setup
logger = setup_logger("answer") # Create or reuse the logger for this file/module

def answer_from_evidence(question: str, results: List[Tuple[Dict[str, str], float]]) -> BotResponse:
    """
    Main step 6 answer decision function. 
    This is the core logic for trust-aware answering. 
    Decision order:
    1. detect ambiguity in the user question
    2. classify evidence strength
    3. detect mixed/conflicting signals
    4. refuse safely when needed
    5. answer only when evidence is strong enough
    Why this funciton is important:
    - it combines retrieval quality + product safety behavior
    - it is the bridge between "found chunks" and "final support response"
    - it teaches the most important real-world habit: uncertainty handling is part of the product
    Returns: BotResponse

    Edit step 11
    This func is your answering layer after retrieval
    Why logging helps here: - This stage is where many user-visible failures happen
    - With the logs, you can distinguish: no retrieval happened;happened but weak; answer succeeded
    - Without logs, all you see is "the bot refused".
    Parameters: question; results: retrieved chunks with scores
    
    # step 1: if the question is too vague, avoid pretending the answer is clear
    if is_query_ambiguous(question): return build_refusal_response("ambiguous", results)
    # step 2: classify how strong the evidence is overall
    evidence_label = classify_evidence_strength(results)
    # step 3: if there is basically no useful evidence, refuse
    if evidence_label == "none": return build_refusal_response("none", results)
    # step 4: if evidence is only partically related, refuse safely
    if evidence_label == "weak": return build_refusal_response("weak", results)
    # step 5: even strong evidence should be checked for mixed signals
    if has_conflicting_signals(results): return build_refusal_response("conflict", results)
    # step 6: only now do we h=build the final grounded answer
    answer_text = build_answer_from_chunks(results) # create answer from retrieved chunk texts
    citations = format_citations(results) # collect citation ids
    confidence = score_confidence(results, evidence_label="strong") # compute final confidence

    return BotResponse(
        intent="policy_answer", # this is a grounded policy answer
        answer=answer_text, # final answer text from evidenc
        citations=citations, # source chunk ids
        needs_human=False, # no escalation needed in the strong-evidence
        confidence=confidence # confidence tied to retrieval quality
    )
    """
    # Log the question entering this answering stage. 
    logger.info(f"Answer generation started for question: {question}")

    # If retrieval returned nothing, refuse safely. 
    if not results: 
        # Log the reason for refusal at warning level, bcz this is not a crash, but it is an important issue.
        logger.warning("Refusal triggered: no retrieval results found.")
        
        # Return a refusal response. 
        return BotResponse(
            intent="refuse", 
            answer="I'm sorry, but I couldn't find enough reliable policy information to answer that confidently.",
            citations=[],
            needs_human=False,
            confidence=0.20
        )

    # Get the score of the top-ranked retrieved chunk
    top_score = results[0][1]

    # Log the top score for debugging evidence strength
    logger.info(f"Top retrieval score: {top_score:.4f}")

    # If the best evidence is too weak, refuse. 
    if top_score < 0.30:
        # Log why refusal happened
        logger.warning(f"Refusal triggered: weak evidence (top_score={top_score:.4f})")

        # Return refusal response
        return BotResponse(
            intent="refuse",
            answer="I'm sorry, but I couldn't find enough reliable policy information to answer that confidently.",
            citations=[],
            needs_human=False,
            confidence=0.20
        )
    
    # Build answer text from retrieved chunks
    answer_text = synthesize_answer(results)

    # Build citations from retrieved chunks
    citations = format_citations(results)

    # Log success of grounded answer generation
    logger.info(f"Policy answer generated with {len(citations)} citation(s).")

    # Return the final grounded response
    return BotResponse(
        intent="policy_answer",
        answer=answer_text,
        citations=citations, 
        needs_human=False,
        confidence=round(top_score, 2)
    )









