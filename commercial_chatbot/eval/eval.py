# Step 7: Evaluation
"""
Evaluate the performance
What we measure?
1. Retrieval hit rate: Did retrieval bring back the expected doc?
    - eg. expected doc: refund_policy; if retrieved top results include refund_policy, that's a hit
2. Citation rate: Did the answer include citations when it should?
3. Refusal correctness: Did the bot refuse when it was supposed to do?
4. Groundedness: Does the answer stay close to retrieved evidence, instead of inventing unsupported claims?
Thinking process:
1. Define what one eval case looks like - we need a consistent structure so the runner can load every case automatically
    - so we create: question; expected_intent; expected_doc; should_refuse
2. Decide what parts of the chatbot to test
    - layer 1 - Retrieval - Test whether expected doc appears in retrieved results
    - layer 2 - Final answer - test: refusal behavior; citations; groundedness
3. Create small metric functions
    - load_eval_cases()
    - retrieval_hit()
    - citation_present()
    - refusal_correct()
    - groundedness_score()
    - evaluate_case()
    - run_eval()
4. Save results
    - Save structured output like: {"summary":{...}, "cases":[...]}
    - This helps you: compare runs; show portfolio evidence; inspect failures later

functions:
1. load_eval_cases()  - load your test dataset from jsonl file
2. normalize_text(text:str) - Turn text into simple tokens: lowercase, remove punctuation, split into words
3. retrieval_hit(results, expected_doc) - check retrieved quality
    - checks whether expected doc appears anywhere in retrieved results
4. citation_present(response) - answer is trustworthy or not?
    - checks whether the final answer contains at least one citation
5. refusal_correct(response, should_refuse) - check if refused correctly
    - If case expects refusal: predicted intent: "refuse"
    - If case expects answer: predicted intent should not be "refuse"
6. groundedness_score(answer_text, results) - computes a rough overlap score between answer words and evidence words
    - bec we want signal of whether the answer is staying close to the retrieved evidence
    - Formula: A = answer tokens; E = evidence tokens; \text{groundedness} = \frac{\text{# of answer tokens also found in evidence}}{\text{# of answer tokens}}
7. evaluate_case(case, top_k=5) - Run one eval case through the whole pipeline
    - 1. Take the q | 2. Retrieve evid | 3. Generate response | 4. Score retrieval|
      5. Score citation presence | 6. Score refusal correctness | 7. Score groundedness | 8. Return a structured result
8. compute_summary(case_results) - Turns per-case results into overall metrics.
9. save_results(path, summary, case_results) - whites eval results to disk
    - It lets you inspect failure cases later; compare versions; cite concrete metrics in README
10. run_eval() - This is the main orchestrator
    Flow: load data| eval every case| compute metrics| save results| print summary


"""

# eval_cases.jsonl for starter dataset

# This file runs evaluation cases against the chatbot pipeline
# The purpose is to measure system quality instead of guessing. 

import json # for reading JSONL and writing JSON results
import re # for simple text cleanup / token matching
from pathlib import Path # cleaner file path handling
from typing import List, Dict, Any, Tuple 

# Import your retrieval function 
from rag.retrieve_hybrid import hybrid_retrieve
# Import your answer generator that turns evidence into a BotResponse
from rag.answer import answer_from_evidence
# Import the response schema for strong typing and clarity
from schemas import BotResponse

def load_eval_cases(path: str) -> List[Dict[str, Any]]:
    """
    Expected line format eg:
    {
        "question": "Can I return a final-sale item?",
        "expected_intent": "policy_answer", 
        "expected_doc": "refund_policy", 
        "should_refuse": false
    }
    returns: A list of dictionaries, one per eval case
    """
    cases = [] # this will store every parsed evaluation case
    # Open the file in text mode with UTF-8 encoding
    with open(path, "r", encoding="utf-8") as f:
        # Read file line by line so each JSON object is processed separately
        for line in f: 
            line = line.strip() # remove witespace/newline around the line
            # Skip empty lines so blank rows do not crash the loader
            if not line: continue
            # Convert the JSON string into a Python dictionary
            cases.append(json.loads(line))
    # Return the full list of cases
    return cases

def normalize_text(text: str) -> List[str]:
    """
    Convert text into a simple normalized token list
    Why this exists: we want a basic way to compare answer text with evidence text for groundedness checks
    Steps: lowercase; keep only letter/#s/spaces; split into words
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
    tokens = text.split() # split into whitespace-separated words
    return tokens # return normallized tokens

def retrieval_hit(results: List[Tuple[Dict[str, str], float]], expected_doc:str) -> bool:
    """
    Check whether retrieval returned the expected doc. 
    Inputs: results - a list like [(chunk_dict, score), [chunk_dict, score]...]
            expected_doc - eg. refund_policy
    returns: True if any retrieved chunk came from expected_doc, else False
    """
    # If we do not expect a doc (eg refusal cases), retrieval hit is not applicable
    if expected_doc is None: return True
    # Loop through each retrieved (chunk, score) pair
    for chunk, _score in results:
        # Compare the chunk's doc_id against the expected doc
        if chunk.get("doc_id") == expected_doc: return True # success as soon as one match is found
    # No match found after checking all results
    return False

def citation_present(response: BotResponse) -> bool:
    """
    Check whether the final response contains at least one citation. 
    Why this matters: A grounded policy answer should usually cite evidence.
    """
    # getattr is safe even if the object structure changes slightly
    citations = getattr(response, "citations", [])
    # Return Ture if citations list has at least one item
    return len(citations) > 0

def refusal_correct(response: BotResponse, should_refuse:bool) -> bool:
    """
    Check whether the bot's refusal behavior matches expectation. 
    Logic - if case should_refuse=True, response intent should be refuse
          - if case should_refuse=False, response intent should NOT be refuse
    Returns: T if refusal behavior is correct, else F.
    """
    # Extract the predicted intent from the BotResponse
    predicted_intent = response.intent
    # If the eval case expected refusal, verify that the model refused
    if should_refuse: return predicted_intent == "refuse"
    # Otherwise, verify that the model did not refuse
    return predicted_intent != "refuse"

def groundedness_score(answer_text: str, results: List[Tuple[Dict[str, str], float]]) -> float:
    """
    Compute a very simple groundedness score using token overlap.
    This is NOT perfect. But it is a strong portfolio-friendly first version. 
    Idea:
    - Gather all retrieved evidence text
    - Normalize answer and evidence into tokens
    - Measure what fraction of answer tokens appear in evidence tokens.
    Interpretation:
    - higher score -> answer stays closer to evidence
    - lower score -> answer may be adding unsupported content
    Returns: A float between 0 and 1
    """
    # If there are no retrieval results, groundedness is zero
    if not results: return 0
    # Join all evidence chunk text into one big evidence string
    evidence_text = " ".join(chunk.get("text", "") for chunk, _score in results)
    # Normalize answer into tokens
    answer_tokens= normalize_text(answer_text)
    # Normalize evidence ino a set for fast membership checking
    evidence_tokens = set(normalize_text(evidence_text))
    # If answer became empty after normalization, return zero
    if not answer_tokens: return 0.0
    # Count how many answer tokens also appear in evidence tokens
    overlap_count = sum(1 for token in answer_tokens if token in evidence_tokens)
    # Divide by total answer token count to get a rough groundedness ratio
    return overlap_count / len(answer_tokens)

def evaluate_case(case: Dict[str, Any], top_k: int=5) -> Dict[str, Any]:
    """
    Evaluate a single test case end-to-end.
    Pipeline:
    1. retrieve evidence
    2. generate final response from evidence
    3. score retrieval / citation / refusal / groundedness
    Returns: A dictionary with detailed case-level results.
    """
    # Extract question from the case dictionary
    question = case["question"]
    # Extract expected doc, if any
    expected_doc = case.get("expected_doc")
    # Extract expected refusal behavior
    should_refuse = case.get("should_refuse", False)
    # Retrieve top=-k evidence chunnks using your retrieval system
    results = hybrid_retrieve(question, top_k=top_k)
    # Turn retrieved evvidence into a final structtured response
    response = answer_from_evidence(question, results)
    # Compute whether retrieval found the epeccted supporting do
    hit = retrieval_hit(results, expected_doc)
    # Compute whether citations were included
    has_citation = citation_present(response)
    # compute wheher refusal behavior was correct
    refusal_ok = refusal_correct(response, should_refuse)
    # Compute a rough groundedness score from answer vs evidence overlap
    grounded_score = groundedness_score(response.answer, results)
    # Extract retrieved doc_ids for debugging and inspection
    retrieved_docs = [chunk.get["doc_id"] for chunk, _score in results]
    # Return a detailed per-case result ditinary
    return {
        "question": question,
        "expected_intent": case.get("expected_intent"),
        "expected_doc": expected_doc,
        "should_refuse": should_refuse,
        "predicted_intent": response.intent,
        "answer": response.answer,
        "citations": response.citations,
        "retrieved_docs": retrieved_docs,
        "retrieval_hit": hit,
        "citation_present": has_citation,
        "refusal_correct": refusal_ok,
        "groundedness_scores": round(grounded_score,3),
        "confidence": response.confidence,
    }

def compute_summary(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate case-level results into overall metrics
    Metrics included:
    - retrieval hit rate
    - citation rate
    - refusal correctness
    - avereage groundedness
    Returns: A summary dicionary for the whole evaluation run. 
    """
    # Number of evaluation cases processed
    total = len(case_results)
    # Guard against division by zero if dataset is empy
    if total == 0:
        return {
            "num_cases": 0,
            "retrieval_hit_rate": 0.0,
            "citation_rate": 0.0,
            "refusal_correct_rate": 0.0,
            "avg_groundedness": 0.0,
        }
    # Sum all retrieval_hit booleans (True counts as 1)
    retrieval_hit = sum(r["retrieval_hit"] for r in case_results)
    # Sum all citation_present booleans
    citation_hits = sum(r["refusal_correct"] for r in case_results)
    # Average the goundedness scores
    avg_groundedness = sum(r["groundednss_score"] for r in case_results) / total

def save_results(path: str, summary: Dict[str, Any], case_results List[Dict[str, Any]]) -> None:
    """
    Save evaluaion summary and detailed per-case results to JSON
    Why save results?
    - Compare runs across versions
    - inspect failures later
    - include evidence in your portfolio / README
    """
    # Build one final output object ith top-level summary and detailed cases
    output = {
        "summary": summary,
        "cases": case_results,
    }
    # Ensure parent folder exists before writing
    Path(path).parent.mkdir(parent=True, exist_ok=True)
    # Write pretty-printed JSON to disk
    with open(path, "w", encoding="utf-8") as f: 
        json.dump(output, f, indent=2, ensure_ascii=False)

def run_eval(
    cases_path: str = "eval/eval_cases.jsonl",
    results_path: str = "eval/eval_results.json",
    top_k: int=5,
) -> Dict[str, Any]:
    """
    Main evaluation entry point.
    Step:
    1. load eval cases
    2. evaluate each case
    3. compute summary metrics
    4. save results
    5. return summary + detailed cases
    This function acts like the orchestrator of the whole eval harness.
    """
    # Load dataset of evaluation examples
    cases = load_eval_cases(cases_path)
    # Store per-case outputs here
    case_results = []
    # Loop through each evaluation case and score it
    for case in cases:
        result = evaluate_case(case, top_k=top_k)
        case_results.append(result)
    # Compute overall metrics across all evaluated cases
    summary = compute_summary(case_results)
    # Save both summary and detailed case results
    save_results(results_path, summary, case_results)
    # Also print summary. to termianl for quick inspection
    print("\n=== Evaluation Summary ===")
    for key, value in summary.items(): print(f"{key}: {value}")
    # Return everything in case another script wants to use it
    return {
        "summary": summary,
        "cases": case_results,
    }

if __name__ == "__main__":
    """
    This lets you run the file directly from terminal:
    python eval/eval_runner.py
    """
    run_eval()








