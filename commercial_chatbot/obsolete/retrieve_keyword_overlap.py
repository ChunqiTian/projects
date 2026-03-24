"""
We will use a simple lexical overlap scorer to find the most relevant chunks for a user question.
That means:
- tokenize the question
- tokenize each chunk
- count word overlap
- rank by score
Note: this is not as advanced as embeddings, but it's a good starting point to understand the retrieval process.

Function summary:
- tokenize(text): turn raw text into normalized word tokens
- score_query_to_chunk(query, chunk_text): measure how related a chunk is to the question 
- retrieve_top_k(query, chunks, k): rank all chuncks and keep the best few
- evidence_is_strong(top_results, threshold): decide whether the bot should answer or refuse based on the best scores
    - answer only if the max(retrieval score) >= threshold, otherwise refuse

"""

import re # used for text cleaning and token extraction
from typing import List, Dict, Tuple # type hints

def tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase word tokens. 
    
    Eg:
    "Refund within 30 days." -> ['refund', 'within', '30', 'days']

    Why this matters:
    Retrieval needs a normalized representation. 
    If we don't normalize text, then:
    - Refund and refund may look different
    - punctuation can interfere with matching
    """
    return re.findall(r"\b\w+\b", text.lower()) # find all word tokens, ignore punctuation, convert to lowercase
    # r"...": defines a raw string
    # \b: word boundary, ensures we match whole words
    # \w+: matches sequences of word characters (letters, digits, underscore)


def score_query_to_chunk(query: str, chunk_text: str) -> float:
    """
    Compute a simple overlap score between user query and chunk text. 

    Strategy:
    1. Tokenize query
    2. Tokenize chunk
    3. Count overlap
    4. Normalize by query length

    Why normalize?
    Longer queries may unfairly produce larger raw overlap counts.

    Score formula:
    score = (number of overlapping tokens) / (total number of query tokens)

    Range: 
    0.0 - 1.0, where 1.0 means perfect overlap.
    """

    query_tokens = set(tokenize(query))   # unique words from the query
    chunk_tokens = set(tokenize(chunk_text)) # unique words from the chunk

    if not query_tokens:  # guard against empty user input
        return 0.0
    
    overlap = query_tokens.intersection(chunk_tokens)   # words appearing in both sets
    score = len(overlap) / len(query_tokens)  # normalize overlap ratio

    return score


def retrieve_top_k(query: str, chunks:List[Dict[str, str]], k: int = 3) -> List[Tuple[Dict[str, str], float]]:
    """
    Rank all chunks by relevance and return the top-k.

    Parameters:
    - query: the user's question
    - chunks: list of searchable chunk records
    - k: number of top chunks to return

    output:
    [
        (chunk_dict, score),
        (chunk_dict, score),
        ...
    ]

    Why this function matters:
    This is the core "retrieval" step of RAG.
    It decides what evidence the answering layer is allowed to use. 
    """

    scored = [] # will store pairs of (chunk, score)

    for chunk in chunks: #check every chunk
        score = score_query_to_chunk(query, chunk["text"]) # compute similarity score
        scored.append((chunk, score)) # store results

    scored.sort(key=lambda x: x[1], reverse=True) # sort by score, highest first
    # x[0] is the chunk; 
    # x[1] is the score - sort by the second element of each tuple
    # reverse=True - sort from highest to lowest score

    return scored[:k] # return only the top-k results


def evidence_is_strong(top_results: List[Tuple[Dict[str, str], float]], threshold: float = 0.2) -> bool:
    """
    Decide whether retrieved evidence is strong enough to answer. 

    Logic:
    - If there are no results, return False
    - If the best score is below the threshold, return False
    - Otherwise, return True

    Why this matters:
    This is your anti-hallucination gate.
    The bot should not answer confidently when retrieval is weak.
    """
    if not top_results: # no retrieved chunks
        return False
    
    best_score = top_results[0][1] #score of the best chunk
    # [0] - first tuple
    # [1] - score in that tuple
    return best_score >= threshold # answer only if best score meets threshold

