"""
hybrid = alpha * sparse + (1-alpha) * dense
- alphs: weight for sparse score in the final combination
alpha = 0.5: equal importance
alpha = 0.7: trust exact matching more
alpha = 0.3: trust semantic meaning more
"""

from typing import List, Dict, Tuple # import type hints for readability and safety

def min_max_normalize(results: List[Tuple[Dict[str, str], float]]) -> List[Tuple[Dict[str, str], float]]:
    """
    Normalize a list of retrieval scores into the range [0,1]
    Why this function exists:
    - Sparse and dense retrievers may produce scores on different scales
    - We need a fairer way to combine them
    - Min-max normalization rescales scores so the best is near 1 and the worst is near 0
    - eg. sparse scores: [0.02, 0.08, 0.1]; dense scores: [0.61, 0.77, 0.82]
    Params: results - list of (chunk, score) tuples
    Returns: same structure, but with normalized scores
    user query -> top matching chunks
    """
    # If the results list is empty, return is as-is
    if not results: return results
    # Extract all scores into a list
    scores = [score for _, score in results]
    # Get the minimum and maximum score values
    min_score = min(scores)
    max_score = max(scores)

    # If all scores are identical, avoid division by zero
    # In that case, assign every result a normalized score of 1
    if max_score == min_score: return [(chunk, 1.0) for chunk, _ in results]

    # Otherwise compute normalized score for each chunk
    normalized = []
    for chunk, score in results:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((chunk, norm_score))
    return normalized


class HybridRetriever:
    """
    A retriever that combines sparse and dense retrieval
    Why this class exists:
    - Sparse retrieval is good for exact term matching
    - Dense retrieval is good for semantic matching
    - Hybrid retrieval combines both strengths
    Strategy:
    1. Retrieve top res from sparse; 2. Retrieve top res from dense
    3. Normalize both score lists; 4. Merge by chunk_id
    5. Compute weighted combined score; 6. Return final top-k
    """
    def __init__(self, sparse_retriever, dense_retriever, alpha: float = 0.5):
        """
        Initialize the hybrid retriever
        Params:
        - sparse_retriever: an instance of SparseRetriever
        - dense_retriever: an instance of DenseRetriever
        - alpha: weight for sparse score in the final combination
        Final score formula: hybrid_score = alpha * sparse_score + (1-alpha)* dense_score
        Eg: alpha=0.5: equal weighting; 0.7: favor sparse; 0.3: favor dense
        """
        self.sparse_retriever = sparse_retriever # store sparse retriever
        self.dense_retriever = dense_retriever # store dense retriever
        self.alpha = alpha # store sparse weighting

    def retreiver(self, query:str, k:int=5, per_source_k:int=8) -> List[Tuple[Dict[str, str], float]]:
        """
        Retreve top-k chunks using hybrid retrival
        Params: query; 
                k: final # of results to return; 
                per_source_k: how many results to pull from each retriever before merging
        Why per_source_k exists:
        - If we only take top-k from each source, we may lose useful chunks too early
        - Taking a slightly larger candidate pool gives mrge more room to work
        Returns: a ranked list of (chunk, hybrid_score) tuple
        """
        # Retrieve candidates from sparse retrieval
        sparse_results = self.sparse_retriever.retrieve(query, k=per_source_k)
        # Retrieve candidates from dense retrieval
        dense_results = self.dense_retriever.retrieve(query, k=per_source_k)

        # Normalize each set of scores so they are more comparable
        sparse_results = min_max_normalize(sparse_results)
        dense_results = min_max_normalize(dense_results)

        # Build dicts keys by chunk_id so we can merge scores from both retrievers
        sparse_dict = {chunk["chunk_id"]: (chunk, score) for chunk, score in sparse_results}
        dense_dict = {chunk["chunk_id"]: (chunk, score) for chunk, score in dense_results}

        # Collect every chunk_id that appears in either retriever
        all_chunk_ids = set(sparse_dict.keys()) | set(dense_dict.key())
        # This list will store the final merged results
        merged_results = []

        # Loop through every candidate chunk_id from either source
        for chunk_id in all_chunk_ids:
            # Get sparse score if present, otherwise 0
            if chunk_id in sparse_dict: chunk, sparse_score = sparse_dict[chunk_id]
            else: 
                chunk = dense_dict[chunk_id][0]
                sparse_score = 0
            
            # Get dense score if present, otherwise 0
            if chunk_id in dense_dict: chunk, dense_score = dense_dict[chunk_id]
            else: dense_score = 0

            # Compute the weighted hybrid score
            hybrid_score = self.alpha * sparse_score + (1 - self.alpha) * dense_score
            # Save the merged result
            merged_results.append((chunk, hybrid_score))

        # Sort the merged res by hybrid score descending
        merged_results.sort(key=lambda x: x[1], reverse=True)

        # return the top-k final chunks
        return merged_results[:k]



