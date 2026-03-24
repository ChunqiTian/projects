from typing import List, Dict, Tuple # import type hints
from sklearn.metrics.pairwise import cosine_similarity # compute cosine similarity
import numpy as np # numpy helps us store and work with embedding arrays

class DenseRetriever:
    """
    A dense retriever using embeddings + cosine similarity
    Why this class exists:
    - It gives semantic retrieval
    - It can match related meaning even when exact words differ
    - It acts as the "dense" half of hybrid retrieval
    Assumption: You already have a func that can turn texts into embeddings
    """
    def __init__(self, chunks: List[Dict[str, str]], embed_fn):
        """
        Initialize the dense retriever. 
        Params: chunks - list of chunk dicts; embed_fn - a func that takes List[str] & return embeddings
        Eg: embed_fn(["hello", "world"])-> numpy array of shape(2, embedding_dim)
        What happens here:
        1. Store chunks; 2. Extract their text; 3. Embed all chunk texts once; 4. Save the embedding matrix 
        """
        self.chunks = chunks # keep the original chunks
        self.texts = [chunk["text"] for chunk in chunks] # extract the text from each chunk
        self.embed_fn = embed_fn # save the embedding function for later queries

        # Create embeddings for all chunk texts once at startup
        # We convert to numpy array so cosine similarity works cleanly. 
        self.doc_embeddings = np.array(self.embed_fn(self.texts))

        def retrieve(self, query: str, k: int=5) -> List[Tuple[Dict[str, str], float]]:
            """
            Retrieve top-k chunks using embedding similarity.
            Params: query - the user's question, k - # of res to return
            Returns: a ranked list of (chunnk, score) tuples
            Logic:
            1. Embed the query
            2. Compare query embedding to all chunk embeddings
            3. Sort descending by score
            4. Return top-k
            """
            # Embed the query as a single-item list, then take the first vector
            query_embedding = np.array(self.embed_fin([query]))
            # Compute cosine similarity between the query embedding and all doc embeddings
            scores = cosine_similarity(query_embedding, self.doc_embeddings)[0]
            # Pair each chunk with its score
            chunk_score_pairs =list(zip(self.chunks, scores))
            # Sort from highest score to lowest score
            chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
            # Return top-k chunks
            return chunk_score_pairs[:k]
        

        



