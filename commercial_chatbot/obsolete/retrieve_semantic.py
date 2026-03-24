# old keyword overlap retrieve method decide relevance by asking: 
    # Do the query and the chunk share the same words?
    # score = (tokens(query) ∩ tokens(chunk_answer)) / tokens(query)
# This semantic (dense) retrieval method decide relevance by asking: 
    # Do the query and the chunk mean similar things, even if the words are different?
    # score = sim(Embed(query), Embed(chunk))
# next: retrieve_dense.py
    # OOP, resuable
    # this file: Open the files again and tun the search for every question
    # new retrieve_dense.py: Already have the retriever ready in memory, now I just ask it new queries

from typing import List, Dict, Tuple # type this
from rag.embed import embed_query # converts query text to a vector
from rag.vector_store import load_index, load_metadata, search_index # helper function for search

def semantic_retrieve(query: str, index_path: str="data/faiss.index",
                      metadata_path: str="data/chunk_metadata.json", top_k:int=5) -> List[Tuple[Dict[str, str], float]]:
    """
    Retrieve the most semantically similar chunks for a user query. 
    This is your new semantic retriever. 
    Steps:
    1. load the FAISS index
    2. load metadata
    3. embed the query
    4. search the index
    5. map vector positions back to original chunk records
    Returns: list of typles: [(chunk_dict, score),...]
    """
    index = load_index(index_path) #load previously saved FAISS vector index
    metadata = load_metadata(metadata_path) # load metadata aligned with vector row order
    query_vector = embed_query(query) #turn user question into embedding vector

    scores, indices = search_index(index, query_vector, top_k=top_k) # run vector similarity search
    results = [] # final result list
    for score, idx in zip(scores[0], indices[0]): # FAISS returns 2D arrays, first row is our single query
        if idx == -1: # FAISS may return -1 if no valid result is found
            continue
        chunk_record = metadata[idx] # retrieve the original chunk metadata by row index
        results.append((chunk_record, float(score))) #store chunk + similarity score
    return results # return ranked retrieval results




