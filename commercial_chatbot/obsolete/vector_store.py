import json # used to save and load metadata as JSON
from pathlib import Path # conenient path handling
from typing import List, Dict, Tuple # type hints

import faiss # vector search library
import numpy as np 

def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    """
    Build a FAISS index from embedding vectors; turns vectors into indexes
    Why IndexFlatIP?
    - Flat means exact nearst-neighbor search (simple and accurate).
    - IP means inner product similarity. 
    - Since embeddings are normalized, inner product works well for semantic similarity.
    Parameters: embeddings - list of embedding vectors
    Return: a FAISS index object
    """
    vectors = np.array(embeddings, dtype="float32") #FAISS expects float 32 arrays
    dim = vectors.shape[1] # embedding dimension eg. 384 for MiniLM
    index = faiss.IndexFlatIP(dim) # create an exact-search inner-product index
    index.add(vectors) # add all vectors into the index
    return index # return the built index

def save_index(index: faiss.IndexFlatIP, index_path:str) -> None:
    # Save a FAISS index to disk
    # Purpose: don't need to build embed everytime - consistent
    faiss.write_index(index, index_path) #write the index file to disk

def save_metadata(metadata: List[Dict[str, str]], metadata_path: str) -> None:
    # Save chunk metadata to a JSON file.
    # Purpose: FAISS stores vectors only, not original text/citations, so we keep metadata in parallel.
        # eg. In FAISS: row 0, row 2,, now mapping to doc id, chunk id...
    Path(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), 
                                   encoding="utf-8") # save pretty JSON for readability
    
def load_index(index_path: str) -> faiss.IndexFlatIP:
    """
    Load a FAISS index from disk.  For consistence purpose
    Returns: the previously saved FAISS index
    """
    return faiss.read_index(index_path) # read and return the index file

def load_metadata(metadata_path: str) -> List[Dict[str, str]]:
    """
    Load chunk metadata from disk. 
    Returns: list of chunk dictionaries
    Purpose: (same as save_metadata)
    """
    text = Path(metadata_path).read_text(encoding="utf-8") # read raw JSON text
    return json.load(text) # parse JSON into Python objects

def search_index(index: faiss.IndexFlatIP, query_vector: List[float], top_k: int=5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for the top-k nearest vectors.
    Parameters:
    - index: loaded FAISS index
    - query_vector: embedded user question
    - top_k: num of results to return
    Returns: 1. scores: similarity scores 2. indices: integer positions of matching chuk vectors
    """
    query = np.array([query_vector], dtype="float32") #FAISS expects a 2D float32 array of shape(1, dim)
    scores, indices = index.search(query, top_k) # perform nearest-neighbor search
    return scores, indices # return both scores and matched vector positions

