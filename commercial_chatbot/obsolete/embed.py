from typing import List # type hints for lists of strings / vectors
from sentence_transformers import SentenceTransformer # embedding model 

# Load the embedding model once at import time. 
# This avoids reloading the model again and again for every query. 
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of text strings into embedding vectors. 
    Why we need this:
    - FAISS can only search vectors, not raw text. 
    - So every chunk must be transformed into a numeric representation. 
    Parameters: text - list of strings
    Returns: list of vectors, one vector per input text
    """
    embeddings = MODEL.encode(texts, noralize_embeddings=True) # make embeddings with L2 normalization
    return embeddings.tolist() # convert numpy array to normal Python

def embed_query(query: str) -> List[float]:
    """
    Convert a single user query into one embedding vector. 
    Why separate from embed_texts?
    - Because retrieval usually handles one query at a time. 
    - this keeps the API clean and easy to understand. 
    Parameters: query - the user's question
    Returns: one embedding vector as a Python list
    """
    embedding = MODEL.encode([query], normalize_embeddings=True)[0] 
        # encode a single-item list, then take first return
        # why normalize vectors - cosine similarity and inner product become closely related
    return embedding.tolist() # convert vector to plain Python list


