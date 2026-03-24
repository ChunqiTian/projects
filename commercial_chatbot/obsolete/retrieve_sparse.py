from typing import List, Dict, Tuple # import type hints for cleaner, safer code
from sklearn.feature_extraction.text import TfidfVectorizer # convert text into TF-IDF vectors
from sklearn.metrics.pairwise import cosine_similarity # compute cosine similarity between vectors

"""
TF-IDF means
- TF = term freq in a chunk
- IDF = inverse doc freq
- rare but important words get more weight
It gives each chunk a vector based on words. 
Then compare the query vector to each chunk vector using cosine similarity.

"""



class SparseRetriever:
    """
    Also see note in main.py (step 5)
    Keyword overlap retriever vs sparse retriever:
    - KOR - word count
    - SR - words -> vectors -> score. [advanced]
    A simple sparse retriever using TF-IDF + cosine similarity

    Why this class exists:
    - It gives us keyword-based retrieval
    - It is strong when exact words matter
    - It acts as the sparse half of hybrid retrieval
    
    Main idea:
    - Fit a TF-IDF vectorizer on all chunk texts
    - Transform the user query into the same vector space
    - Compute similarity between query and every chunk
    - Return the best matching chunks
    """

    def __init__(self, chunks: List[Dict[str, str]]):
        """
        Initialize the sparse retriever
        Param: chunks - a list of chunk dict, each containing: chunk_id, doc_id, text
        What happens here:
        1. Save the original chunks
        2. Extract just the text from each chunk
        3. Build a TF-IDF vectorizer
        4. Fit the vectorizer on all chunk texts
        5. Store the doc matrix for future retrieval
        """
        self.chunks = chunks # store original chunks
        self.texts = [chunk["text"] for chunk in chunks] # extract the plain text from each chunk

        # Create a TF-IDF vectorizer
        # Lowercase=True means text is normalized to lower case
        # stop_words="english" removes common Eng words like "the", "is", "a"
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")

        # Fit the vectorizer on the chunk texts and transform them into a TF-IDF matrix
        # Shape will be: (num_of_chunks, vocabulary_size)
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, k: int=5) -> List[Tuple[Dict[str, str], float]]:
        """
        Retrieve the top-k chunks for a query using sparse TF-IDF matching
        Params: query - the user q; k - how many chunks to return
        Returns: a list of tuples: [(chunk_dict, score), ...]
        Detailed logic:
        1. Transform the query into a TF-IDF vector
        2. Compare it with every chunk vector using cosine similarity
        3. Sort scores descending
        4. Return the top-k chunks with their scores
        """
        # Convert the query text into the same TF-IDF vector space as the docs. 
        query_vector = self. vectorizer.transform([query])
        # Compute cosine similarity between query vector and all chunk vectors
        # Result shape is (1, #_of_chunks), so we take [0] to get a flat array. 
        scores = cosine_similarity(query_vector, self.doc_matrix)[0]
        # Create pairs of (chunk, score) for every chunk.
        chunk_score_pairs = list(zip(self.chunks, scores))
        # Sort by similarity score in descending order (highest score first)
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
        # Return only the top-k results
        return chunk_score_pairs[:k]
    







