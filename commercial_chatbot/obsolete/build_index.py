# Build vector database from docs
from pathlib import Path # for creating the data directory
from rag.chunk import load_and_chunk_documents # read and chunk knowledge base files
from rag.embed import embed_texts # embed all chunk texts
from rag.vector_store import build_faiss_index, save_index, save_metadata # vector store helpers

def main() -> None:
    """
    Build the semantic search index from knowledge base documents. 
    It prepares the data needed for fast semantic retrieval later
    workflow:
    1. read and chunk docs
    2. embed chunk texts
    3. build FAISS index
    4. save index + metadata
    """

    Path("data").mkdir(exist_ok=True) # create the data folder if it doesn't already exist
    chunks = load_and_chunk_documents("knowledge_base") #load docs and split into chunk dictionaries
    texts = [item["text"] for item in chunks] # extract raw text from each chunk for embedding

    embeddings = embed_texts(texts) # convert chunk texts into vectors
    index = build_faiss_index(embeddings) # build searchable FAISS index from vectors

    save_index(index, "data/faiss.index") # save vector index to disk
    save_metadata(chunks, "data/chunk_metadata.json") # save parallel metadata file

    print(f"Indexed {len(chunks)} chunks into data/faiss.index") # confirmation message

if __name__ == "__main__":
    main() # run the indexing script








