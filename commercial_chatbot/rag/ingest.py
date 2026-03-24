# This file runs the ingestion pipeline and saves chunk metadata. 

import json # for saving chunk data
import os # for directory creation
from typing import List, Dict # type hints

from chunk import build_chunks # import chunk-building logic from chunk.py

def save_chunks(chunks: List[Dict[str, str]], output_path: str) -> None:
    """
    Save chunk records to a JSON file
    Why this function exists:
    - retrieval later needs persistent access to chunk metadata
    - saving once avoids rebuilding every time
    docs -> chunks -> saved metadata
    chunks -> embeddings -> FAISS index (later version)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # create output folder if it does not exist
    with open(output_path, "w", encoding="utf-8") as f: # open target file for writing
        json.dump(chunks, f, indent=2, ensure_ascii=False) # save readable JSON

def run_ingestion(
    docs_folder: str="knowledge_base",
    output_path: str="data/chunk_metadata.json",
    chunk_size: int=2
) -> List[Dict[str, str]]:
    """
    Run the ingestion pipeline.
    steps:
    1. build chunks from source docs
    2. save chunks to dist
    3. return the chunk list
    """
    chunks = build_chunks(docs_folder, chunk_size=chunk_size) # create chunk records
    save_chunks(chunks, output_path) # save them to disk
    return chunks # return for inspection or later use

def main() -> None: # Run ingestion directly from terminal
    chunks = run_ingestion() # execute the pipeline
    print(f"Save {len(chunks)} chunks to data/chunk_metadaata.json") # print summary

if __name__ == "__main__":
    main() # run only when this file is executed idrectly

