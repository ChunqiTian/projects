"""
- load_docs(folder_path):loading markdown files
    - input: folder path
    - output: list of raw docs
- split_into_chunks(text, chunk_size): splitting docs into chunks
    - input: doc text and chunk size
    - output: list of chunk strings
- build_chunks(folder_path, chunk_size): building final chunk records
    - input: folder path and chunk size
    - output: list of chunk records with IDs
"""

from pathlib import Path # work with file paths cleanly
from typing import List, Dict # type hints for readable code
"""
def load_docs(folder_path: str) -> List[Dict[str, str]]:
    
    Load all .md files in the folder and return a dict of {doc_name: content}
    1. Opens the folder you provide
    2. Finds every markdown file
    3. Reads each file as text
    4. Returns a list of dictionaries like:   [{"doc_id": "refund_policy", "text": "..."}]
    
    Why this matters:
    Retrieval can't happen unless we first ingest the source documents.
    This function is the doc loading step of the RAG pipeline.
    
    docs = [] # list of docs

    folder = Path (folder_path) # convert the folder path string into a Path object for easier file handling

    for md_file in Path(folder_path).glob("*.md"): # loop through every markdow file
        text = md_file.read_text(encoding="utf-8") # read file content as a string

        docs.append(
            {
                "doc_id": md_file.stem, # file name without ".dm", eg. "refund_policy"
                "text": text.strip(), # remove extra whitespace from start/end
            }
        )
    return docs # return all loaded docs
    """

def split_into_chunks(text: str, chunk_size: int=2) -> List[str]:
    """
    Split a doc into small paragraph-based chunks.

    Parameters:
    - text: the full doc text
    - chunk_size: how many paragraphs to group into one chunk

    Why paragraph-based chunking?
    It is simple, readable, and usually better than cutting by raw character count 
    when you are just starting.

    Example:
    If a doc has 5 paragraphs and chunk_size=2, the chunks may become:
    chunk 0 -> paragraphs 1 and 2
    chunk 1 -> paragraphs 3 and 4
    chunk 2 -> paragraph 5

    This gives retrieval smaller units of meaning.
    """        
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()] # split on blank lines
    chunks = [] # this will store grouped chunk strings
    for i in range(0, len(paragraphs), chunk_size): # move through paragraphs in steps of chunk_size
        group = paragraphs[i:i+chunk_size] # take a slice of paragraphs
        chunk_text = "\n\n".join(group) # join them back into one chunk
        chunks.append(chunk_text) # store the chunk
    return chunks # return a list of chunks


def build_chunks(folder_path: str, chunk_size: int=2) -> List[Dict[str, str]]:
    """
    Build the final chunk records used by retrieval.

    Output format example:
    [
        {
            "chunk_id": "refund_policy: chunk_0",
            "doc_id": "refund_policy",
            "text": "Refund Policy ...",
        }
    ]

    Why this funcion exists:
    - split_into_chunks() splits each doc into pieces
    - build_chunks() converts everything into one clean searchable structure
    """

    all_chunks = [] # this will hold every chunk from every doc
    folder = Path(folder_path) # convert string path into a path object

    for file_path in folder.glob("*.md"): # loop through evvery markdown file
        doc_id = file_path.stem # file name without ".md", used as the doc id
        text = file_path.read_text(encoding="utf-8") # read the doc text

        chunks = split_into_chunks(text) # split the doc into chunks

        for idx, chunk_text in enumerate(chunks): # assign an index to each chunk for unique ID
            all_chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"chunk_{idx}",
                    "text": chunk_text,
                }
            )
    return all_chunks # ready for retrieval




