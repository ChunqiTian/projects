"""
When you run this file
1. It creates the sample chunks
2. SparseRetriever builds embeddings from the same chunk texts
3. DenseRetriever builds embeddings from the same chunk texts
4. HybridRetriever asks both retrievers for candidate chunks
5. It normalizeds both score lists
6. It merges scores by chunk_id
7. It ranks the chunks and returns the best final evidence
"""

from typing import List # imort List type hint
from rag.retrieve_sparse import SparseRetriever 
from rag.retrieve_dense import DenseRetriever
from rag.retrieve_hybrid import HybridRetriever

def fake_embeded_fn(texts: List[str]) -> List[list[float]]:
    """
    A tiny fake embedding function for learning/demo only
    Why this exists:
    - Your real project should use an actual embedding model
    - But this demo shows how the retriever code is wired together
    Important: Replace this with your real embedding funciton from previous steps
    """
    # Create simple toy vectors based on presence of a few keywords
    # This is NOT real semantic embedding; it is only for demonstrating code structure
    vectors = []

    for text in texts: 
        text_lower = text.lower()
        refund_score = 1.0 if "refund" in text_lower else 0.0
        password_score = 1.0 if "password" in text_lower else 0.0
        billing_score = 1.0 if "billing" in text_lower else 0.0
        shipping_score = 1.0 if "shippping" in text_lower else 0.0
        vectors.append([refund_score, password_score, billing_score, shipping_score])
    return vectors

def main():
    """
    Demo the hybrid retriever on a tiny support knowledge base. 
    what this function does:
    1. Create sample chunks
    2. Build sparse retriever
    3. Build dense retriever
    4. Build hybrid retriever
    5. Run a sample query
    6. Print ranked results
    """
    # Create a tiny mock knowledge base of support chunks
    chunks = [
        {
            "chunk_id": "refund_policy:chunk_0",
            "doc_id": "refund_policy",
            "text": "Refunds are allowed within 30 days for unused items in original packaging."
        },
        {
            "chunk_id": "refund_policy:chunk_1",
            "doc_id": "refund_policy",
            "text": "Final sale items are not eligible for refund."
        },
        {
            "chunk_id": "password_help:chunk_0",
            "doc_id": "password_help",
            "text": "To reset your password, click the forgot password link on the sign-in page."
        },
        {
            "chunk_id": "billing_faq:chunk_0",
            "doc_id": "billing_faq",
            "text": "Billing statements are available in the account settings page under invoices."
        },
        {
            "chunk_id": "shipping_policy:chunk_0",
            "doc_id": "shipping_policy",
            "text": "Standard shipping takes 5 to 7 business days."
        },
    ]

    # Build the sparse retriever
    sparse = SparseRetriever(chunks)

    # Build the dense retriever
    dense = DenseRetriever(chunks, embed_fn=fake_embeded_fn)

    # Build the hybrid retriever with equal weighting
    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, alpha=0.5)

    # Example user query
    query = "Can I get a refund for a final sale item?"

    # Retrieve the final hybrid top results
    results = hybrid.retreiver(query, k=3, per_source_k=5)

    # print the user query
    print(f"\nQuery: {query}\n")

    # Print ranked retrieval results
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"{rank}, {chunk["chunk_id"]} | score={score:.3f}")
        print(f"   {chunk["text"]}\n")

if __name__ == "__main__":
    main()

"""
How to connect this to your previous project

You probably already have:
- chunk building
- citations
- answer generation
- refusal threshold

So your pipeline may become:

chunks = build_chunks("knowledge_base")
sparse = SparseRetriever(chunks)
dense = DenseRetriever(chunks, embed_fn=your_real_embed_fn)
hybrid = HybridRetriever(sparse, dense, alpha=0.5)
results = hybrid.retrieve(user_question, k=3)
answer = answer_from_evidence(user_question, results)
"""
