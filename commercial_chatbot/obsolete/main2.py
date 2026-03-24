#Step 4 - Embeddings + vector search
"""
query text -> embedding vector -> nearest chunk vectors -> best chunk

Retrieve(query) = argmax_i similarity_func(Embed(query), Embed(chunk_i))

docs
├── rag/
│   ├── chunk.py
│   ├── embed.py
│   ├── vector_store.py
│   └── retrieve.py
│
└── data/
    ├── chunks.json
    ├── chunk_metadata.json
    └── faiss.index

Offline indexing step:
read docs -> chunk docs -> embed each chunk -> store vectors in FAISS
 -> store metadata in JSON

 Online retreival step
 embed query -> serch FAISS -> get top-k chunk ids -> load metadata
  -> return chunks + scores 

eg. user: How quickly will my package arrive?
    doc: standard shipping takes 5-7 days
- Keyword matching may miss

Goal: Make retrieval more semantic

You add
- embeddings
- vector index
- nearest-neighbor retrieval

Tooling options:
- sentence-transformers
- faiss or chromadb

Main learning: semantic similarity in vector space (one of most important LLM engineering skill)

Flow:
1. User question
2. Convert question into embedding vector
3. Compare against stored chunk vectors
4. Find most semantically similar chunks
5. Send top chunks to answer generator
"""

from schemas import BotResponse  # structured response model
from guardrails import should_escalate, out_of_scope_reason  # safety / routing helpers
from rag.answer import answer_from_evidence  # builds answer from retrieval evidence
from rag.retrieve_semantic import semantic_retrieve  # NEW semantic retriever

def handle_message(user_text: str) -> BotResponse: # replaced by handler.py in step 8
    """
    Main chatbot handler
    Flow: 1. guardrails; 2. retrieval; 3. answer synthesis
    """
    reason = out_of_scope_reason(user_text)  # check if request is outside supported scope
    if reason: # if outside scope, refuse
        return BotResponse(
            inten="refuse",
            answer=reason,  
            citations=[],
            needs_human=False,
            confidence=0.95
        )
    
    if should_escalate(user_text):  # if human support is needed
        return BotResponse(
            intent="ecalate",
            answer="I can help with policy questions, but this request should go to a human support agent.",
            citations=[],
            needs_human=True, 
            confidence=0.98   
        )

    results = semantic_retrieve(user_text, top_k=5)  # retrieve the most semantically similar chunks
    return answer_from_evidence(user_text, results)  # synthesize answer from retrieved evidence


# Step 5 - Hybird Retrieval
"""
A production-quality system usually does not rely on only one retriever. It combines:
- Sparse retrieval -> exact word matching
- Dense retrieval -> semantic meaning matching
Previously: User Q -> vector search -> top-k chunks -> answer
Now upgrade to : User Q -> sparse retrieval / dense retrieval -> merge+rerank -> final evidence -> answer

Thinking process:
1. Define sparse / dense?
- "Can I get my money back for a final sale item?" -> "returns", "refund rules" -> dense (meaning)
- "How do I reset my password?" -> "password reset" -> sparse (exact words)
2. Keep the architecture modular
- Don't write one huge func that does everything but split to small pieces:
    - build sparse/dense index  -> score query with sparse/dense -> normalize scores -> merge results -> return top chunks
- Because later may replace: TF-IDF with BM25 / fake embeddings with read embeddings / simple merge with weighted reranking
3. Standardize the chunk format
- sparse and dense returns chunk ids, then merge can compare them easily
4. Think about score comparability 
- sparse (TF-IDF) and dense scores have diff numeric scales -> normalize them to 0-1 -> weighted avg them
    - eg. hybrid score = 0.5 * sparse score + 0.5 * dense score
5. Decide merge strategy before coding
A. weighted score merge - for each chunk: final score = alpha * sparse + (1-alpha) * dense
B. rank fusion - Instead of raw scores, combine positions: eg. sparse rank #1, dense #4, combine ranks
C. retrieve separately, then rerank with another model

Files: chunk.py; sparse_retriever.py; dense_retriever.py; hybrid_retriever; demo_hybrid.py
"""




def main() -> None:
    # Simple CLI loop for testing the chatbot. 

    print("Commercial Chatbot (step 4 - semantic retrieval). Type 'q' to quit.")
    
    while True:
        user_text = input("\nYou: ").strip() # read user input
        if user_text.lower() in {"q", "quit"}:  # exit condition
            break

        response = handle_message(user_text) # process user message

        print("\nBot (structured JSON)")
        print(response.model_dump_json(indent=2)) # pretty-pprint structureed output

if __name__=="__main__":
    main() # start CLI ap

# Note step 8 intro in file action_router.py










