# Commercial Support Chatbot

A portfolio-style customer support chatbot built with retrieval-augmented generation (RAG), citation-based answering, refusal logic, evaluation, and workflow routing.

## Project Goal

The goal of this project is to build a trustworthy support assistant that can:

- answer support and policy questions from internal documents
- show citations for retrieved evidence
- refuse when evidence is weak or missing
- route some requests to tools or escalation workflows
- expose debugging and evaluation signals for transparency

This project is designed as a learning and portfolio project that demonstrates practical LLM system design beyond simple chat completion.

---

## Why This Project Exists

Many beginner chatbot projects can generate fluent answers, but they often:

- hallucinate
- hide where answers came from
- fail silently on weak evidence
- lack evaluation
- lack workflow orchestration

This project focuses on solving those issues by combining:

- retrieval
- grounded answer synthesis
- citations
- refusal logic
- evaluation harness
- optional tool routing
- debugging visibility

---

## Key Features

- Knowledge-base question answering from support documents
- Chunk-based retrieval
- Citation formatting like `[refund_policy:chunk_0]`
- Refusal when evidence is weak
- Tool routing for operational workflows
- Human escalation support
- Evaluation harness with quality metrics
- Streamlit UI for demo and portfolio presentation
- Logging for retrieval and decision visibility

---

## Architecture

High-level system flow:

1. User sends a message
2. Router decides whether to:
   - answer from docs
   - call a tool
   - escalate
3. If docs are needed:
   - retrieve top chunks
   - check evidence strength
   - synthesize answer from evidence only
   - attach citations
4. UI displays answer, citations, confidence, and optional debug info

See full architecture notes in `docs/architecture.md`.

---

## Safety Design

This chatbot is intentionally designed to be conservative.

Safety behaviors include:

- refusal when retrieval evidence is too weak
- grounded answering from retrieved text only
- transparent citation display
- human escalation path for unsupported or sensitive cases
- explicit logging for debugging internal decisions

This reduces the risk of unsupported or hallucinated answers.

---

## Retrieval Design

The retrieval pipeline uses chunked support documents.

Current design:

- documents are loaded from a knowledge-base folder
- each document is split into chunks
- chunks are stored with metadata:
  - `doc_id`
  - `chunk_id`
  - `text`
- retrieval returns top matching chunks with scores
- answer generation uses only retrieved evidence

Depending on your current step, retrieval may be:

- keyword retrieval
- vector retrieval
- or hybrid retrieval

---

## Citation Behavior

When the chatbot answers from documents, it includes citations such as:

`[refund_policy:chunk_0]`

This allows users and evaluators to inspect where the answer came from.

Citation behavior goals:

- make answers more trustworthy
- improve debugging
- support evaluation
- reduce black-box behavior

---

## Example Queries

### Policy question
**User:**  
Can I return a final-sale item?

**Bot:**  
Final-sale items are not eligible for refunds based on the policy documents reviewed.  
Citations: `[refund_policy:chunk_1]`

### Tool-routed question
**User:**  
Where is my order 12345?

**Bot:**  
I checked order `12345`. It is currently in transit and expected to arrive in 2 business days.

### Escalation
**User:**  
I want to speak to a human representative.

**Bot:**  
I can escalate this request to a human support agent.

More examples are in `docs/example_queries.md`.

---

## Evaluation Results

Example metrics tracked in this project include:

- retrieval hit rate
- citation presence rate
- refusal correctness
- format validity
- groundedness checks

Example summary:

- Retrieval hit rate: 85%
- Citation presence rate: 100%
- Refusal correctness: 90%
- Format validity: 100%

These values are examples. Replace them with your real eval results from `eval/baseline_results.json`.

---

## UI / Demo

The project can be demonstrated through:

- command-line interface
- Streamlit chat interface
- debug panel showing retrieval chunks, scores, and routing

Suggested screenshots:

- main chat UI
- citation display
- debug expander
- refusal example
- tool-routing example

See `docs/screenshots.md`.

---

## Limitations

Current limitations include:

- retrieval quality depends on chunking quality
- the bot may miss answers if wording differs too much from source documents
- refusal threshold may need tuning
- tool routing is rule-based or mock-based in early versions
- evaluation dataset is relatively small
- no full production authentication or backend security layer

---

## Future Improvements

Possible next improvements:

- reranking after retrieval
- better confidence calibration
- stronger prompt injection defenses
- larger and more diverse evaluation set
- support for conversation memory
- richer admin dashboard
- production database and API deployment
- real ticketing / order APIs instead of mock tools

---

## Project Structure

```text
commercial-chatbot/
├── main.py
├── app.py
├── eval_runner.py
├── requirements.txt
├── README.md
├── rag/
│   ├── ingest.py
│   ├── chunk.py
│   ├── retrieve.py
│   ├── answer.py
│   └── ...
├── eval/
│   ├── eval_dataset.json
│   └── baseline_results.json
├── docs/
│   ├── architecture.md
│   ├── example_queries.md
│   └── screenshots.md
└── assets/
    └── screenshots and diagrams