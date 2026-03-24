# test_step3.py

# Import the function you want to test.
from answer import answer_from_evidence


# Create fake retrieval results in the same shape as your retriever returns:
# a list of (chunk_dict, score_float) tuples.
results = [
    (
        {
            "doc_id": "refund_policy",
            "chunk_id": "chunk_0",
            "text": "Customers may request a full refund within 30 days if the item is unused and in original packaging. Final-sale items are not eligible for refunds."
        },
        0.91
    )
]

# Create a sample user question.
question = "Can I return a final-sale item?"

# Call your Step 3 function.
response = answer_from_evidence(question, results)

# Print the structured result.
print(response)

# If BotResponse is a Pydantic model, this usually looks nicer:
print(response.model_dump_json(indent=2))