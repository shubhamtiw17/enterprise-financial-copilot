

def evaluate_faithfulness(question: str, answer: str, context: str) -> float:
    """
    Scores whether the answer is grounded in the retrieved context.
    Returns a score between 0.0 and 1.0.
    """
    # Simple heuristic: check overlap of key terms between answer and context
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    overlap = answer_tokens & context_tokens
    if not answer_tokens:
        return 0.0
    return round(len(overlap) / len(answer_tokens), 4)
