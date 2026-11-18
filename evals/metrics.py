"""Retrieval metrics: recall@k and NDCG@k."""
import math


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of the relevant documents that appear in the top-k of `ranked`.

    Args:
        ranked:   document IDs ordered from most to least relevant.
        relevant: {doc_id: relevance grade} for the ground-truth relevant documents.
        k:        cutoff.
    """
    if len(relevant) == 0:
        return 0.0
    found = 0
    for doc in ranked[:k]:
        if doc in relevant:
            found += 1
    return found / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (linear gain, log2 discount).

        DCG@k  = sum_{i=1..k} grade(ranked[i]) / log2(i + 1)
        IDCG@k = DCG@k of the ideal ranking (relevant docs sorted by grade)
        NDCG@k = DCG@k / IDCG@k
    """
    # DCG: what the retriever scored
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        dcg += relevant.get(doc, 0) / math.log2(i + 2)

    # IDCG: what the perfect ranking would score
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(ideal_grades):
        idcg += grade / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg
