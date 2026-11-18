"""Reciprocal Rank Fusion (Cormack et al., 2009)."""


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse several rankings into one.

    Each document scores, for every ranking it appears in, 1 / (k + rank),
    with rank starting at 1. Documents missing from a ranking get nothing from it.

    Args:
        rankings: list of rankings; each ranking is an ordered list of doc IDs.
        k:        smoothing constant (60 is the standard value).

    Returns:
        Doc IDs sorted by fused score, highest first.

    Example:
        A = ["x", "y"], B = ["y"]   ->   y: 1/62 + 1/61,  x: 1/61   ->  ["y", "x"]
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for i, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0) + 1 / (k + i + 1)
    return sorted(scores, key=scores.get, reverse=True)
