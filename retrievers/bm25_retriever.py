"""BM25 keyword retriever.

Used both as the classic-retrieval baseline and as the `search` tool of every agent
(SID-2 also gives its agents a BM25 tool).
"""
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (punctuation removed)."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, corpus: dict[str, dict]):
        self.corpus = corpus
        self.doc_ids = list(corpus)
        docs = [tokenize(self.full_text(d)) for d in self.doc_ids]
        self.bm25 = BM25Okapi(docs)

    def full_text(self, doc_id: str) -> str:
        doc = self.corpus[doc_id]
        return (doc["title"] + " " + doc["text"]).strip()

    def search_with_scores(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def search(self, query: str, k: int = 10) -> list[str]:
        return [doc_id for doc_id, _ in self.search_with_scores(query, k)]

    def snippet(self, doc_id: str, query: str, size: int = 60) -> str:
        """Return the `size`-word window of the document with the most query terms.

        Same idea as SID-2's snippets: search results show only the most promising
        window of each document; the agent can read the full text if it needs to.
        """
        words = self.full_text(doc_id).split()
        if len(words) <= size:
            return " ".join(words)
        q = set(tokenize(query))
        best_start, best_hits = 0, -1
        step = max(1, size // 3)
        for start in range(0, len(words) - size + 1, step):
            window = words[start:start + size]
            hits = sum(1 for w in window if re.sub(r"\W+", "", w.lower()) in q)
            if hits > best_hits:
                best_start, best_hits = start, hits
        prefix = "... " if best_start > 0 else ""
        suffix = " ..." if best_start + size < len(words) else ""
        return prefix + " ".join(words[best_start:best_start + size]) + suffix
