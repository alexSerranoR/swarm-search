"""BM25: el buscador clásico por palabras clave. Baseline de RAG clásico (fase 2).

También es la herramienta de búsqueda que usarán el agente (fase 3) y el swarm (fase 4),
igual que SID usa BM25 como una de las tools de sus agentes.
"""
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Minúsculas y palabras alfanuméricas (quita la puntuación)."""
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
        """Devuelve la ventana de `size` palabras del documento con más palabras de la query.

        Es la idea de los 'snippets' de SID-2: el agente no lee documentos enteros en cada
        búsqueda, solo el trozo más prometedor. Si le interesa, pide el documento completo.
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
