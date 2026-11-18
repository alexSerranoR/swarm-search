"""Buscador aleatorio: el 'suelo absoluto'. Sirve para probar que el pipeline funciona.

Todo buscador de este proyecto sigue la misma interfaz:
    search(query: str, k: int) -> list[str]   # doc_ids ordenados
"""
import random


class RandomRetriever:
    def __init__(self, corpus: dict[str, dict], seed: int = 0):
        self.doc_ids = list(corpus)
        self.rng = random.Random(seed)

    def search(self, query: str, k: int = 10) -> list[str]:
        return self.rng.sample(self.doc_ids, k)
