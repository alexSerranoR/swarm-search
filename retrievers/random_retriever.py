"""Random retriever: the absolute floor, used to sanity-check the pipeline.

Every retriever in this project exposes the same interface:
    search(query: str, k: int) -> list[str]   # ranked doc IDs
"""
import random


class RandomRetriever:
    def __init__(self, corpus: dict[str, dict], seed: int = 0):
        self.doc_ids = list(corpus)
        self.rng = random.Random(seed)

    def search(self, query: str, k: int = 10) -> list[str]:
        return self.rng.sample(self.doc_ids, k)
