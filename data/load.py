"""Carga un dataset en formato BEIR. Código de fontanería: ya está hecho.

Formato BEIR:
- corpus.jsonl  -> {"_id", "title", "text"} por línea
- queries.jsonl -> {"_id", "text"} por línea
- qrels/test.tsv -> query-id \t corpus-id \t score   (score > 0 = relevante)
"""
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent


def load_corpus(name: str = "scifact") -> dict[str, dict]:
    corpus = {}
    with open(DATA_DIR / name / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc["text"]}
    return corpus


def load_queries(name: str = "scifact") -> dict[str, str]:
    queries = {}
    with open(DATA_DIR / name / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(name: str = "scifact", split: str = "test") -> dict[str, dict[str, int]]:
    """Devuelve {query_id: {doc_id: grado_de_relevancia}} solo con grados > 0."""
    qrels: dict[str, dict[str, int]] = {}
    with open(DATA_DIR / name / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # cabecera
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load_dataset(name: str = "scifact", split: str = "test"):
    """Devuelve (corpus, queries, qrels) con solo las queries que tienen qrels en ese split."""
    corpus = load_corpus(name)
    qrels = load_qrels(name, split)
    queries = {qid: t for qid, t in load_queries(name).items() if qid in qrels}
    return corpus, queries, qrels
