"""Runner de evaluación: ejecuta un buscador sobre todas las queries y agrega métricas.
Código de fontanería: ya está hecho. Usa TUS métricas de evals/metrics.py.

Uso:
    python evals/run_eval.py --retriever random
    python evals/run_eval.py --retriever random --limit 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.load import load_dataset  # noqa: E402
from evals.metrics import ndcg_at_k, recall_at_k  # noqa: E402

K = 10


def build_retriever(name: str, corpus):
    if name == "random":
        from retrievers.random_retriever import RandomRetriever
        return RandomRetriever(corpus)
    # Fase 2+: añade aquí "bm25", "agent", "swarm", "rrf_n"...
    raise ValueError(f"Retriever desconocido: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="random")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--limit", type=int, default=None, help="evaluar solo N queries")
    args = parser.parse_args()

    corpus, queries, qrels = load_dataset(args.dataset)
    retriever = build_retriever(args.retriever, corpus)

    qids = list(queries)[: args.limit] if args.limit else list(queries)
    per_query = []
    t0 = time.perf_counter()
    for qid in qids:
        start = time.perf_counter()
        ranked = retriever.search(queries[qid], k=K)
        latency = time.perf_counter() - start
        per_query.append({
            "qid": qid,
            "recall@10": recall_at_k(ranked, qrels[qid], K),
            "ndcg@10": ndcg_at_k(ranked, qrels[qid], K),
            "latency_s": latency,
            "ranked": ranked,
        })
    total = time.perf_counter() - t0

    n = len(per_query)
    summary = {
        "retriever": args.retriever,
        "dataset": args.dataset,
        "n_queries": n,
        "recall@10": sum(r["recall@10"] for r in per_query) / n,
        "ndcg@10": sum(r["ndcg@10"] for r in per_query) / n,
        "mean_latency_s": sum(r["latency_s"] for r in per_query) / n,
        "total_time_s": total,
    }

    out = ROOT / "results" / f"{args.dataset}_{args.retriever}.json"
    out.write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2))

    print(f"\n{args.retriever} en {args.dataset} ({n} queries)")
    print(f"  recall@10 : {summary['recall@10']:.3f}")
    print(f"  ndcg@10   : {summary['ndcg@10']:.3f}")
    print(f"  latencia  : {summary['mean_latency_s'] * 1000:.1f} ms/query")
    print(f"  guardado en {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
