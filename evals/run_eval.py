"""Runner de evaluación: ejecuta un buscador sobre todas las queries y agrega métricas.
Usa TUS métricas de evals/metrics.py.

Uso:
    python evals/run_eval.py --retriever random
    python evals/run_eval.py --retriever bm25
    python evals/run_eval.py --retriever agent --limit 5 --verbose   # prueba barata primero
    python evals/run_eval.py --retriever agent --limit 50
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


def build_retriever(name: str, corpus, verbose: bool = False):
    if name == "random":
        from retrievers.random_retriever import RandomRetriever
        return RandomRetriever(corpus)
    if name == "bm25":
        from retrievers.bm25_retriever import BM25Retriever
        return BM25Retriever(corpus)
    if name == "agent":
        from agents.search_agent import SearchAgent
        return SearchAgent(corpus, verbose=verbose)
    # Fase 4: "swarm", "rrf_n"...
    raise ValueError(f"Retriever desconocido: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="random")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--limit", type=int, default=None, help="evaluar solo N queries")
    parser.add_argument("--verbose", action="store_true", help="mostrar las tool calls del agente")
    args = parser.parse_args()

    corpus, queries, qrels = load_dataset(args.dataset)
    print(f"Construyendo '{args.retriever}' sobre {len(corpus)} documentos...")
    retriever = build_retriever(args.retriever, corpus, args.verbose)

    qids = list(queries)[: args.limit] if args.limit else list(queries)
    per_query = []
    t0 = time.perf_counter()
    for n_done, qid in enumerate(qids, 1):
        start = time.perf_counter()
        ranked = retriever.search(queries[qid], k=K)
        latency = time.perf_counter() - start
        row = {
            "qid": qid,
            "recall@10": recall_at_k(ranked, qrels[qid], K),
            "ndcg@10": ndcg_at_k(ranked, qrels[qid], K),
            "latency_s": latency,
            "ranked": ranked,
        }
        # Los agentes guardan estadísticas del episodio (turnos, tokens...) en last_stats
        stats = getattr(retriever, "last_stats", None)
        if stats:
            row["stats"] = dict(stats)
        per_query.append(row)
        if args.verbose or (stats and n_done % 10 == 0):
            extra = f"  turnos={stats['turns']} busquedas={stats['searches']}" if stats else ""
            print(f"[{n_done}/{len(qids)}] {qid}  ndcg={row['ndcg@10']:.2f}{extra}")
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

    with_stats = [r["stats"] for r in per_query if "stats" in r]
    if with_stats:
        summary["mean_turns"] = sum(s["turns"] for s in with_stats) / n
        summary["mean_searches"] = sum(s["searches"] for s in with_stats) / n
        summary["total_input_tokens"] = sum(s["input_tokens"] for s in with_stats)
        summary["total_output_tokens"] = sum(s["output_tokens"] for s in with_stats)
        summary["pct_reported"] = sum(1 for s in with_stats if s["reported"]) / n

    suffix = f"_n{args.limit}" if args.limit else ""
    out = ROOT / "results" / f"{args.dataset}_{args.retriever}{suffix}.json"
    out.write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2))

    print(f"\n{args.retriever} en {args.dataset} ({n} queries)")
    print(f"  recall@10 : {summary['recall@10']:.3f}")
    print(f"  ndcg@10   : {summary['ndcg@10']:.3f}")
    print(f"  latencia  : {summary['mean_latency_s'] * 1000:.1f} ms/query")
    if with_stats:
        print(f"  turnos    : {summary['mean_turns']:.1f} de media, "
              f"{summary['mean_searches']:.1f} busquedas/query")
        print(f"  tokens    : {summary['total_input_tokens']:,} input / "
              f"{summary['total_output_tokens']:,} output (míralo en la Console para el coste)")
        print(f"  reportó   : {summary['pct_reported'] * 100:.0f}% de las queries")
    print(f"  guardado en {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
