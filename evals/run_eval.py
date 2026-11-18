"""Evaluation runner: runs a retriever over the test queries and aggregates metrics.

Usage:
    python evals/run_eval.py --retriever random
    python evals/run_eval.py --retriever bm25 --limit 50
    python evals/run_eval.py --retriever agent --limit 5 --verbose     # cheap smoke test first
    python evals/run_eval.py --retriever rrf_n --limit 30              # N independent agents + RRF
    python evals/run_eval.py --retriever swarm --limit 30              # N communicating agents + RRF
    python evals/run_eval.py --dataset fiqa --retriever swarm --agents 6 --limit 100
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


def build_retriever(name: str, corpus, verbose: bool = False, dataset: str = "scifact",
                    agents: int = 3):
    if name == "random":
        from retrievers.random_retriever import RandomRetriever
        return RandomRetriever(corpus)
    if name == "bm25":
        from retrievers.bm25_retriever import BM25Retriever
        return BM25Retriever(corpus)
    if name == "agent":
        from agents.search_agent import SearchAgent
        return SearchAgent(corpus, verbose=verbose, dataset=dataset)
    if name == "rrf_n":
        from agents.swarm import RRFEnsemble
        return RRFEnsemble(corpus, n=agents, verbose=verbose, dataset=dataset)
    if name == "swarm":
        from agents.swarm import Swarm
        return Swarm(corpus, n=agents, verbose=verbose, dataset=dataset)
    raise ValueError(f"Retriever desconocido: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="random")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N queries")
    parser.add_argument("--verbose", action="store_true", help="print every agent tool call")
    parser.add_argument("--agents", type=int, default=3, help="number of agents for rrf_n and swarm")
    args = parser.parse_args()

    corpus, queries, qrels = load_dataset(args.dataset)
    print(f"Building '{args.retriever}' over {len(corpus)} documents...")
    retriever = build_retriever(args.retriever, corpus, args.verbose, args.dataset, args.agents)

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
        # Agents store per-episode stats (turns, tokens, ...) in last_stats
        stats = getattr(retriever, "last_stats", None)
        if stats:
            row["stats"] = dict(stats)
        per_query.append(row)
        if args.verbose or (stats and n_done % 10 == 0):
            extra = f"  turns={stats['turns']} searches={stats['searches']}" if stats else ""
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
        summary["mean_messages"] = sum(s.get("messages", 0) for s in with_stats) / n
        if any("dup_query_rate" in s for s in with_stats):
            summary["dup_query_rate"] = sum(s.get("dup_query_rate", 0) for s in with_stats) / n

    name = args.retriever
    if args.retriever in ("rrf_n", "swarm") and args.agents != 3:
        name += f"{args.agents}"
    suffix = f"_n{args.limit}" if args.limit else ""
    out = ROOT / "results" / f"{args.dataset}_{name}{suffix}.json"
    out.write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2))

    print(f"\n{args.retriever} on {args.dataset} ({n} queries)")
    print(f"  recall@10 : {summary['recall@10']:.3f}")
    print(f"  ndcg@10   : {summary['ndcg@10']:.3f}")
    print(f"  latency   : {summary['mean_latency_s'] * 1000:.1f} ms/query")
    if with_stats:
        print(f"  turns     : {summary['mean_turns']:.1f} mean, "
              f"{summary['mean_searches']:.1f} searches/query")
        print(f"  tokens    : {summary['total_input_tokens']:,} input / "
              f"{summary['total_output_tokens']:,} output")
        print(f"  reported  : {summary['pct_reported'] * 100:.0f}% of queries")
        if summary["mean_messages"]:
            print(f"  messages  : {summary['mean_messages']:.1f} per query")
        if "dup_query_rate" in summary:
            print(f"  dup. queries: {summary['dup_query_rate'] * 100:.0f}% of searches repeat another agent's query")
    print(f"  saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
