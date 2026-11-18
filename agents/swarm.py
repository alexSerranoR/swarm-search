"""Fase 4: la pregunta central del paper de SID-2.

Con el MISMO presupuesto (N agentes, mismos turnos máximos, mismo modelo):
    rrf_n  -> N agentes independientes; al final fusionamos sus rankings con TU rrf()
    swarm  -> N agentes en paralelo que pueden mandarse mensajes mientras buscan;
              al final también fusionamos con rrf() (igual que SID-2)

La única diferencia entre los dos es la COMUNICACIÓN. Si el swarm gana, coordinar aporta.

Simplificaciones respecto a SID-2 (cuéntalas si publicas resultados):
- Sin `fork` ni `wait`: siempre hay N agentes fijos desde el principio.
- Sin entrenamiento (RL): solo prompting. La Figura 4 del paper dice que eso limita mucho.
- `message` es un broadcast a todos (un tablón compartido), no mensajes 1 a 1.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agents.search_agent import SearchAgent, new_stats
from retrievers.bm25_retriever import BM25Retriever
from retrievers.rrf import rrf

N_AGENTS = int(os.environ.get("SWARM_AGENTS", "3"))

SWARM_ADDON = """

You are agent {i} of {n} in a team searching IN PARALLEL for the same query.
Your teammates run at the same time and share one final score: the team's fused ranking.
Coordination rules (important):
- In your FIRST turn, run your searches AND call `message` with your angle (which wordings or
  concepts you will cover) so teammates can pick different ones.
- After every search turn, if you found a promising document, `message` its ID and why.
- Messages from teammates appear at the end of your tool results. Read them before searching:
  NEVER repeat a query a teammate already ran; cover angles they have not covered, or verify
  their candidates. If a teammate already found strong candidates, you may finish early.
- Keep messages short and concrete. Agent 1 starts with the most literal reading of the query;
  higher-numbered agents should favor alternative wordings and related concepts.
- You still finish with `report_ids` (your own ranking; the team's rankings are fused)."""

MESSAGE_TOOL = {
    "name": "message",
    "description": "Send a short message to ALL teammates (they see it in their next tool result).",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}

INDEPENDENT_ADDON = """

You are agent {i} of {n} searching independently for the same query; your ranking will be
fused with the others. Agent 1 starts with the most literal reading of the query;
higher-numbered agents should favor alternative wordings and related concepts."""


class Board:
    """Tablón compartido. Cada agente tiene un cursor: solo recibe lo que no ha visto."""

    def __init__(self, n: int):
        self.lock = threading.Lock()
        self.posts: list[tuple[int, str]] = []
        self.cursor = [0] * n

    def post(self, sender: int, text: str) -> None:
        with self.lock:
            self.posts.append((sender, text))

    def inbox(self, me: int) -> str:
        with self.lock:
            new = [(s, t) for s, t in self.posts[self.cursor[me]:] if s != me]
            self.cursor[me] = len(self.posts)
        return "\n".join(f"[agent {s + 1}] {t}" for s, t in new)


def _merge_stats(all_stats: list[dict], wall_time: float) -> dict:
    merged = new_stats()
    for s in all_stats:
        for key in ("turns", "searches", "reads", "messages", "input_tokens", "output_tokens"):
            merged[key] += s[key]
        merged["reported"] = merged["reported"] or s["reported"]
    merged["llm_time_s"] = wall_time  # en paralelo: cuenta el tiempo real, no la suma
    merged["agents"] = len(all_stats)
    merged["agents_reported"] = sum(1 for s in all_stats if s["reported"])
    # Coordinación medible: % de búsquedas que repiten EXACTAMENTE la de otro agente
    norm = [[q.strip().lower() for q in s.get("queries", [])] for s in all_stats]
    total = sum(len(qs) for qs in norm)
    dup = 0
    for i, qs in enumerate(norm):
        others = {q for j, o in enumerate(norm) if j != i for q in o}
        dup += sum(1 for q in qs if q in others)
    merged["dup_query_rate"] = dup / total if total else 0.0
    merged["queries"] = [q for s in all_stats for q in s.get("queries", [])]
    return merged


class _MultiAgentBase:
    coordinated = False

    def __init__(self, corpus: dict[str, dict], n: int = N_AGENTS, verbose: bool = False,
                 dataset: str = "scifact"):
        bm25 = BM25Retriever(corpus)  # un solo índice compartido por todos los agentes
        self.n = n
        self.agents = [SearchAgent(corpus, verbose=verbose, bm25=bm25, dataset=dataset)
                       for _ in range(n)]
        self.last_stats: dict = {}

    def _run_one(self, i: int, query: str, board: Board | None):
        agent = self.agents[i]
        if board is None:
            return agent.run_episode(
                query, system=agent.system + INDEPENDENT_ADDON.format(i=i + 1, n=self.n),
                tag=f"[a{i + 1}] ")

        def handle_extra(name: str, args: dict) -> str:
            if name == "message":
                board.post(i, args.get("text", ""))
                return "Message sent."
            return f"Unknown tool: {name}"

        return agent.run_episode(
            query,
            system=agent.system + SWARM_ADDON.format(i=i + 1, n=self.n),
            extra_tools=[MESSAGE_TOOL],
            handle_extra=handle_extra,
            inbox=lambda: board.inbox(i),
            tag=f"[a{i + 1}] ",
        )

    def search(self, query: str, k: int = 10) -> list[str]:
        board = Board(self.n) if self.coordinated else None
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.n) as pool:
            outs = list(pool.map(lambda i: self._run_one(i, query, board), range(self.n)))
        wall = time.perf_counter() - t0

        rankings = [reported for reported, _ in outs if reported]
        fused = rrf(rankings) if rankings else []
        stats = _merge_stats([s for _, s in outs], wall)
        ranked = self.agents[0].pad_with_bm25(fused, query, k, stats)
        if board is not None:
            stats["board"] = [f"agent {s + 1}: {t}" for s, t in board.posts]
        self.last_stats = stats
        return ranked


class RRFEnsemble(_MultiAgentBase):
    """N agentes independientes + RRF (la línea base del paper)."""
    coordinated = False


class Swarm(_MultiAgentBase):
    """N agentes que se comunican mientras buscan + RRF."""
    coordinated = True
