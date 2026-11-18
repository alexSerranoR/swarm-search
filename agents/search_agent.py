"""Single retrieval agent with a hand-written tool loop (no frameworks).

One episode (same shape as SID-2, but prompted instead of RL-trained):
    1. The model gets the query and 3 tools: search, read_doc and report_ids.
    2. It searches, reads snippets, reformulates and searches again (it can issue several
       searches in parallel in a single turn).
    3. When it is confident, it calls report_ids with a ranked list of doc IDs.
    4. If it reports fewer than k IDs (or runs out of turns), the ranking is padded with
       BM25 results for the original query. Padding is counted in the stats.

The loop lives in `run_episode()`. The swarm (agents/swarm.py) reuses this exact loop,
adding a `message` tool and an inbox.
"""
import json
import os
import time
from typing import Callable

import anthropic

from retrievers.bm25_retriever import BM25Retriever

# Small, fast model. Override with the AGENT_MODEL environment variable.
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-5-5")
MAX_TURNS = 8          # hard cap on model turns per episode (SID-2 uses a similar limit)
SEARCH_LIMIT = 10      # results per search call
SNIPPET_WORDS = 60     # snippet size in words

# Per-dataset corpus description, task and relevance guidance.
PROFILES = {
    "scifact": {
        "prefix": "Claim",
        "corpus": "a corpus of scientific paper abstracts",
        "task": "Given a claim, your job is to find the documents that support or refute it.",
        "relevance": """- A document is relevant if it provides evidence about the claim, whether it SUPPORTS or
  REFUTES it. Claims are often paraphrased, simplified or even wrong: the relevant paper
  may use different wording or numbers. Do not demand an exact match.
- Most claims have only 1-2 relevant documents. Precision at the top of your ranking matters most.""",
    },
    "fiqa": {
        "prefix": "Question",
        "corpus": "a corpus of answers from a personal finance and investing Q&A forum",
        "task": "Given a financial question, your job is to find the forum answers that best answer it.",
        "relevance": """- An answer is relevant if it directly addresses the question asked (not just the same topic).
- Questions and answers often use DIFFERENT vocabulary (e.g. question says "401k match",
  answer says "employer contribution"). Rephrase with the words an answerer would use.
- Questions usually have 1-4 relevant answers. Precision at the top of your ranking matters most.""",
    },
}

SYSTEM_TEMPLATE = """You are a retrieval agent searching {corpus}.
{task}

How to work:
- Use `search` with short keyword queries (BM25: exact words matter). Try synonyms,
  technical terms, and different angles. You may call `search` several times in one turn.
- Results show a snippet of each document. Use `read_doc` only if a snippet is ambiguous.
{relevance}
- When done, call `report_ids` ONCE with up to 10 document IDs, most relevant first.
  NEVER report an empty list: if unsure, report your best candidates anyway, ranked by likelihood.
  Be efficient: stop as soon as you are confident."""


def system_prompt(dataset: str) -> str:
    return SYSTEM_TEMPLATE.format(**PROFILES[dataset])


SYSTEM_PROMPT = system_prompt("scifact")  # backwards compatibility

TOOLS = [
    {
        "name": "search",
        "description": "BM25 keyword search over the corpus. Returns document IDs, titles and the best-matching snippet.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keyword query"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_doc",
        "description": "Read the full text of a document by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "report_ids",
        "description": "Finish the episode by reporting the relevant document IDs, most relevant first (max 10).",
        "input_schema": {
            "type": "object",
            "properties": {"doc_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["doc_ids"],
        },
    },
]


def new_stats() -> dict:
    return {"turns": 0, "searches": 0, "reads": 0, "messages": 0, "input_tokens": 0,
            "output_tokens": 0, "reported": False, "padded": 0, "llm_time_s": 0.0,
            "queries": []}


class SearchAgent:
    def __init__(self, corpus: dict[str, dict], model: str = MODEL, verbose: bool = False,
                 bm25: BM25Retriever | None = None, dataset: str = "scifact"):
        self.bm25 = bm25 or BM25Retriever(corpus)
        self.corpus = corpus
        self.model = model
        self.verbose = verbose
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.dataset = dataset
        self.system = system_prompt(dataset)
        self.prefix = PROFILES[dataset]["prefix"]
        self.last_stats: dict = {}

    # ---------------- tools ----------------

    def _tool_search(self, query: str, seen: set[str]) -> str:
        lines = []
        for doc_id, score in self.bm25.search_with_scores(query, SEARCH_LIMIT):
            title = self.corpus[doc_id]["title"]
            if doc_id in seen:
                # SID-2-style context management: never repeat text the agent has already seen.
                lines.append(f"[{doc_id}] {title}  (already seen)")
            else:
                seen.add(doc_id)
                lines.append(f"[{doc_id}] {title}\n    {self.bm25.snippet(doc_id, query, SNIPPET_WORDS)}")
        return "\n".join(lines) if lines else "No results."

    def _tool_read_doc(self, doc_id: str) -> str:
        if doc_id not in self.corpus:
            return f"Unknown doc_id: {doc_id}"
        return self.bm25.full_text(doc_id)

    # ---------------- the loop (the harness) ----------------

    def run_episode(
        self,
        query: str,
        system: str | None = None,
        extra_tools: list[dict] | None = None,
        handle_extra: Callable[[str, dict], str] | None = None,
        inbox: Callable[[], str] | None = None,
        tag: str = "",
    ) -> tuple[list[str], dict]:
        """Run one full episode. Returns (reported IDs, not padded; stats).

        extra_tools / handle_extra: additional tools (the swarm adds `message`).
        inbox: callable returning new messages from teammates; they are appended to the
               next tool result, as in SID-2.
        """
        system = system or self.system
        tools = TOOLS + (extra_tools or [])
        messages = [{"role": "user", "content": f"{self.prefix}: {query}"}]
        seen: set[str] = set()
        stats = new_stats()
        reported: list[str] = []

        for _ in range(MAX_TURNS):
            t0 = time.perf_counter()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                tools=tools,
                messages=messages,
            )
            stats["llm_time_s"] += time.perf_counter() - t0
            stats["turns"] += 1
            stats["input_tokens"] += response.usage.input_tokens
            stats["output_tokens"] += response.usage.output_tokens

            # Append the model's turn to the history
            messages.append({"role": "assistant", "content": response.content})

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if not tool_calls:
                break  # the model answered without calling a tool: episode over

            # Execute ALL tool calls of this turn (there may be several in parallel)
            results = []
            done = False
            for call in tool_calls:
                if call.name == "search":
                    stats["searches"] += 1
                    stats["queries"].append(call.input["query"])
                    out = self._tool_search(call.input["query"], seen)
                elif call.name == "read_doc":
                    stats["reads"] += 1
                    out = self._tool_read_doc(call.input["doc_id"])
                elif call.name == "report_ids":
                    reported = [str(d) for d in call.input.get("doc_ids", [])]
                    stats["reported"] = True
                    done = True
                    out = "Reported."
                elif handle_extra is not None:
                    if call.name == "message":
                        stats["messages"] += 1
                    out = handle_extra(call.name, call.input)
                else:
                    out = f"Unknown tool: {call.name}"
                if self.verbose:
                    print(f"  {tag}-> {call.name}({json.dumps(call.input)[:80]})")
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": out})

            if done:
                break

            # Inbox: teammates' messages are appended to the last tool result (as in SID-2)
            if inbox is not None:
                news = inbox()
                if news:
                    results[-1]["content"] += f"\n\n--- Messages from teammates ---\n{news}"

            messages.append({"role": "user", "content": results})

        clean: list[str] = []
        for d in reported:
            if d in self.corpus and d not in clean:
                clean.append(d)
        return clean, stats

    def pad_with_bm25(self, ranked: list[str], query: str, k: int, stats: dict) -> list[str]:
        """If there are fewer than k IDs, pad with BM25 results for the original query."""
        ranked = list(ranked)
        if len(ranked) < k:
            for d in self.bm25.search(query, k * 2):
                if d not in ranked:
                    ranked.append(d)
                    stats["padded"] += 1
                if len(ranked) >= k:
                    break
        return ranked[:k]

    def search(self, query: str, k: int = 10) -> list[str]:
        reported, stats = self.run_episode(query)
        ranked = self.pad_with_bm25(reported, query, k, stats)
        self.last_stats = stats
        return ranked
