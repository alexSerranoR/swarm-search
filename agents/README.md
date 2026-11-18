# agents/

- `search_agent.py`: a single retrieval agent. `run_episode()` is the whole harness: model call →
  execute all tool calls of the turn → append results (plus teammates' messages, if any) → repeat until
  `report_ids` or the turn limit.
- `swarm.py`: `RRFEnsemble` (N independent agents + RRF) and `Swarm` (N agents with a shared message
  board + RRF). Both reuse `run_episode()`; the only difference is the `message` tool and the inbox.

Every agent exposes the same interface as any retriever: `search(query, k) -> list[str]`.
