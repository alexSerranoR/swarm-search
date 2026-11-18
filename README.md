# swarm-search

Replicación a pequeña escala de la idea de **SID-2 Swarm**: ¿coordinar varios agentes buscadores
gana a fusionar búsquedas independientes con RRF, con el mismo presupuesto?

Report original: https://www.sid.ai/research/sid-2-technical-report

## Fases

| Fase | Qué | Estado |
|---|---|---|
| 1 | Evals: `recall@k`, `NDCG@k` y RRF | 👉 **TÚ** (stubs en `evals/metrics.py` y `retrievers/rrf.py`) |
| 2 | Baseline RAG clásico (BM25 + embeddings) | pendiente |
| 3 | Un agente buscador con bucle de tools escrito a mano | pendiente |
| 4 | Swarm (fork / message / report_ids) vs N agentes + RRF | pendiente |
| 5 | Gráfica calidad vs coste/latencia y conclusiones | pendiente |

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/download.py          # descarga SciFact (BEIR) a data/scifact/
```

## Tu tarea de la fase 1

1. Abre `evals/metrics.py` e implementa `recall_at_k` y `ndcg_at_k` (los docstrings explican qué deben hacer).
2. Abre `retrievers/rrf.py` e implementa `rrf`.
3. Comprueba tu trabajo:

```bash
pytest -v
```

Los tests tienen valores **calculados a mano**. Cuando todos pasen en verde, la fase 1 está hecha.

4. Prueba el pipeline completo con un buscador aleatorio (debe dar casi 0; es el "suelo absoluto"):

```bash
python evals/run_eval.py --retriever random
```

## Reglas para aprender (no hacer trampas 😄)

- Escribe tú el código de los stubs. Pregunta a Claude conceptos o pide revisión, no el fichero.
- Si te atascas más de 20 min, pide una **pista**, no la solución.
- Antes de implementar NDCG, calcula a mano el ejemplo de `tests/test_metrics.py::test_ndcg_ejemplo_basico`.

## Estructura

```
data/        descarga y carga del dataset (hecho)
evals/       métricas (TÚ) y runner de evaluación (hecho)
retrievers/  RRF (TÚ), buscador aleatorio (hecho); aquí irá BM25 en la fase 2
agents/      fase 3 y 4
results/     aquí se guardan los resultados en JSON
tests/       tests de la fase 1
```
