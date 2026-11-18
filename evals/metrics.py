"""Métricas de retrieval. 👉 ESTO LO IMPLEMENTAS TÚ.

Comprueba con:  pytest tests/test_metrics.py -v
"""
import math  # te hará falta para log2


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    counter = 0
    for doc in ranked[:k]:
        if doc in relevant:
            counter += 1
    return counter / len(relevant)



def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    # Paso 1: DCG, los puntos de tu buscador
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        grado = relevant.get(doc, 0)
        dcg += grado / math.log2(i + 2)

    # Paso 2: IDCG, los puntos del ranking perfecto
    grados_ideales = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for i, grado in enumerate(grados_ideales):
        idcg += grado / math.log2(i + 2)

    # Paso 3: dividir, sin dividir entre 0
    if idcg == 0:
        return 0.0
    return dcg / idcg
