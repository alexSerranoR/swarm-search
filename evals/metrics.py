"""Métricas de retrieval. 👉 ESTO LO IMPLEMENTAS TÚ.

Comprueba con:  pytest tests/test_metrics.py -v
"""
import math  # te hará falta para log2


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Fracción de los documentos relevantes que aparecen en el top-k.

    Args:
        ranked:   IDs de documentos ordenados del más al menos relevante (según el buscador).
        relevant: {doc_id: grado} con los documentos relevantes de verdad (grado > 0).
        k:        cuántas posiciones del ranking miramos.

    Returns:
        (nº de relevantes dentro de ranked[:k]) / (nº total de relevantes).
        Si no hay relevantes, devuelve 0.0.

    Ejemplo:
        ranked = ["a", "b", "c"], relevant = {"a": 1, "c": 1, "e": 1}, k = 2
        -> en el top-2 solo está "a" -> 1/3
    """
    # TODO: implementa
    raise NotImplementedError


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain en el top-k.

    Idea: un documento relevante vale más cuanto más arriba aparezca.

        DCG@k  = suma, para cada posición i = 1..k, de  grado(ranked[i]) / log2(i + 1)
                 (grado = 0 si el documento no es relevante)
        IDCG@k = el DCG del ranking PERFECTO: los relevantes ordenados por grado
                 de mayor a menor, cortado a k posiciones
        NDCG@k = DCG@k / IDCG@k        (0.0 si IDCG es 0)

    Ojo: las posiciones empiezan en 1 en la fórmula, pero los índices de Python en 0.

    Pista para comprobarte: en la posición 1 el descuento es log2(2) = 1,
    en la posición 3 es log2(4) = 2.
    """
    # TODO: implementa
    raise NotImplementedError
