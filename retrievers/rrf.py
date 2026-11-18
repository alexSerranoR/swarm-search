"""Reciprocal Rank Fusion. 👉 ESTO LO IMPLEMENTAS TÚ.

Comprueba con:  pytest tests/test_rrf.py -v
"""


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fusiona varios rankings en uno.

    Cada documento suma, por cada lista en la que aparece:  1 / (k + posición)
    con la posición empezando en 1. Si no aparece en una lista, esa lista no le suma nada.

    Args:
        rankings: lista de rankings; cada ranking es una lista de doc_ids ordenada.
        k:        constante de suavizado (60 es el valor estándar).

    Returns:
        Lista de doc_ids ordenada por puntuación RRF de mayor a menor.
        En caso de empate, cualquier orden vale (los tests no dependen de ello).

    Ejemplo:
        A = ["x", "y"], B = ["y"]
        y: 1/62 + 1/61   x: 1/61   ->  ["y", "x"]
    """
    # TODO: implementa
    raise NotImplementedError
