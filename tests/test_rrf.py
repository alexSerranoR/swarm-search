"""Tests de RRF. Valores calculados a mano con k=60."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrievers.rrf import rrf  # noqa: E402


def test_rrf_premia_el_consenso():
    # A = [x, z, y], B = [y, x]
    # x: 1/61 + 1/62 = 0.03252   y: 1/63 + 1/61 = 0.03227   z: 1/62 = 0.01613
    assert rrf([["x", "z", "y"], ["y", "x"]]) == ["x", "y", "z"]


def test_rrf_una_sola_lista_no_cambia_el_orden():
    assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]


def test_rrf_incluye_documentos_de_todas_las_listas():
    assert set(rrf([["a"], ["b"], ["c", "a"]])) == {"a", "b", "c"}


def test_rrf_el_que_aparece_en_mas_listas_gana():
    # "c" va el último en cada lista pero aparece en las 3
    result = rrf([["a", "c"], ["b", "c"], ["d", "c"]])
    assert result[0] == "c"


def test_rrf_k_pequeno_da_mas_peso_a_la_cima():
    # A=[x], B=[p, q, y], C=[r, s, y]
    # "x" va 1º en una lista; "y" va 3º en dos listas.
    # k=0 : x = 1/1  = 1.0     ; y = 1/3 + 1/3   = 0.667   -> x gana (k pequeño premia la cima)
    # k=60: x = 1/61 = 0.01639 ; y = 1/63 + 1/63 = 0.03175 -> y gana (k grande premia el consenso)
    rankings = [["x"], ["p", "q", "y"], ["r", "s", "y"]]
    assert rrf(rankings, k=0).index("x") < rrf(rankings, k=0).index("y")
    assert rrf(rankings, k=60).index("y") < rrf(rankings, k=60).index("x")
